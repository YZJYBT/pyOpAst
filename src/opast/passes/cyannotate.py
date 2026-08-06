"""Cython C-type annotation (opt-in ``--cython``): hand Cython the types
opast can prove, and nothing else.

Cython's ceiling is type information, not code generation.  In pure-Python
mode it only has its own "safe" inference, which is both weaker than opast's
analysis (it cannot bound a value's range) and, on a compiler whose ``long``
is 32 bits, not actually safe: stock Cython silently miscompiles
``[j * j % 97 for j in range(0, 400000, 4)]`` because it infers a C ``long``
for ``j`` and the product wraps.  opast's interval analysis answers exactly
that question -- ``j * j`` is in ``(0, 159996800016)``, which needs 64 bits
-- so this pass emits a type only where the proof carries it.

What it emits, per function::

    @cython.locals(i=cython.longlong, acc=cython.longlong, x=cython.double)
    def work(...):

The decorator form is deliberate: it leaves the signature and
``__annotations__`` byte-identical, so nothing Python-visible changes, while
Cython reads it exactly like an inline annotation.  A guarded ``import
cython`` with a tiny inline shim keeps the output runnable on an interpreter
that has no Cython installed -- the file is still plain Python, which is the
whole output contract; compiling it is the user's separate step.

What is typed:

* ``cython.double`` for names **proven float**.  A Python float *is* a C
  double -- the representation is identical, so there is no range question
  to answer.
* ``cython.longlong`` for names **proven int whose interval fits int64**.
  An int without a proven bound is left alone: Python ints are arbitrary
  precision and a C integer would silently wrap.  This is why an
  ``int``-annotated *parameter* is normally not typed -- the annotation says
  "int", not "small" -- while a ``range()`` counter or a ``% M`` accumulator
  usually is.

Structural rejections (per name, any doubt -> not typed):

* read anywhere before it is **definitely bound** -- a C variable has no
  unbound state, so ``UnboundLocalError`` would silently become 0.0/0.  A
  name qualifies when it is a parameter, when a top-level assignment binds
  it before every use, or when it is a top-level ``for`` target used only
  inside that loop (the zero-iteration case never escapes);
* captured by a nested function, lambda, class or comprehension (Cython
  cannot close over a typed local);
* ``del``eted, declared ``global``/``nonlocal``, or compared with
  ``is``/``is not`` (a C round-trip need not preserve object identity).

Whole functions are skipped when they are async, generators, already
decorated, nested inside another function, or live in a dynamic region.
"""

from __future__ import annotations

import ast

from ..analysis import (
    AnnotationTrust,
    all_bound_names,
    binding_names,
    builtin_gate,
    fresh_container_names,
    infer_float_names,
    infer_int_names,
    infer_int_ranges,
    param_names,
)
from ..safety import iter_region, region_is_dynamic, tree_has_dynamic
from .base import ScopedTransformer
from .licm import container_gate

#: The C types we emit.  ``longlong`` rather than ``long`` on purpose: MSVC's
#: ``long`` is 32 bits, which is where stock Cython's inference goes wrong.
_INT_TYPE = "longlong"
_FLOAT_TYPE = "double"

_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1

#: Name the shim binds at module level; the pass bails out if it is taken.
_SHIM_NAME = "cython"

_SHIM_SOURCE = """\
try:
    import cython
except ImportError:
    class cython:
        longlong = int
        double = float

        @staticmethod
        def locals(**_opast_types):
            def _opast_identity(_opast_fn):
                return _opast_fn
            return _opast_identity

        @staticmethod
        def infer_types(_opast_on):
            def _opast_identity(_opast_fn):
                return _opast_fn
            return _opast_identity
"""


def _fits_int64(interval) -> bool:
    """The interval is known and entirely inside a C ``long long``."""
    if not isinstance(interval, tuple) or len(interval) != 2:
        return False
    low, high = interval
    if low is None or high is None:  # open-ended (NEG_INF / POS_INF)
        return False
    return _INT64_MIN <= low and high <= _INT64_MAX


def _is_generator(func: ast.AST) -> bool:
    return any(
        isinstance(n, (ast.Yield, ast.YieldFrom)) for n in iter_region(func)
    )


def _nested_scopes(func: ast.AST):
    """Nested scopes of *func* -- everything Cython cannot share a typed
    local with."""
    for node in ast.walk(func):
        if node is func:
            continue
        if isinstance(node, (
            ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef,
            ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp,
        )):
            yield node


def _unsafe_names(func: ast.AST) -> set[str]:
    """Names disqualified by how they are *used*, whatever their type."""
    bad: set[str] = set()
    for scope in _nested_scopes(func):
        for node in ast.walk(scope):
            if isinstance(node, ast.Name):
                bad.add(node.id)
            elif isinstance(node, ast.arg):
                bad.add(node.arg)
    for node in iter_region(func):
        if isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bad.add(target.id)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bad.update(node.names)
        elif isinstance(node, ast.Compare):
            if any(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
                for operand in (node.left, *node.comparators):
                    if isinstance(operand, ast.Name):
                        bad.add(operand.id)
    return bad


def _arithmetic_operands(func: ast.AST) -> set[str]:
    """Every name read as an *operand* of arithmetic in *func*.

    Typing only part of an expression is not free: each typed value feeding
    an untyped operand is boxed back into a Python object and unboxed again.
    Measured, that costs more than the typing buys (0.72x on a loop whose
    counter was typed and whose accumulator was not), so the pass types a
    function only when this whole set is covered.  Call *targets* are not
    operands -- the call's result is, and it is a Python object whichever
    name produced it, so a call in an arithmetic expression disqualifies the
    function through its own name anyway.
    """
    called: set[int] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(id(node.func))
    names: set[str] = set()
    for node in ast.walk(func):
        if not isinstance(node, (ast.BinOp, ast.UnaryOp, ast.AugAssign, ast.Compare)):
            continue
        for sub in ast.walk(node):
            if (
                isinstance(sub, ast.Name)
                and isinstance(sub.ctx, ast.Load)
                and id(sub) not in called
            ):
                names.add(sub.id)
    return names


def _stmt_region(stmt: ast.stmt):
    """*stmt*'s own region -- like :func:`iter_region` but for a single
    statement, so nested scopes are yielded without being entered."""
    return iter_region(ast.Module(body=[stmt], type_ignores=[]))


def _definitely_bound(func: ast.AST) -> set[str]:
    """Names that provably hold a value at every point they are read.

    Typing a name that can be read while unbound would turn
    ``UnboundLocalError`` into a silent zero, so the rule is deliberately
    coarse: parameters, top-level assignments that precede every use, and
    top-level ``for`` targets whose uses never leave the loop.
    """
    ok: set[str] = set(param_names(func))
    body = list(func.body)

    first_bind: dict[str, int] = {}
    bind_stmt: dict[str, ast.stmt] = {}
    uses: dict[str, set[int]] = {}
    for index, stmt in enumerate(body):
        for node in _stmt_region(stmt):
            for name in binding_names(node):
                if name not in first_bind:
                    first_bind[name] = index
                    bind_stmt[name] = stmt
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                uses.setdefault(node.id, set()).add(index)

    for name, index in first_bind.items():
        if name in ok:
            continue
        stmt = bind_stmt[name]
        seen = uses.get(name, set())
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            # A plain top-level binding dominates everything after it.  The
            # binding statement itself may read the name only on its right
            # hand side, which would be a read-before-binding.
            targets = (
                stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            )
            if not any(
                isinstance(t, ast.Name) and t.id == name for t in targets
            ):
                continue
            value_reads = {
                n.id
                for n in ast.walk(stmt.value)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
            } if stmt.value is not None else set()
            if name in value_reads:
                continue
            if all(use >= index for use in seen):
                ok.add(name)
        elif isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name):
            # The loop target only exists while the loop runs: a zero-trip
            # loop leaves it unbound, so uses must stay inside the loop.
            if stmt.target.id == name and seen <= {index}:
                ok.add(name)
    return ok


class CythonAnnotation(ScopedTransformer):
    """One-shot pass: attach ``@cython.locals(...)`` where types are proven."""

    name = "cython-annotate"

    def run(self, tree: ast.Module) -> ast.Module:
        if tree_has_dynamic(tree):
            self.skipped_scopes += 1
            return tree
        # The shim binds the name ``cython``; if the module already uses it
        # (including an existing ``import cython``) leave everything alone
        # rather than fight over the binding.
        if _SHIM_NAME in all_bound_names(tree):
            return tree
        self._trust = AnnotationTrust(
            tree if "annotations" in self.aggressive else None
        )
        self._len_ok = container_gate(tree)
        self._range_ok = builtin_gate(tree, "range")

        # Decide the types first: the eligibility rules look at a function's
        # *own* decorators, which the directive below would otherwise mask.
        typed: list[tuple[ast.FunctionDef, dict[str, str]]] = []
        for parent in self._function_owners(tree):
            for stmt in parent.body:
                if isinstance(stmt, ast.FunctionDef):
                    types = self._types_for(stmt)
                    if types:
                        typed.append((stmt, types))

        annotated = 0
        # Disarm Cython's own inference everywhere.  It is the part of stock
        # Cython that is not semantics-preserving: it silently picks a C
        # ``long`` (32 bits under MSVC) for a comprehension variable whose
        # product overflows, and hands an unbound local the C default 0
        # instead of raising UnboundLocalError.  From here the types are
        # opast's to prove -- which is the whole point of the flag.
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node.decorator_list.append(_directive("infer_types", False))
                annotated += 1
        for stmt, types in typed:
            stmt.decorator_list.insert(0, _locals_decorator(types))
            annotated += 1
        if annotated:
            at = _shim_position(tree)
            tree.body[at:at] = ast.parse(_SHIM_SOURCE).body
            self.changes += annotated
        return tree

    @staticmethod
    def _function_owners(tree: ast.Module):
        """Module and class bodies -- functions nested inside a *function*
        are skipped: their locals may be cells for an inner scope."""
        yield tree
        for stmt in tree.body:
            if isinstance(stmt, ast.ClassDef) and not stmt.decorator_list:
                yield stmt

    def _types_for(self, func: ast.FunctionDef) -> dict[str, str]:
        if func.decorator_list or _is_generator(func) or region_is_dynamic(func):
            return {}
        containers = (
            fresh_container_names(func) if self._len_ok else frozenset()
        )
        int_trusted = self._trust.ints(func)
        ints = infer_int_names(
            func,
            containers,
            range_ok=self._range_ok,
            trusted=int_trusted,
            typed_calls=self._trust.int_returns,
        )
        floats = infer_float_names(
            func,
            ints,
            containers,
            trusted=self._trust.floats(func),
            typed_calls=self._trust.float_returns,
        )
        if not ints and not floats:
            return {}
        ranges = infer_int_ranges(
            func, ints, containers, range_ok=self._range_ok,
            trusted=int_trusted,
        )
        allowed = _definitely_bound(func) - _unsafe_names(func)
        types = {
            name: _FLOAT_TYPE for name in sorted(floats) if name in allowed
        }
        types.update({
            name: _INT_TYPE
            for name in sorted(ints)
            if name in allowed and _fits_int64(ranges.get(name))
        })
        # All-or-nothing: a typed value meeting an untyped one pays for a
        # round trip through PyObject, which measured *worse* than leaving
        # the function alone (see _arithmetic_operands).
        if _arithmetic_operands(func) - set(types):
            return {}
        return dict(sorted(types.items()))


def _directive(name: str, value) -> ast.expr:
    """``cython.<name>(<value>)`` -- a Cython compiler directive in the
    pure-Python decorator form (a no-op decorator when merely run)."""
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id=_SHIM_NAME, ctx=ast.Load()),
            attr=name,
            ctx=ast.Load(),
        ),
        args=[ast.Constant(value=value)],
        keywords=[],
    )


def _locals_decorator(types: dict[str, str]) -> ast.expr:
    """``cython.locals(name=cython.<type>, ...)``"""
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id=_SHIM_NAME, ctx=ast.Load()),
            attr="locals",
            ctx=ast.Load(),
        ),
        args=[],
        keywords=[
            ast.keyword(
                arg=name,
                value=ast.Attribute(
                    value=ast.Name(id=_SHIM_NAME, ctx=ast.Load()),
                    attr=ctype,
                    ctx=ast.Load(),
                ),
            )
            for name, ctype in types.items()
        ],
    )


def _shim_position(tree: ast.Module) -> int:
    """Index for the shim: after the docstring and any ``__future__``
    imports, which must stay first."""
    index = 0
    body = tree.body
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        index = 1
    while (
        index < len(body)
        and isinstance(body[index], ast.ImportFrom)
        and body[index].module == "__future__"
    ):
        index += 1
    return index
