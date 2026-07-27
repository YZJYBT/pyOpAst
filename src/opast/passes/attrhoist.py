"""Loop-invariant attribute hoisting (aggressive option ``attrs``).

Hoists attribute chains that a loop reads every iteration into a pre-loop
temporary::

    for i in range(n):                 _opast_attr_0 = math.floor
        t += math.floor(i)      ->     for i in range(n):
                                           t += _opast_attr_0(i)

Measured on CPython 3.14 (which already caches ``LOAD_ATTR`` in the
adaptive interpreter): dotted *module* access in a call position gains
**1.77x** -- the per-iteration ``LOAD_GLOBAL`` + ``LOAD_ATTR`` dict chain
becomes one ``LOAD_FAST`` -- and plain attribute *value* reads gain about
1.24x.  Crucially, the folklore bound-method caching (``st = o.step``)
is an **anti-optimization** on 3.14, measured at 0.90x: ``o.step(t)``
takes the specialized method-call path that never materialises a bound
method, while calling the hoisted object goes through the slower generic
path.  Hence the split rule:

* an attribute chain in **call position** (``chain(...)``) is hoisted only
  when its root name is bound by a plain top-level ``import`` -- then the
  attribute is a module member, not a method, and the call path is
  unchanged;
* a chain in **value position** is hoisted for any eligible root.

Eligibility (all required):

* the root is a plain name, not rebound anywhere in the loop's subtree,
  and either definitely bound before the loop (LICM's dominance scan,
  parameters included) or a stable module import (bound exactly once, by
  a direct top-level ``import``, never ``global``-declared, imported
  before the enclosing function's ``def``);
* no prefix of the chain is assigned or deleted anywhere in the loop
  (``obj.x = ...`` kills hoists of ``obj.x`` *and* ``obj.x.y``; mutating
  method calls fall under the option's assumption instead);
* the chain appears in the loop body or a ``while`` test (a ``for``
  loop's iterable is evaluated once and left alone), outside nested
  scopes.

Evaluation order is preserved the hard way.  A ``for`` loop does two
things before its first body iteration -- evaluate the iterable, then
call its ``__iter__`` -- and either may be exactly what sets the
attribute the body reads (``prepare(c)``, or a side-effecting user
``__iter__``).  Argument evaluation can be ordered with a temp, but
``for`` always re-invokes ``__iter__`` after the temps, and
pre-acquiring the iterator would call a user ``__iter__`` twice (the
protocol requires it to return self, not to be side-effect-free).  Body
hoisting is therefore allowed only when the iterable's ``__iter__`` is
*provably pure C*: a constant, a fresh-container name (escape analysis),
a display/comprehension/generator expression, or a call to an unshadowed
builtin constructor (``range``/``enumerate``/``zip``/...); those with
effectful *evaluation* get an expression temp before the attribute
temps.  Anything else -- bare names of unknown type, user calls, every
``async for`` -- rejects the loop.  A ``while`` test needs no such
treatment: it runs as part of every iteration, so a test that mutates
the hoisted attribute already falls under the stated assumption.

Why this is aggressive rather than proof-backed: an attribute lookup may
run arbitrary code (``__getattr__``, properties, descriptors), and the
loop body may rebind the attribute through a call this pass cannot see.
Hoisting changes how often and when the lookup runs -- once, before the
loop -- so the stated assumption is that hoisted attributes are *stable*:
side-effect-free to read and not rebound while the loop runs.  A
zero-iteration loop additionally performs the one lookup the original
never did.  Function scopes only; module-level loops become functions
under ``loop-state`` outlining first, and this pass picks them up on the
next fixpoint iteration.
"""

from __future__ import annotations

import ast

from ..analysis import (
    binding_names,
    builtin_gate,
    fresh_container_names,
    param_names,
)
from ..safety import iter_region, region_is_dynamic
from .base import ScopedTransformer
from .licm import (
    _LOOPS,
    container_gate,
    definite_bindings,
    subtree_bindings,
    unbound_risk_names,
)

_TEMP_PREFIX = "_opast_attr"

#: Builtin constructors whose result is a C object with a side-effect-free
#: ``__iter__`` (each must additionally be unshadowed module-wide).
#: ``iter`` itself is deliberately absent: ``for x in iter(xs)`` calls the
#: user ``__iter__`` twice already, and interposing temps between those two
#: calls would reorder against a side-effecting second call.
_PURE_ITER_CTORS = frozenset(
    {"range", "enumerate", "zip", "map", "filter", "reversed", "sorted",
     "list", "tuple", "set", "frozenset", "dict"}
)

#: Display/comprehension expressions whose value likewise has a pure
#: ``__iter__`` (fresh builtin containers, or a generator, whose
#: ``__iter__`` is identity by construction).
_PURE_ITER_DISPLAYS = (
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _chain_root(node: ast.Attribute):
    """The base ``Name`` of a pure attribute chain, else None."""
    cursor = node
    while isinstance(cursor, ast.Attribute):
        if not isinstance(cursor.ctx, ast.Load):
            return None
        cursor = cursor.value
    if isinstance(cursor, ast.Name) and isinstance(cursor.ctx, ast.Load):
        return cursor.id
    return None


def _chain_dumps_with_prefixes(node: ast.Attribute):
    """Dumps of the chain and every attribute prefix (``a.b.c`` yields the
    dumps of ``a.b.c`` and ``a.b``)."""
    out = []
    cursor = node
    while isinstance(cursor, ast.Attribute):
        out.append(ast.dump(cursor))
        cursor = cursor.value
    return out


def _stable_imports(tree: ast.Module) -> dict[str, int]:
    """Names bound exactly once module-wide, by a direct top-level plain
    ``import`` -- guaranteed modules.  Maps name -> line of the import."""
    counts: dict[str, int] = {}
    for node in iter_region(tree):
        for name in binding_names(node):
            counts[name] = counts.get(name, 0) + 1
    declared: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Global):
            declared.update(node.names)
    stable: dict[str, int] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                bound = alias.asname or alias.name.split(".")[0]
                if counts.get(bound, 0) == 1 and bound not in declared:
                    stable[bound] = stmt.lineno
    return stable


class _Hoister(ast.NodeTransformer):
    """Replaces eligible maximal attribute chains with temp names."""

    def __init__(self, owner, eligible_roots, module_roots, killed) -> None:
        self.owner = owner
        self.eligible_roots = eligible_roots
        self.module_roots = module_roots
        self.killed = killed  # dumps of written chains (and their reads)
        self.temps: dict[str, tuple[str, ast.expr]] = {}

    def _skip(self, node: ast.AST) -> ast.AST:
        return node

    visit_FunctionDef = _skip
    visit_AsyncFunctionDef = _skip
    visit_Lambda = _skip
    visit_ClassDef = _skip

    def _try(self, node: ast.Attribute, call_position: bool):
        root = _chain_root(node)
        if root is None:
            return None
        if call_position:
            if root not in self.module_roots:
                return None  # bound-method hoisting is slower on 3.14
        elif root not in self.eligible_roots:
            return None
        dumps = _chain_dumps_with_prefixes(node)
        if any(d in self.killed for d in dumps):
            return None
        key = dumps[0]
        entry = self.temps.get(key)
        if entry is None:
            entry = (self.owner._fresh_name(), node)
            self.temps[key] = entry
        return ast.copy_location(ast.Name(id=entry[0], ctx=ast.Load()), node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Attribute):
            replaced = self._try(node.func, call_position=True)
            if replaced is not None:
                node.func = replaced
            else:
                node.func = self.generic_visit(node.func)
            node.args = [self.visit(a) for a in node.args]
            for kw in node.keywords:
                kw.value = self.visit(kw.value)
            return node
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if isinstance(node.ctx, ast.Load):
            replaced = self._try(node, call_position=False)
            if replaced is not None:
                return replaced
        return self.generic_visit(node)


class AttributeHoisting(ScopedTransformer):
    name = "attr-hoist"

    def run(self, tree: ast.Module) -> ast.Module:
        if "attrs" not in self.aggressive:
            return tree
        if region_is_dynamic(tree):
            self.skipped_scopes += 1
            return tree
        self._imports = _stable_imports(tree)
        # For-loop body hoists need the iterable's __iter__ to be provably
        # pure (see _hoist_loop); the proof leans on unshadowed builtin
        # constructors and on the fresh-container escape analysis.
        self._pure_ctors = {
            name for name in _PURE_ITER_CTORS if builtin_gate(tree, name)
        }
        self._container_gate = container_gate(tree)
        self._counter = 0
        for n in ast.walk(tree):
            ident = n.id if isinstance(n, ast.Name) else (
                n.arg if isinstance(n, ast.arg) else None
            )
            if ident and ident.startswith(_TEMP_PREFIX + "_"):
                suffix = ident[len(_TEMP_PREFIX) + 1:]
                if suffix.isdigit():
                    self._counter = max(self._counter, int(suffix) + 1)
        return self.visit(tree)

    def _fresh_name(self) -> str:
        name = f"{_TEMP_PREFIX}_{self._counter}"
        self._counter += 1
        return name

    def _visit_function(self, node: ast.AST) -> ast.AST:
        if region_is_dynamic(node):
            self.skipped_scopes += 1
            return node
        node = self.generic_visit(node)  # nested functions first
        # Module imports readable from this function: not shadowed by any
        # local binding, and imported before the function's definition.
        func_bound = set(param_names(node))
        for n in iter_region(node):
            func_bound.update(binding_names(n))
        self._func_imports = {
            name
            for name, line in self._imports.items()
            if name not in func_bound and line < node.lineno
        }
        self._fresh = (
            fresh_container_names(node)
            if self._container_gate
            else frozenset()
        )
        node.body = self._process_block(node.body, set(param_names(node)))
        return node

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    # -- dominance-aware block walk (mirrors LICM) --------------------------
    def _process_block(self, stmts, bound: set[str]) -> list:
        bound = set(bound)
        out = []
        for stmt in stmts:
            if isinstance(stmt, _LOOPS):
                temps = self._hoist_loop(stmt, bound)
                out.extend(temps)
                for t in temps:
                    bound.add(t.targets[0].id)
                inner = bound - unbound_risk_names(stmt)
                if isinstance(stmt, (ast.For, ast.AsyncFor)):
                    inner |= {
                        n.id
                        for n in ast.walk(stmt.target)
                        if isinstance(n, ast.Name)
                        and isinstance(n.ctx, ast.Store)
                    }
                stmt.body = self._process_block(stmt.body, inner)
            elif isinstance(stmt, ast.If):
                stmt.body = self._process_block(stmt.body, bound)
                stmt.orelse = self._process_block(stmt.orelse, bound)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                stmt.body = self._process_block(stmt.body, bound)
            elif isinstance(stmt, ast.Try) or (
                hasattr(ast, "TryStar") and isinstance(stmt, ast.TryStar)
            ):
                stmt.body = self._process_block(stmt.body, bound)
                for handler in stmt.handlers:
                    handler.body = self._process_block(handler.body, bound)
                stmt.orelse = self._process_block(stmt.orelse, bound)
                stmt.finalbody = self._process_block(stmt.finalbody, bound)
            elif isinstance(stmt, ast.Match):
                for case in stmt.cases:
                    case.body = self._process_block(case.body, bound)
            out.append(stmt)
            bound |= definite_bindings(stmt)
            bound -= unbound_risk_names(stmt)
        return out

    def _hoist_loop(self, loop, bound: set[str]) -> list:
        loop_bound = subtree_bindings(loop)
        eligible = (bound | self._func_imports) - loop_bound
        module_roots = self._func_imports - loop_bound
        if not eligible:
            return []
        # Written/deleted chains, re-dumped with a Load ctx so they compare
        # equal to read occurrences.  A read is rejected when any of its
        # prefixes matches (write obj.x kills reads of obj.x and obj.x.y).
        killed = {
            ast.dump(_as_load(n))
            for n in ast.walk(loop)
            if isinstance(n, ast.Attribute) and not isinstance(n.ctx, ast.Load)
        }
        # A for loop does two things before its first body iteration --
        # evaluate the iterable, then call its __iter__ -- and either may
        # be what sets the attribute the body reads.  Argument-evaluation
        # order is preserved with an expression temp, but there is NO way
        # to stop ``for`` from calling __iter__ after the temps (and
        # pre-acquiring via iter() would call a user __iter__ twice, which
        # the protocol does not promise to tolerate).  So body hoisting is
        # allowed only when the iterable's __iter__ is provably pure C:
        # a constant, a fresh-container name, a display/comprehension, or
        # a call to an unshadowed builtin constructor.  Everything else
        # rejects the loop.  Async iterables have no provable case.
        iter_temp_value = None
        if isinstance(loop, ast.AsyncFor):
            return []
        if isinstance(loop, ast.For):
            it = loop.iter
            if isinstance(it, ast.Constant) and isinstance(
                it.value, (str, bytes)
            ):
                # Actually iterable C types only: ``for i in 1`` must keep
                # raising TypeError *before* any hoisted lookup can raise
                # its own error, so non-iterable constants are rejected.
                pass
            elif (
                isinstance(it, ast.Name)
                and isinstance(it.ctx, ast.Load)
                and it.id in self._fresh
            ):
                pass  # name load is effect-free; fresh container iter is C
            elif isinstance(it, _PURE_ITER_DISPLAYS):
                iter_temp_value = it  # evaluation may have effects: order it
            elif (
                isinstance(it, ast.Call)
                and isinstance(it.func, ast.Name)
                and it.func.id in self._pure_ctors
            ):
                iter_temp_value = it
            else:
                return []

        hoister = _Hoister(self, eligible, module_roots, killed)
        if isinstance(loop, ast.While):
            loop.test = hoister.visit(loop.test)
        loop.body = [hoister.visit(s) for s in loop.body]
        if not hoister.temps:
            return []
        temps = []
        if iter_temp_value is not None:
            iter_name = self._fresh_name()
            target = ast.copy_location(
                ast.Name(id=iter_name, ctx=ast.Store()), loop
            )
            temps.append(
                ast.copy_location(
                    ast.Assign(targets=[target], value=iter_temp_value), loop
                )
            )
            loop.iter = ast.copy_location(
                ast.Name(id=iter_name, ctx=ast.Load()), loop
            )
        for name, expr in hoister.temps.values():
            target = ast.copy_location(ast.Name(id=name, ctx=ast.Store()), loop)
            temps.append(
                ast.copy_location(
                    ast.Assign(targets=[target], value=expr), loop
                )
            )
            self.changes += 1
        return temps


def _as_load(attr: ast.Attribute) -> ast.Attribute:
    """A copy of *attr* with Load context, for dump comparison against the
    read occurrences."""
    clone = ast.Attribute(value=attr.value, attr=attr.attr, ctx=ast.Load())
    return clone
