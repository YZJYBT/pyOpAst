"""numpy vectorization extraction (aggressive ``numpy``).

One-shot pass executed once after the optimization fixpoint (before the
jit pass).  Two loop shapes are extracted into generated module-level
helper pairs -- an exact-Python version and a numpy version -- joined by
a **self-contained dispatcher embedded into the output** (the optimized
source keeps zero runtime dependencies: no opast, and numpy strictly
optional -- ``-O3 --disable jit`` builds stay dependency-free):

* **maps**: ``acc = [ELT for v in xs]`` (the shape ``loop-to-comp``
  produces) becomes ``acc = _opast_vecN(xs, <invariants...>)``;
* **reductions**: ``for v in xs: total += ELT`` (or ``total = total +
  ELT``) becomes ``total = _opast_vecN(total, xs, <invariants...>)``.

The dispatcher runs the numpy path under ``errstate(divide/invalid=
'raise')``, verifies the first call against the exact Python path, and
permanently falls back on any exception, dtype rejection, or mismatch --
so numpy being absent, a heterogeneous list, a zero divisor, or a NaN all
degrade to the original semantics (including the original exception).
What genuinely may differ is the option's stated bet: intermediates are
fixed-width int64/float64 (large ints wrap instead of growing) and a
float reduction is reassociated into one partial sum.

Static gates (any doubt -> no rewrite):

* the iterable is a plain name proven fresh and re-iterable
  (:func:`opast.passes.fission._reiterable_names` -- the first-call
  verification runs *both* paths, so a one-shot iterator would be drained)
  or a direct ``range(...)`` call (numpy path uses ``arange``; invalid
  range arguments raise in the Python path *first*, so ``arange``'s laxer
  typing is never observed);
* ELT is arithmetic (``+ - * / // % **``, unary ``+/-``) over the loop
  target, invariant names (passed as call arguments, so evaluated once --
  sound because a comprehension/single-statement loop body gives them no
  chance to be rebound mid-loop) and numeric constants; it references the
  target and contains at least one operation;
* element dtype is gated at runtime: ``kind == 'i'`` normally (a mixed
  int/float list would silently promote ints to floats), relaxed to
  ``kind in 'if'`` when ELT provably yields floats for any numeric input
  (a ``/`` on every path or a float constant operand), where the promotion
  is exactly what Python would do anyway; empty sequences short-circuit to
  the Python path before the dtype gate;
* reductions only: function scope, not under ``try``/``with`` (the
  one-shot form leaves the accumulator untouched on a mid-loop exception,
  observable only by a handler), accumulator is a plain local (not
  ``global``/``nonlocal``-declared, not read inside ELT), the loop has no
  ``else``, and the loop target is dead afterwards (the rewrite removes
  its binding).

Generated names use the reserved ``_opast_vec`` prefix; the jit pass
skips them (the numpy version could never compile, and decorating the
Python fallback would stack dispatchers).
"""

from __future__ import annotations

import ast
import copy

from ..analysis import all_bound_names, fresh_container_names
from ..safety import region_is_dynamic, tree_has_dynamic
from .base import ScopedTransformer
from .fission import _reiterable_names
from .licm import container_gate

_PREFIX = "_opast_vec"

#: Self-contained runtime injected once per module that gets any rewrite:
#: the optimized output must run with **zero** dependencies -- no opast, no
#: numpy (``--disable jit`` aggressive builds are documented dependency-
#: free, and vectorization must not break that).  Mirrors
#: ``jitsupport``'s dispatch discipline: first-call verification against
#: the exact Python result, per-site permanent fallback on any exception,
#: mismatch, or absent numpy; the numpy path runs under
#: ``errstate(divide/invalid='raise')`` so a zero divisor becomes a
#: fallback that re-runs (and re-raises) the exact original computation.
_RUNTIME = '''
def _opast_vec_match(_e, _g):
    if isinstance(_e, (list, tuple)) and isinstance(_g, (list, tuple)):
        return len(_e) == len(_g) and all(
            _opast_vec_match(_a, _b) for _a, _b in zip(_e, _g)
        )
    if isinstance(_e, float) or isinstance(_g, float):
        _e, _g = float(_e), float(_g)
        if _e != _e and _g != _g:
            return True
        return _e == _g or abs(_e - _g) <= 1e-12 * max(abs(_e), abs(_g), 1.0)
    return type(_e) is type(_g) and _e == _g

def _opast_vector_dispatch(_py, _np_fn):
    _state = []
    _verify = not __import__("os").environ.get("OPAST_JIT_NO_VERIFY")

    def _call(*_args):
        if _state and _state[0] == "python":
            return _py(*_args)
        try:
            import numpy as _np
        except Exception:
            _state[:] = ["python"]
            return _py(*_args)
        if _state and _state[0] == "numpy":
            try:
                with _np.errstate(divide="raise", invalid="raise"):
                    return _np_fn(_np, *_args)
            except Exception:
                _state[:] = ["python"]
                return _py(*_args)
        _expected = _py(*_args)
        if not _verify:
            _state[:] = ["numpy"]
            return _expected
        try:
            with _np.errstate(divide="raise", invalid="raise"):
                _got = _np_fn(_np, *_args)
        except Exception:
            _state[:] = ["python"]
            return _expected
        _state[:] = ["numpy" if _opast_vec_match(_expected, _got) else "python"]
        return _expected

    return _call
'''

_VEC_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_VEC_UNARY = (ast.USub, ast.UAdd)


def _worth_it(iter_kind, check: "_EltCheck") -> bool:
    """Measured activation rule (CPython 3.14, best-of-7): range sources
    win at any element weight (arange is free: 1.5x light maps, 8-24x
    reductions), while list sources pay an asarray and tolist round trip
    that only ≥4-operation elements out-earn (~2-3x; a 2-op element
    measured 0.61-0.92x).  Anything below stays a plain loop."""
    if not check.op_count:
        return False
    return iter_kind[0] == "range" or check.op_count >= 4


def _forces_float(node: ast.AST) -> bool:
    """True when ELT yields a float for *any* numeric operands, making the
    int->float promotion of a mixed input list unobservable."""
    if isinstance(node, ast.Constant):
        return type(node.value) is float
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, ast.Div):
            return True
        return _forces_float(node.left) or _forces_float(node.right)
    if isinstance(node, ast.UnaryOp):
        return _forces_float(node.operand)
    return False


class _EltCheck:
    """Grammar check collecting invariant names."""

    def __init__(self, target: str, forbidden: frozenset[str]) -> None:
        self.target = target
        self.forbidden = forbidden
        self.invariants: set[str] = set()
        self.uses_target = False
        self.op_count = 0

    def ok(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Constant):
            return type(node.value) in (int, float, bool)
        if isinstance(node, ast.Name):
            if not isinstance(node.ctx, ast.Load) or node.id in self.forbidden:
                return False
            if node.id == self.target:
                self.uses_target = True
            else:
                self.invariants.add(node.id)
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, _VEC_BINOPS):
            self.op_count += 1
            return self.ok(node.left) and self.ok(node.right)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, _VEC_UNARY):
            self.op_count += 1
            return self.ok(node.operand)
        return False


def _rename_target(elt: ast.AST, target: str, replacement: str) -> ast.AST:
    class _R(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.AST:
            if node.id == target and isinstance(node.ctx, ast.Load):
                return ast.copy_location(
                    ast.Name(id=replacement, ctx=ast.Load()), node
                )
            return node

    return _R().visit(copy.deepcopy(elt))


def _range_args(node: ast.AST):
    """Argument expressions of a plain ``range(...)`` call, else None."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
        and not node.keywords
        and 1 <= len(node.args) <= 3
        and not any(isinstance(a, ast.Starred) for a in node.args)
    ):
        return list(node.args)
    return None


class VectorizeLoops(ScopedTransformer):
    name = "vectorize"

    def run(self, tree: ast.Module) -> ast.Module:
        if tree_has_dynamic(tree):
            self.skipped_scopes += 1
            return tree
        self._container_ok = container_gate(tree)
        self._range_ok = "range" not in all_bound_names(tree)
        self._declared: set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.Global, ast.Nonlocal)):
                self._declared.update(n.names)
        taken = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                taken.add(n.id)
            elif isinstance(n, ast.arg):
                taken.add(n.arg)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                taken.add(n.name)
        self._counter = 0
        for name in taken:
            if name.startswith(_PREFIX):
                digits = name[len(_PREFIX):].split("_")[0]
                if digits.isdigit():
                    self._counter = max(self._counter, int(digits) + 1)

        new_body: list[ast.stmt] = []
        for stmt in tree.body:
            self._pending: list[ast.stmt] = []
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not region_is_dynamic(stmt):
                    self._process_scope(stmt)
            else:
                # Module level: maps only (atomic value swap); a rewritten
                # module-level reduction would leave the accumulator global
                # untouched on a mid-loop exception.
                replaced = self._try_map_stmt(stmt, module_level=True)
                if replaced is not None:
                    stmt = replaced
            new_body.extend(self._pending)
            new_body.append(stmt)
        tree.body = new_body
        if self.changes:
            insert_at = 0
            for i, s in enumerate(tree.body):
                if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant):
                    insert_at = i + 1
                    continue
                if isinstance(s, ast.ImportFrom) and s.module == "__future__":
                    insert_at = i + 1
                    continue
                break
            tree.body[insert_at:insert_at] = ast.parse(_RUNTIME).body
            ast.fix_missing_locations(tree)
        return tree

    # -- scopes -------------------------------------------------------------
    def _process_scope(self, fn: ast.AST) -> None:
        # Top-level functions only (v1): nested defs are left alone.
        self._fresh = (
            _reiterable_names(fn, fresh_container_names(fn))
            if self._container_ok
            else frozenset()
        )
        self._scope = fn
        fn.body = self._process_block(fn.body, guarded=False)

    def _process_block(self, stmts: list, guarded: bool) -> list:
        out: list[ast.stmt] = []
        for stmt in stmts:
            replaced = None
            if isinstance(stmt, ast.Assign):
                replaced = self._try_map_stmt(stmt, module_level=False)
            elif isinstance(stmt, ast.For) and not guarded:
                replaced = self._try_reduction(stmt)
            if replaced is not None:
                out.append(replaced)
                continue
            if isinstance(stmt, (ast.For, ast.While)):
                stmt.body = self._process_block(stmt.body, guarded)
                stmt.orelse = self._process_block(stmt.orelse, guarded)
            elif isinstance(stmt, ast.If):
                stmt.body = self._process_block(stmt.body, guarded)
                stmt.orelse = self._process_block(stmt.orelse, guarded)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                stmt.body = self._process_block(stmt.body, True)
            elif isinstance(stmt, ast.Try) or (
                hasattr(ast, "TryStar") and isinstance(stmt, ast.TryStar)
            ):
                stmt.body = self._process_block(stmt.body, True)
                for handler in stmt.handlers:
                    handler.body = self._process_block(handler.body, True)
                stmt.orelse = self._process_block(stmt.orelse, True)
                stmt.finalbody = self._process_block(stmt.finalbody, True)
            elif isinstance(stmt, ast.Match):
                for case in stmt.cases:
                    case.body = self._process_block(case.body, guarded)
            out.append(stmt)
        return out

    # -- maps ---------------------------------------------------------------
    def _try_map_stmt(self, stmt: ast.stmt, module_level: bool):
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and isinstance(stmt.value, ast.ListComp)
        ):
            return None
        comp = stmt.value
        if len(comp.generators) != 1:
            return None
        gen = comp.generators[0]
        if gen.ifs or gen.is_async or not isinstance(gen.target, ast.Name):
            return None
        iter_kind = self._classify_iter(gen.iter, module_level)
        if iter_kind is None:
            return None
        target = gen.target.id
        check = _EltCheck(target, frozenset())
        if not check.ok(comp.elt) or not check.uses_target:
            return None
        if not _worth_it(iter_kind, check):
            return None
        # The iterable itself appearing as a value operand (``v * xs``:
        # Python list repetition vs numpy broadcasting) would only be caught
        # by first-call verification; reject it statically instead.
        if iter_kind[0] == "name" and iter_kind[1].id in check.invariants:
            return None
        call = self._emit(comp.elt, target, check, iter_kind, reduction=False)
        new = ast.Assign(targets=stmt.targets, value=call)
        self.changes += 1
        return ast.copy_location(new, stmt)

    # -- reductions ---------------------------------------------------------
    def _try_reduction(self, loop: ast.For):
        if loop.orelse or not isinstance(loop.target, ast.Name):
            return None
        if len(loop.body) != 1:
            return None
        body = loop.body[0]
        acc = None
        elt = None
        if (
            isinstance(body, ast.AugAssign)
            and isinstance(body.op, ast.Add)
            and isinstance(body.target, ast.Name)
        ):
            acc, elt = body.target.id, body.value
        elif (
            isinstance(body, ast.Assign)
            and len(body.targets) == 1
            and isinstance(body.targets[0], ast.Name)
            and isinstance(body.value, ast.BinOp)
            and isinstance(body.value.op, ast.Add)
            and isinstance(body.value.left, ast.Name)
            and body.value.left.id == body.targets[0].id
        ):
            acc, elt = body.targets[0].id, body.value.right
        if acc is None or acc in self._declared:
            return None
        target = loop.target.id
        if acc == target:
            return None
        iter_kind = self._classify_iter(loop.iter, module_level=False)
        if iter_kind is None:
            return None
        check = _EltCheck(target, frozenset({acc}))
        if not check.ok(elt) or not check.uses_target:
            return None
        if not _worth_it(iter_kind, check):
            return None
        if not self._names_dead_after(loop, {target}):
            return None
        call = self._emit(elt, target, check, iter_kind, reduction=True, acc=acc)
        new = ast.Assign(
            targets=[ast.Name(id=acc, ctx=ast.Store())], value=call
        )
        self.changes += 1
        return ast.copy_location(new, loop)

    def _names_dead_after(self, loop: ast.For, names: set[str]) -> bool:
        stack = list(ast.iter_child_nodes(self._scope))
        while stack:
            n = stack.pop()
            if n is loop:
                continue
            if isinstance(n, ast.Name) and n.id in names:
                return False
            if isinstance(n, ast.arg) and n.arg in names:
                return False
            if isinstance(
                n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ) and n.name in names:
                return False
            stack.extend(ast.iter_child_nodes(n))
        return True

    # -- shared -------------------------------------------------------------
    def _classify_iter(self, node: ast.AST, module_level: bool):
        """("name", Name) for a proven re-iterable, ("range", args) for a
        plain range call, else None."""
        if (
            not module_level
            and isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in getattr(self, "_fresh", frozenset())
        ):
            return ("name", node)
        if self._range_ok:
            args = _range_args(node)
            if args is not None:
                return ("range", args)
        return None

    def _emit(self, elt, target, check, iter_kind, reduction, acc=None):
        """Generate the helper pair + dispatcher binding into ``_pending``
        and return the replacement call expression."""
        n = self._counter
        self._counter += 1
        base = f"{_PREFIX}{n}"
        invariants = sorted(check.invariants)
        kinds = "if" if _forces_float(elt) else "i"

        kind, payload = iter_kind
        if kind == "name":
            iter_params = ["_opast_xs"]
            py_iter_src = "_opast_xs"
            np_source = "_opast_np.asarray(_opast_xs)"
            call_iter_args = [copy.deepcopy(payload)]
        else:
            iter_params = [f"_opast_r{i}" for i in range(len(payload))]
            py_iter_src = f"range({', '.join(iter_params)})"
            np_source = f"_opast_np.arange({', '.join(iter_params)})"
            call_iter_args = [copy.deepcopy(a) for a in payload]

        acc_param = ["_opast_acc"] if reduction else []
        params = acc_param + iter_params + invariants
        elt_src = ast.unparse(elt)
        np_elt_src = ast.unparse(_rename_target(elt, target, "_opast_arr"))

        if reduction:
            py_src = (
                f"def {base}_py({', '.join(params)}):\n"
                f"    for {target} in {py_iter_src}:\n"
                f"        _opast_acc = _opast_acc + ({elt_src})\n"
                f"    return _opast_acc\n"
            )
            np_src = (
                f"def {base}_np(_opast_np, {', '.join(params)}):\n"
                f"    _opast_arr = {np_source}\n"
                f"    if _opast_arr.size == 0:\n"
                f"        return _opast_acc\n"
                f"    if _opast_arr.dtype.kind not in {kinds!r}:\n"
                f"        raise TypeError('opast-vector: unsupported dtype')\n"
                f"    return _opast_acc + ({np_elt_src}).sum().item()\n"
            )
        else:
            py_src = (
                f"def {base}_py({', '.join(params)}):\n"
                f"    return [({elt_src}) for {target} in {py_iter_src}]\n"
            )
            np_src = (
                f"def {base}_np(_opast_np, {', '.join(params)}):\n"
                f"    _opast_arr = {np_source}\n"
                f"    if _opast_arr.size == 0:\n"
                f"        return []\n"
                f"    if _opast_arr.dtype.kind not in {kinds!r}:\n"
                f"        raise TypeError('opast-vector: unsupported dtype')\n"
                f"    return ({np_elt_src}).tolist()\n"
            )
        bind_src = f"{base} = _opast_vector_dispatch({base}_py, {base}_np)\n"
        for src in (py_src, np_src, bind_src):
            self._pending.extend(ast.parse(src).body)

        call_args = []
        if reduction:
            call_args.append(ast.Name(id=acc, ctx=ast.Load()))
        call_args.extend(call_iter_args)
        call_args.extend(ast.Name(id=name, ctx=ast.Load()) for name in invariants)
        return ast.Call(
            func=ast.Name(id=base, ctx=ast.Load()), args=call_args, keywords=[]
        )
