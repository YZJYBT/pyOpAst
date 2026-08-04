"""Loop fission: split an ``append`` out of a mixed loop body (loop-fission).

**Not registered in the pipeline.**  Measured 2026-08-04 on CPython 3.14.2
(10k-element proven-fresh list, aggressive ``pure-calls``, best-of-7):
fission vs. no-fission 0.946x with a light residual body, 0.877x with a
heavy one, 1.020x when the helper call dominates -- the second traversal
plus the intermediate list costs more than the comprehension's
LIST_APPEND buys.  The pass is complete and correct (all guards below
hold); it is kept as the reference implementation for a future numpy
vectorization probe, where the extracted comprehension would become a
vectorized expression and the payoff side actually exists.  To experiment:
``from opast.passes import LoopFission`` and run it manually.

Rewrites

    acc = []
    for v in xs:            # xs provably a fresh list/tuple/dict
        acc.append(f(v))
        <other statements>

into

    acc = [f(v) for v in xs]
    for v in xs:
        <other statements>

so the accumulation runs on the comprehension's dedicated LIST_APPEND
bytecode (and can feed comp-to-map) while the rest of the body stays a
plain loop.  ``loop-to-comp`` handles the case where nothing is left over;
this pass exists for bodies that do more than accumulate.

Fission reorders work -- all appends now happen before any residual
statement runs -- and iterates the source twice, so it fires only when
both are provably unobservable:

* the iterable is a plain name proven by
  :func:`opast.analysis.fresh_container_names` (never escapes, never
  mutated, bound exactly once) **and** its single binding shape proves a
  re-iterable container with stable order: a list/tuple display or
  comprehension, or a ``list``/``tuple``/``sorted``/``dict`` call, or a
  dict display.  Sets are excluded on principle (no positional identity),
  generators/iterators by construction (the first traversal would drain
  them), and anything unproven -- parameters included -- because
  re-iterability cannot be established;
* the element expression is pure and total: constants, loop-target names,
  invariant definitely-bound names appended as-is, proven int/float
  arithmetic (:func:`opast.analysis.hoistable_num_expr`), tuple displays
  of those -- and, under aggressive ``pure-calls``, calls to trusted-pure
  module functions over such arguments (that option's assumption already
  covers the moved evaluation).  Every name it reads is either a loop
  target or written nowhere in the loop body, so the value computed before
  the residual ran equals the value the original interleaving computed;
* the accumulator seed ``acc = []`` immediately precedes the loop, the
  accumulator's only occurrence in the loop is the extracted call's
  receiver, and it is never referenced inside any nested scope of the
  function -- so no closure can observe a partially-filled list if the
  residual loop raises or returns early;
* the loop body contains no ``break``/``continue`` (either could skip
  appends the comprehension already performed), no loop ``else``, and no
  nested scope / ``yield`` / ``await`` / walrus (comprehension scoping);
  loop-target names are not rebound inside the body (the comprehension
  always sees the fresh per-iteration value);
* the pair does not sit inside ``try``/``with`` (a handler could observe
  the partially-filled statement-form list) and the module has no dynamic
  constructs; function scopes only.

Element expressions with no operation or call are not worth a second
traversal (``acc.append(v)`` alone is loop-to-comp's business when the
body is only that, and a wash here) and are skipped.
"""

from __future__ import annotations

import ast
import copy

from ..analysis import (
    AnnotationTrust,
    bound_names,
    builtin_gate,
    fresh_container_names,
    hoistable_num_expr,
    infer_float_names,
    infer_int_names,
    param_names,
)
from ..safety import iter_region, region_is_dynamic, tree_has_dynamic
from .base import ScopedTransformer
from .licm import (
    container_gate,
    definite_bindings,
    subtree_bindings,
    unbound_risk_names,
)
from .looptocomp import _flat_target_names

#: Binding value shapes proving a re-iterable, stably-ordered container.
_LIST_DISPLAYS = (ast.List, ast.Tuple, ast.ListComp)
_LIST_CALLS = frozenset({"list", "tuple", "sorted"})
_DICT_DISPLAYS = (ast.Dict, ast.DictComp)
_DICT_CALLS = frozenset({"dict"})

#: Node types whose presence anywhere in the loop subtree disqualifies it.
_FORBIDDEN = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.NamedExpr,
    ast.Break,
    ast.Continue,
    ast.Return,
    ast.Raise,
)


def _reiterable_names(scope: ast.AST, fresh: frozenset[str]) -> frozenset[str]:
    """Fresh-container names whose binding shape also proves list/tuple/dict
    semantics (re-iterable, order stable across two traversals)."""
    if not fresh:
        return frozenset()
    proven: set[str] = set()
    for n in iter_region(scope):
        if not (
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in fresh
        ):
            continue
        value = n.value
        if isinstance(value, (*_LIST_DISPLAYS, *_DICT_DISPLAYS)):
            proven.add(n.targets[0].id)
        elif (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in (_LIST_CALLS | _DICT_CALLS)
        ):
            proven.add(n.targets[0].id)
    return frozenset(proven)


class LoopFission(ScopedTransformer):
    name = "loop-fission"

    def run(self, tree: ast.Module) -> ast.Module:
        # Same whole-module gate as loop-to-comp: exec/eval anywhere could
        # rebind the accumulator mid-loop, and the fresh-container proof
        # needs the len/constructor builtins unshadowed.
        if tree_has_dynamic(tree):
            self.skipped_scopes += 1
            return tree
        self._container_ok = container_gate(tree)
        self._range_ok = builtin_gate(tree, "range")
        self._trust = AnnotationTrust(None)
        self._declared: set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.Global, ast.Nonlocal)):
                self._declared.update(n.names)
        return self.visit(tree)

    def _visit_function(self, node: ast.AST) -> ast.AST:
        if region_is_dynamic(node):
            self.skipped_scopes += 1
            return node
        node = self.generic_visit(node)  # nested functions first
        containers = (
            fresh_container_names(node) if self._container_ok else frozenset()
        )
        reiterable = _reiterable_names(node, containers)
        if not reiterable:
            return node
        self._ints = infer_int_names(node, containers, range_ok=self._range_ok)
        self._floats = infer_float_names(node, ints=self._ints, containers=containers)
        self._containers = containers
        self._scope_pure = (
            self.pure_calls - bound_names(node) if self.pure_calls else frozenset()
        )
        self._reiterable = reiterable
        self._scope = node
        node.body = self._process_block(node.body, set(param_names(node)))
        return node

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    # -- dominance-aware block walk (same shape as LICM) --------------------
    def _process_block(self, stmts: list, bound: set[str]) -> list:
        bound = set(bound)
        out: list[ast.stmt] = []
        i = 0
        while i < len(stmts):
            stmt = stmts[i]
            nxt = stmts[i + 1] if i + 1 < len(stmts) else None
            if nxt is not None:
                replacement = self._try_fission(stmt, nxt, bound)
                if replacement is not None:
                    new_seed, loop = replacement
                    self.changes += 1
                    out.append(new_seed)
                    bound |= definite_bindings(new_seed)
                    self._descend(loop, bound)
                    out.append(loop)
                    bound |= definite_bindings(loop)
                    bound -= unbound_risk_names(loop)
                    i += 2
                    continue
            self._descend(stmt, bound)
            out.append(stmt)
            bound |= definite_bindings(stmt)
            bound -= unbound_risk_names(stmt)
            i += 1
        return out

    def _descend(self, stmt: ast.stmt, bound: set[str]) -> None:
        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            inner = bound - unbound_risk_names(stmt)
            if isinstance(stmt, (ast.For, ast.AsyncFor)):
                inner |= {
                    n.id
                    for n in ast.walk(stmt.target)
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                }
            stmt.body = self._process_block(stmt.body, inner)
            stmt.orelse = self._process_block(stmt.orelse, inner)
        elif isinstance(stmt, ast.If):
            stmt.body = self._process_block(stmt.body, bound)
            stmt.orelse = self._process_block(stmt.orelse, bound)
        elif isinstance(stmt, ast.Match):
            for case in stmt.cases:
                case.body = self._process_block(case.body, bound)
        # try/with bodies are never fissioned (partial list observable by
        # the handler / __exit__) and not descended into for this purpose.

    # -- the rewrite --------------------------------------------------------
    def _try_fission(self, seed: ast.stmt, loop: ast.stmt, bound: set[str]):
        if not (
            isinstance(seed, ast.Assign)
            and len(seed.targets) == 1
            and isinstance(seed.targets[0], ast.Name)
            and isinstance(seed.value, ast.List)
            and not seed.value.elts
        ):
            return None
        acc = seed.targets[0].id
        if acc in self._declared:
            return None
        if not (
            isinstance(loop, ast.For)
            and not loop.orelse
            and isinstance(loop.iter, ast.Name)
            and loop.iter.id in self._reiterable
            and len(loop.body) >= 2
        ):
            return None
        target_names = _flat_target_names(loop.target)
        if target_names is None or set(target_names) & self._declared:
            return None

        for n in ast.walk(loop):
            if isinstance(n, _FORBIDDEN):
                return None

        # Exactly one occurrence of the accumulator in the loop: the
        # receiver of one top-level ``acc.append(elt)`` statement.
        if sum(
            isinstance(n, ast.Name) and n.id == acc for n in ast.walk(loop)
        ) != 1:
            return None
        append_idx = None
        for idx, stmt in enumerate(loop.body):
            call = stmt.value if isinstance(stmt, ast.Expr) else None
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "append"
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == acc
                and len(call.args) == 1
                and not isinstance(call.args[0], ast.Starred)
                and not call.keywords
            ):
                append_idx = idx
                elt = call.args[0]
                break
        if append_idx is None:
            return None

        # No closure may observe the accumulator: an early exit of the
        # residual loop would otherwise expose full-vs-partial contents.
        if self._captured_by_nested_scope(acc):
            return None

        body_bound = set()
        for stmt in loop.body:
            body_bound |= subtree_bindings(stmt)
        # The comprehension always sees the fresh per-iteration target; a
        # rebound target (or iterable) would make that position-dependent.
        if set(target_names) & body_bound or loop.iter.id in body_bound:
            return None
        if not self._elt_ok(elt, set(target_names), body_bound, bound):
            return None
        if not _worth_it(elt):
            return None

        comp = ast.ListComp(
            elt=elt,
            generators=[
                ast.comprehension(
                    target=copy.deepcopy(loop.target),
                    iter=copy.deepcopy(loop.iter),
                    ifs=[],
                    is_async=0,
                )
            ],
        )
        new_seed = ast.copy_location(
            ast.Assign(targets=[seed.targets[0]], value=comp), seed
        )
        ast.fix_missing_locations(new_seed)
        loop.body = [s for i, s in enumerate(loop.body) if i != append_idx]
        return new_seed, loop

    def _captured_by_nested_scope(self, name: str) -> bool:
        scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
        for n in ast.walk(self._scope):
            if n is self._scope or not isinstance(n, scopes):
                continue
            if any(
                isinstance(m, ast.Name) and m.id == name for m in ast.walk(n)
            ):
                return True
        return False

    def _elt_ok(self, node, targets: set[str], body_bound: set[str],
                bound: set[str]) -> bool:
        """Pure, total, and independent of every residual write."""
        if isinstance(node, ast.Constant):
            return True
        if isinstance(node, ast.Name):
            if not isinstance(node.ctx, ast.Load):
                return False
            if node.id in targets:
                return True
            # Invariant: never written in the body, definitely bound before.
            return node.id not in body_bound and node.id in bound
        if isinstance(node, ast.Tuple) and isinstance(node.ctx, ast.Load):
            return all(
                self._elt_ok(e, targets, body_bound, bound) for e in node.elts
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self._scope_pure
            and node.func.id not in body_bound
        ):
            return all(
                self._elt_ok(a, targets, body_bound, bound) for a in node.args
            ) and all(
                k.arg is not None
                and self._elt_ok(k.value, targets, body_bound, bound)
                for k in node.keywords
            )
        # Proven-numeric arithmetic (pure *and* total); its names are
        # int/float-proven, still guarded against residual writes.
        if hoistable_num_expr(node, self._ints, self._floats, self._containers):
            return all(
                n.id in targets or (n.id not in body_bound and n.id in bound)
                for n in ast.walk(node)
                if isinstance(n, ast.Name) and n.id != "len"
            )
        return False


def _worth_it(elt: ast.AST) -> bool:
    """A second traversal must buy actual per-element work."""
    return any(
        isinstance(n, (ast.BinOp, ast.UnaryOp, ast.Call)) for n in ast.walk(elt)
    )
