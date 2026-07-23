"""Loop-invariant code motion (LICM).

Hoists loop-invariant expressions out of ``while``/``for`` bodies (and
``while`` tests) into a fresh ``_pyopt_inv_N = <expr>`` assignment placed
immediately before the loop.

Hoisting changes *when* (and for zero-iteration loops, *whether*) an
expression is evaluated, so an expression qualifies only if it is provably
pure and total (see :func:`pyopt.analysis.hoistable_int_expr`):

* int arithmetic over names proven plain-int by
  :func:`pyopt.analysis.infer_int_names` (``//``/``%`` need a non-zero int
  constant divisor, shifts a 0..256 constant amount);
* ``len(x)`` where *x* qualifies under
  :func:`pyopt.analysis.fresh_container_names` -- a fresh, never-escaping,
  never-mutated builtin container -- additionally gated module-wide on zero
  dynamic constructs and no shadowing of ``len``/constructor builtins;
* every name is invariant: not bound anywhere in the loop's subtree
  (assignments, targets, ``del``, nested ``global``/``nonlocal``
  declarations all count);
* every name is *definitely bound* before the loop, established by a
  straight-line dominance scan (plain assignments accumulate; ``del``,
  ``except ... as`` and conditional bindings subtract or don't count) -- so
  a zero-iteration loop can't turn a non-executed error into a hoisted one;
* contains at least one name (pure-constant expressions are left to the
  folding pass).

Temporaries use the reserved ``_pyopt_inv_`` prefix; the counter starts
above any existing suffix in the module.  Nested scopes inside loop bodies
are never entered; module level is never hoisted (globals may be rebound by
called functions).  Function scopes only, as with algebraic simplification.
"""

from __future__ import annotations

import ast

from ..analysis import (
    LEN_GATE_NAMES,
    all_bound_names,
    binding_names,
    builtin_gate,
    contains_name,
    fresh_container_names,
    hoistable_int_expr,
    infer_int_names,
)
from ..safety import region_is_dynamic, tree_has_dynamic
from .base import ScopedTransformer

_TEMP_PREFIX = "_pyopt_inv"

_LOOPS = (ast.While, ast.For, ast.AsyncFor)


def subtree_bindings(node: ast.AST) -> set[str]:
    """Every name bound anywhere in *node*'s subtree (nested scopes and
    their ``global``/``nonlocal`` declarations included)."""
    bound: set[str] = set()
    for n in ast.walk(node):
        bound.update(binding_names(n))
        if isinstance(n, ast.arg):
            bound.add(n.arg)
    return bound


def unbound_risk_names(stmt: ast.AST) -> set[str]:
    """Names that may be unbound (or externally rebindable) after/inside
    *stmt*: ``del`` targets, ``except ... as`` names (deleted on handler
    exit), and ``global``/``nonlocal`` declarations."""
    risk: set[str] = set()
    for n in ast.walk(stmt):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Del):
            risk.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            risk.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            risk.update(n.names)
    return risk


def definite_bindings(stmt: ast.AST) -> set[str]:
    """Names certainly bound once *stmt* completes (straight-line kinds only;
    conditional binders like if/loops/try contribute nothing)."""
    added: set[str] = set()
    if isinstance(stmt, ast.Assign):
        for target in stmt.targets:
            for n in ast.walk(target):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    added.add(n.id)
    elif isinstance(stmt, ast.AnnAssign):
        if stmt.value is not None and isinstance(stmt.target, ast.Name):
            added.add(stmt.target.id)
    elif isinstance(stmt, ast.AugAssign):
        if isinstance(stmt.target, ast.Name):
            added.add(stmt.target.id)  # completing implies it was bound
    elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        added.add(stmt.name)
    elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
        for a in stmt.names:
            if a.name != "*":
                added.add(a.asname or a.name.split(".")[0])
    return added


def container_gate(tree: ast.Module) -> bool:
    """Module-wide precondition for trusting ``len`` and the builtin
    container constructors."""
    return not tree_has_dynamic(tree) and not (LEN_GATE_NAMES & all_bound_names(tree))


class _Hoister(ast.NodeTransformer):
    """Replaces maximal hoistable subexpressions with temp names."""

    def __init__(self, ints: set[str], containers: set[str],
                 owner: "LoopInvariantMotion") -> None:
        self.ints = ints
        self.containers = containers
        self.owner = owner
        self.temps: dict[str, tuple[str, ast.expr]] = {}

    def _skip(self, node: ast.AST) -> ast.AST:
        return node

    visit_FunctionDef = _skip
    visit_AsyncFunctionDef = _skip
    visit_Lambda = _skip
    visit_ClassDef = _skip

    def _maybe_hoist(self, node: ast.AST) -> ast.AST:
        if hoistable_int_expr(node, self.ints, self.containers) and contains_name(node):
            key = ast.dump(node)
            entry = self.temps.get(key)
            if entry is None:
                entry = (self.owner._fresh_name(), node)
                self.temps[key] = entry
            return ast.copy_location(ast.Name(id=entry[0], ctx=ast.Load()), node)
        return self.generic_visit(node)

    visit_BinOp = _maybe_hoist
    visit_UnaryOp = _maybe_hoist
    visit_Call = _maybe_hoist


class LoopInvariantMotion(ScopedTransformer):
    name = "licm"

    def run(self, tree: ast.Module) -> ast.Module:
        if region_is_dynamic(tree):
            self.skipped_scopes += 1
            return tree
        self._len_ok = container_gate(tree)
        self._range_ok = builtin_gate(tree, "range")
        self._counter = 0
        for n in ast.walk(tree):
            ident = None
            if isinstance(n, ast.Name):
                ident = n.id
            elif isinstance(n, ast.arg):
                ident = n.arg
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
        containers = fresh_container_names(node) if self._len_ok else frozenset()
        ints = infer_int_names(node, containers, range_ok=self._range_ok)
        if ints or containers:
            node.body = self._process_block(node.body, set(), ints, containers)
        return node

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    # -- dominance-aware block walk ----------------------------------------
    def _process_block(self, stmts, bound: set[str], ints, containers) -> list:
        bound = set(bound)
        out = []
        for stmt in stmts:
            if isinstance(stmt, _LOOPS):
                temps = self._hoist_loop(stmt, bound, ints, containers)
                out.extend(temps)
                for t in temps:
                    bound.add(t.targets[0].id)
                inner = bound - unbound_risk_names(stmt)
                if isinstance(stmt, (ast.For, ast.AsyncFor)):
                    # The loop target is (re)bound before every body run, so
                    # inside the body it is definitely bound; a mid-body
                    # ``del`` is handled by the recursive walk itself.
                    inner |= {
                        n.id
                        for n in ast.walk(stmt.target)
                        if isinstance(n, ast.Name)
                        and isinstance(n.ctx, ast.Store)
                    }
                stmt.body = self._process_block(stmt.body, inner, ints, containers)
            elif isinstance(stmt, ast.If):
                stmt.body = self._process_block(stmt.body, bound, ints, containers)
                stmt.orelse = self._process_block(stmt.orelse, bound, ints, containers)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                stmt.body = self._process_block(stmt.body, bound, ints, containers)
            elif isinstance(stmt, ast.Try) or (
                hasattr(ast, "TryStar") and isinstance(stmt, ast.TryStar)
            ):
                stmt.body = self._process_block(stmt.body, bound, ints, containers)
                for handler in stmt.handlers:
                    handler.body = self._process_block(
                        handler.body, bound, ints, containers
                    )
                stmt.orelse = self._process_block(stmt.orelse, bound, ints, containers)
                stmt.finalbody = self._process_block(
                    stmt.finalbody, bound, ints, containers
                )
            elif isinstance(stmt, ast.Match):
                for case in stmt.cases:
                    case.body = self._process_block(case.body, bound, ints, containers)
            out.append(stmt)
            bound |= definite_bindings(stmt)
            bound -= unbound_risk_names(stmt)
        return out

    def _hoist_loop(self, loop, bound: set[str], ints, containers) -> list:
        loop_bound = subtree_bindings(loop)
        avail_ints = {n for n in ints if n in bound and n not in loop_bound}
        avail_containers = {
            n for n in containers if n in bound and n not in loop_bound
        }
        if not avail_ints and not avail_containers:
            return []
        hoister = _Hoister(avail_ints, avail_containers, self)
        if isinstance(loop, ast.While):
            loop.test = hoister.visit(loop.test)
        loop.body = [hoister.visit(s) for s in loop.body]
        temps = []
        for name, expr in hoister.temps.values():
            target = ast.copy_location(ast.Name(id=name, ctx=ast.Store()), loop)
            assign = ast.copy_location(
                ast.Assign(targets=[target], value=expr), loop
            )
            temps.append(assign)
            self.changes += 1
        return temps
