"""Common subexpression elimination (CSE).

Within one statement block, an expression appearing two or more times is
computed once into a ``_pyopt_cse_N`` temporary inserted before the
statement of its first occurrence, and every occurrence is replaced by the
temp.  Eligible expressions are exactly the LICM ones
(:func:`pyopt.analysis.hoistable_int_expr`): provably pure and total int
arithmetic over proven-int names, plus ``len()`` of fresh never-escaping
builtin containers (module-wide gate on no dynamic constructs / unshadowed
builtins).  Purity+totality is what makes this sound without occurrence
dominance: even if a later occurrence sits inside a branch or nested loop,
evaluating the expression once up front is unobservable.

Additional requirements per expression:

* all names are definitely bound before the first occurrence's statement
  (same straight-line dominance scan as LICM);
* no name is rebound anywhere in the statement span from the first to the
  last occurrence (full-subtree binding scan, nested ``global``/``nonlocal``
  included);
* it contains at least one name and at least one operation or ``len`` call
  (repeated plain names/constants are free anyway).

Occurrences are matched *maximally* (a recorded expression's subexpressions
are not counted separately) and a candidate that is a subexpression of
another selected candidate is deferred to a later pipeline iteration.
Blocks are processed independently; nested scope bodies are never entered.
Function scopes only.
"""

from __future__ import annotations

import ast
import copy

from ..analysis import (
    builtin_gate,
    contains_name,
    fresh_container_names,
    hoistable_int_expr,
    infer_int_names,
    is_len_call,
)
from ..safety import region_is_dynamic
from .base import ScopedTransformer
from .licm import (
    _LOOPS,
    container_gate,
    definite_bindings,
    subtree_bindings,
    unbound_risk_names,
)

_TEMP_PREFIX = "_pyopt_cse"


def _worthy(node: ast.AST, containers) -> bool:
    """Repeating this expression actually saves work."""
    if not contains_name(node):
        return False
    return isinstance(node, (ast.BinOp, ast.UnaryOp)) or is_len_call(node, containers)


class _Collector(ast.NodeVisitor):
    """Records maximal eligible expressions per statement index."""

    def __init__(self, ints, containers) -> None:
        self.ints = ints
        self.containers = containers
        self.table: dict[str, list[tuple[int, ast.AST]]] = {}
        self._idx = 0

    def collect(self, idx: int, stmt: ast.stmt) -> None:
        self._idx = idx
        self.visit(stmt)

    def _skip(self, node: ast.AST) -> None:
        pass

    visit_FunctionDef = _skip
    visit_AsyncFunctionDef = _skip
    visit_Lambda = _skip
    visit_ClassDef = _skip

    def _record(self, node: ast.AST) -> None:
        if hoistable_int_expr(node, self.ints, self.containers) and _worthy(
            node, self.containers
        ):
            self.table.setdefault(ast.dump(node), []).append((self._idx, node))
            # keep descending: a shared subexpression may repeat inside
            # otherwise-distinct larger expressions
        self.generic_visit(node)

    visit_BinOp = _record
    visit_UnaryOp = _record
    visit_Call = _record


class _Replacer(ast.NodeTransformer):
    """Replaces selected maximal expressions with their temp names, using
    the same traversal shape as :class:`_Collector`."""

    def __init__(self, active: dict[str, str], ints, containers, owner) -> None:
        self.active = active
        self.ints = ints
        self.containers = containers
        self.owner = owner

    def _skip(self, node: ast.AST) -> ast.AST:
        return node

    visit_FunctionDef = _skip
    visit_AsyncFunctionDef = _skip
    visit_Lambda = _skip
    visit_ClassDef = _skip

    def _replace(self, node: ast.AST) -> ast.AST:
        temp = self.active.get(ast.dump(node))
        if temp is not None:
            return ast.copy_location(ast.Name(id=temp, ctx=ast.Load()), node)
        return self.generic_visit(node)

    visit_BinOp = _replace
    visit_UnaryOp = _replace
    visit_Call = _replace


class CommonSubexpressionElimination(ScopedTransformer):
    name = "cse"

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

    # -- one block ----------------------------------------------------------
    def _process_block(self, stmts, bound: set[str], ints, containers) -> list:
        stmts = self._eliminate(stmts, bound, ints, containers)

        # Recurse into nested blocks for expressions local to them.
        tracked = set(bound)
        for stmt in stmts:
            if isinstance(stmt, _LOOPS):
                inner = tracked - unbound_risk_names(stmt)
                if isinstance(stmt, (ast.For, ast.AsyncFor)):
                    # Loop targets are (re)bound before every body run.
                    inner |= {
                        n.id
                        for n in ast.walk(stmt.target)
                        if isinstance(n, ast.Name)
                        and isinstance(n.ctx, ast.Store)
                    }
                stmt.body = self._process_block(stmt.body, inner, ints, containers)
            elif isinstance(stmt, ast.If):
                stmt.body = self._process_block(stmt.body, tracked, ints, containers)
                stmt.orelse = self._process_block(
                    stmt.orelse, tracked, ints, containers
                )
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                stmt.body = self._process_block(stmt.body, tracked, ints, containers)
            elif isinstance(stmt, ast.Try) or (
                hasattr(ast, "TryStar") and isinstance(stmt, ast.TryStar)
            ):
                stmt.body = self._process_block(stmt.body, tracked, ints, containers)
                for handler in stmt.handlers:
                    handler.body = self._process_block(
                        handler.body, tracked, ints, containers
                    )
                stmt.orelse = self._process_block(
                    stmt.orelse, tracked, ints, containers
                )
                stmt.finalbody = self._process_block(
                    stmt.finalbody, tracked, ints, containers
                )
            elif isinstance(stmt, ast.Match):
                for case in stmt.cases:
                    case.body = self._process_block(
                        case.body, tracked, ints, containers
                    )
            tracked |= definite_bindings(stmt)
            tracked -= unbound_risk_names(stmt)
        return stmts

    def _eliminate(self, stmts, bound: set[str], ints, containers) -> list:
        collector = _Collector(ints, containers)
        bound_at: list[set[str]] = []
        rebinds: list[set[str]] = []
        running = set(bound)
        for idx, stmt in enumerate(stmts):
            bound_at.append(set(running))
            collector.collect(idx, stmt)
            rebinds.append(subtree_bindings(stmt))
            running |= definite_bindings(stmt)
            running -= unbound_risk_names(stmt)

        selected: dict[str, tuple[int, str, ast.AST]] = {}
        relevant = set(ints) | set(containers)
        for key, occurrences in collector.table.items():
            if len(occurrences) < 2:
                continue
            first = occurrences[0][0]
            last = occurrences[-1][0]
            sample = occurrences[0][1]
            # Only local names matter for boundness/rebinding; the ``len``
            # name itself is builtin-stable via the module-wide gate.
            names = {
                n.id for n in ast.walk(sample) if isinstance(n, ast.Name)
            } & relevant
            if not names <= bound_at[first]:
                continue
            if any(names & rebinds[i] for i in range(first, last + 1)):
                continue
            selected[key] = (first, "", sample)
        if not selected:
            return list(stmts)

        # A candidate that is a subexpression of another selected candidate
        # is deferred to a later pipeline iteration (dump() of a subtree is a
        # substring of its parent's dump()).
        for key in list(selected):
            if any(key != other and key in other for other in selected):
                del selected[key]

        for key, (first, _, sample) in list(selected.items()):
            selected[key] = (first, self._fresh_name(), sample)

        out: list[ast.stmt] = []
        for idx, stmt in enumerate(stmts):
            for key, (first, temp, sample) in selected.items():
                if first == idx:
                    target = ast.copy_location(
                        ast.Name(id=temp, ctx=ast.Store()), stmt
                    )
                    assign = ast.copy_location(
                        ast.Assign(targets=[target], value=copy.deepcopy(sample)),
                        stmt,
                    )
                    out.append(assign)
                    self.changes += 1
            active = {
                key: temp
                for key, (first, temp, _) in selected.items()
                if first <= idx
            }
            if active:
                stmt = _Replacer(active, ints, containers, self).visit(stmt)
            out.append(stmt)
        return out
