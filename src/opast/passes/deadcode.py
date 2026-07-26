"""Dead code elimination.

* statements after ``return`` / ``raise`` / ``break`` / ``continue`` in the
  same block (including an ``if`` whose two branches both terminate);
* ``if`` with a constant test -> replaced by the live branch;
* ``while`` with a constant-false test -> replaced by its ``else`` block;
  ``while`` with a constant-true test -> dead ``else`` block removed;
* ``assert`` with a constant-true test;
* bare constant expression statements (docstrings are preserved);
* redundant ``pass`` statements.

Only statements are transformed; expressions are handled by other passes.
Blocks that become empty are padded with ``pass`` to stay syntactically valid.
"""

from __future__ import annotations

import ast

from ..analysis import param_names
from ..safety import region_is_dynamic
from .base import ScopedTransformer
from .licm import definite_bindings, unbound_risk_names

_TERMINAL = (ast.Return, ast.Raise, ast.Break, ast.Continue)


def _is_terminal(stmt: ast.stmt) -> bool:
    if isinstance(stmt, _TERMINAL):
        return True
    if isinstance(stmt, ast.If):
        return bool(
            stmt.body
            and stmt.orelse
            and _is_terminal(stmt.body[-1])
            and _is_terminal(stmt.orelse[-1])
        )
    return False


class DeadCodeElimination(ScopedTransformer):
    name = "dead-code"

    def __init__(self) -> None:
        super().__init__()
        #: Parameters of the scope being processed -- bound on entry, so a
        #: bare mention of one is removable (see _process).
        self._scope_params: frozenset[str] = frozenset()
        self._settled: set[str] = set()

    def run(self, tree: ast.Module) -> ast.Module:
        if region_is_dynamic(tree):
            self.skipped_scopes += 1
            return tree
        tree.body = self._process(tree.body, docstring=True)
        return tree

    # -- scopes ---------------------------------------------------------
    def _visit_scope(self, node: ast.AST) -> ast.AST:
        if region_is_dynamic(node):
            self.skipped_scopes += 1
            return node
        outer, self._scope_params = self._scope_params, param_names(node)
        outer_settled, self._settled = self._settled, set()
        try:
            node.body = self._process(node.body, docstring=True)
        finally:
            self._scope_params = outer
            self._settled = outer_settled
        return node

    visit_FunctionDef = _visit_scope
    visit_AsyncFunctionDef = _visit_scope
    visit_ClassDef = _visit_scope

    def visit_Lambda(self, node: ast.Lambda) -> ast.AST:
        return node  # no statements inside

    # -- compound statements --------------------------------------------
    def visit_If(self, node: ast.If):
        node.body = self._process(node.body)
        node.orelse = self._process(node.orelse, ensure=False)
        if isinstance(node.test, ast.Constant):
            self.changes += 1
            return node.body if node.test.value else node.orelse
        return node

    def visit_While(self, node: ast.While):
        node.body = self._process(node.body)
        node.orelse = self._process(node.orelse, ensure=False)
        if isinstance(node.test, ast.Constant):
            if not node.test.value:
                # Body never runs; the loop finishes without ``break`` so the
                # ``else`` block runs unconditionally.
                self.changes += 1
                return node.orelse
            if node.orelse:
                # ``while True`` only exits via break/return/raise, none of
                # which reach ``else``.
                self.changes += 1
                node.orelse = []
        return node

    def _visit_loop(self, node):
        node.body = self._process(node.body)
        node.orelse = self._process(node.orelse, ensure=False)
        return node

    visit_For = _visit_loop
    visit_AsyncFor = _visit_loop

    def _visit_with(self, node):
        node.body = self._process(node.body)
        return node

    visit_With = _visit_with
    visit_AsyncWith = _visit_with

    def _visit_try(self, node):
        node.body = self._process(node.body)
        for handler in node.handlers:
            handler.body = self._process(handler.body)
        node.orelse = self._process(node.orelse, ensure=False)
        node.finalbody = self._process(node.finalbody, ensure=False)
        return node

    visit_Try = _visit_try
    if hasattr(ast, "TryStar"):
        visit_TryStar = _visit_try

    def visit_Match(self, node: ast.Match):
        for case in node.cases:
            case.body = self._process(case.body)
        return node

    def visit_Assert(self, node: ast.Assert):
        if isinstance(node.test, ast.Constant) and bool(node.test.value):
            self.changes += 1
            return None
        return node

    # -- block processing -------------------------------------------------
    def _process(self, stmts, docstring: bool = False, ensure: bool = True):
        # ``_settled`` tracks names definitely bound at this point.  It is
        # inherited by nested blocks (bindings before a loop hold inside it)
        # and restored on the way out, so an ``except``/``finally`` block
        # never inherits bindings made in the ``try`` body -- those may not
        # have executed.
        saved = self._settled
        self._settled = set(saved) | set(self._scope_params)
        out: list[ast.stmt] = []
        terminated = False
        for i, stmt in enumerate(stmts):
            if terminated:
                self.changes += 1
                continue
            result = self.visit(stmt)
            if result is None:
                continue
            for produced in result if isinstance(result, list) else [result]:
                if isinstance(produced, ast.Expr) and isinstance(
                    produced.value, ast.Constant
                ):
                    if docstring and i == 0 and isinstance(produced.value.value, str):
                        out.append(produced)
                        continue
                    self.changes += 1
                    continue
                if (
                    isinstance(produced, ast.Expr)
                    and isinstance(produced.value, ast.Name)
                    and isinstance(produced.value.ctx, ast.Load)
                    and produced.value.id in self._settled
                ):
                    # ``x`` alone, x provably bound: loading a bound name has
                    # no side effect and cannot raise, so the statement is
                    # dead.  Unbound names must keep raising, hence the
                    # dominance check -- the unused pass downgrades dead
                    # stores to exactly this shape (e.g. once copy
                    # propagation ate the last read of a hoisted temporary).
                    self.changes += 1
                    continue
                self._settled |= definite_bindings(produced)
                self._settled -= unbound_risk_names(produced)
                out.append(produced)
                if _is_terminal(produced):
                    terminated = True

        if len(out) > 1:
            without_pass = [s for s in out if not isinstance(s, ast.Pass)]
            if without_pass:
                if len(without_pass) < len(out):
                    self.changes += len(out) - len(without_pass)
                    out = without_pass
            else:
                self.changes += len(out) - 1
                out = out[:1]

        if not out and ensure:
            out = [ast.Pass()]
        self._settled = saved
        return out
