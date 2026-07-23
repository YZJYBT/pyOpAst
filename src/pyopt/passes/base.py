"""Base class for optimisation passes with per-scope dynamic-code fallback."""

from __future__ import annotations

import ast

from ..safety import region_is_dynamic


class ScopedTransformer(ast.NodeTransformer):
    """A :class:`ast.NodeTransformer` that skips dynamic scopes.

    Entry point is :meth:`run`.  Before descending into any scope (including
    the module itself) the scope's own region is checked for dynamic
    constructs; a tainted scope is returned untouched and not descended into,
    so nested scopes of a tainted scope are skipped as well.
    """

    name = "pass"

    def __init__(self) -> None:
        self.changes = 0
        self.skipped_scopes = 0

    def run(self, tree: ast.Module) -> ast.Module:
        if region_is_dynamic(tree):
            self.skipped_scopes += 1
            return tree
        return self.visit(tree)

    def _visit_scope(self, node: ast.AST) -> ast.AST:
        if region_is_dynamic(node):
            self.skipped_scopes += 1
            return node
        return self.generic_visit(node)

    visit_FunctionDef = _visit_scope
    visit_AsyncFunctionDef = _visit_scope
    visit_Lambda = _visit_scope
    visit_ClassDef = _visit_scope
