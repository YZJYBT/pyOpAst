"""Global/builtin name localization in loops.

A name read inside a hot loop body costs a ``LOAD_GLOBAL`` (module dict, then
builtins dict) on every iteration.  When the name provably always refers to
the same object, binding it once to a ``_pyopt_glb_N`` local right before the
loop turns every read into a ``LOAD_FAST``.

A name qualifies when the whole module is free of dynamic constructs and:

* **builtin**: the name is never bound anywhere in the tree (any scope,
  ``global`` declarations and parameters included) and is a real attribute of
  :mod:`builtins` -- so it is always bound and always the builtin; or
* **stable module global**: the name is bound exactly once in the module
  region, by a *direct* top-level statement (``def``/``class``/import/plain
  assignment -- conditional bindings inside ``if``/``try`` don't count), is
  never declared ``global`` in any scope, and that binding statement precedes
  the top-level statement containing the loop's enclosing function.  A call
  can only reach the loop after the enclosing top-level ``def``/``class``
  statement has executed, which is after the binding executed -- so the name
  is definitely bound and can never be rebound.

Only reads in the loop's *per-iteration* parts (body, and the test for
``while``) make a name worth localizing; the hoisted local then replaces
every read in the whole loop statement.  Nested scope bodies keep reading
the global (identical value; closures stay correct), and comprehensions are
skipped entirely (their targets shadow at runtime).  Loops at module level
are left alone: a module-level temp would still be a global.

Runs last in the pipeline so that inlining and comp-to-map get first claim
on the names they understand (``map``/``filter`` introduced by comp-to-map
are picked up here on the next iteration).  Under ``--jit`` the numba
whitelist builtins are excluded (``jit_mode``): rewriting ``abs(x)`` to
``_pyopt_glb_0(x)`` inside a would-be-jitted function would make it fail
numba's typing and lose the compilation.
"""

from __future__ import annotations

import ast
import builtins
from collections import Counter

from ..analysis import all_bound_names, bound_names
from ..safety import SCOPE_NODES, _boundary_children, region_is_dynamic, tree_has_dynamic
from .base import ScopedTransformer
from .licm import definite_bindings

_TEMP_PREFIX = "_pyopt_glb"

_COMP_NODES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)

#: Builtins the numba whitelist may call by name (pyopt.passes.jit); kept as
#: globals under --jit so jitted functions still type-check.
_JIT_WHITELIST_BUILTINS = frozenset(
    {"range", "abs", "min", "max", "round", "divmod", "int", "float", "bool"}
)


def _region_nodes(roots):
    """Walk *roots* yielding nodes of the enclosing region only: nested scope
    bodies are skipped (their boundary parts -- decorators, defaults,
    annotations -- still belong here) and comprehensions are skipped whole."""
    stack = list(roots)
    while stack:
        node = stack.pop()
        yield node
        if isinstance(node, SCOPE_NODES):
            stack.extend(_boundary_children(node))
        elif isinstance(node, _COMP_NODES):
            continue
        else:
            stack.extend(ast.iter_child_nodes(node))


class _Renamer(ast.NodeTransformer):
    """Replaces Load-context names per *mapping* with the same region walk
    shape as :func:`_region_nodes`."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def _skip(self, node: ast.AST) -> ast.AST:
        return node

    visit_Lambda = _skip
    visit_ListComp = _skip
    visit_SetComp = _skip
    visit_DictComp = _skip
    visit_GeneratorExp = _skip

    def _visit_scope_stmt(self, node: ast.AST) -> ast.AST:
        # Only the boundary parts belong to this region; the body does not.
        for i, dec in enumerate(node.decorator_list):
            node.decorator_list[i] = self.visit(dec)
        return node

    visit_FunctionDef = _visit_scope_stmt
    visit_AsyncFunctionDef = _visit_scope_stmt
    visit_ClassDef = _visit_scope_stmt

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.mapping:
            return ast.copy_location(
                ast.Name(id=self.mapping[node.id], ctx=ast.Load()), node
            )
        return node


class GlobalLocalization(ScopedTransformer):
    name = "localize"

    def __init__(self) -> None:
        super().__init__()
        self.jit_mode = False

    def run(self, tree: ast.Module) -> ast.Module:
        if tree_has_dynamic(tree):
            self.skipped_scopes += 1
            return tree
        self._all_bound = all_bound_names(tree)
        global_declared: set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Global):
                global_declared.update(n.names)

        # Stable module globals: bound exactly once in the module region, by
        # a direct top-level statement, never declared global anywhere.
        counts: Counter[str] = Counter()
        direct_index: dict[str, int] = {}
        for idx, stmt in enumerate(tree.body):
            for node in _region_nodes([stmt]):
                if isinstance(node, ast.Name):
                    if isinstance(node.ctx, (ast.Store, ast.Del)):
                        counts[node.id] += 1
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                       ast.ClassDef)):
                    counts[node.name] += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        if alias.name != "*":
                            counts[alias.asname or alias.name.split(".")[0]] += 1
                elif isinstance(node, (ast.Global, ast.Nonlocal)):
                    counts.update(node.names)
            for name in definite_bindings(stmt):
                direct_index.setdefault(name, idx)
        # Comprehensions are skipped by the region walk above but can bind
        # walrus targets into the enclosing scope -- count those too.
        for n in ast.walk(tree):
            if isinstance(n, _COMP_NODES):
                for sub in ast.walk(n):
                    if isinstance(sub, ast.NamedExpr) and isinstance(
                        sub.target, ast.Name
                    ):
                        counts[sub.target.id] += 1
        self._module_single = {
            name: direct_index[name]
            for name, c in counts.items()
            if c == 1 and name in direct_index and name not in global_declared
        }

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

        self._scopes: list[frozenset[str]] = []
        self._fn_depth = 0
        new_body: list[ast.stmt] = []
        for idx, stmt in enumerate(tree.body):
            self._top_index = idx
            result = self.visit(stmt)
            if isinstance(result, list):
                new_body.extend(result)
            elif result is not None:
                new_body.append(result)
        tree.body = new_body
        return tree

    def _fresh_name(self) -> str:
        name = f"{_TEMP_PREFIX}_{self._counter}"
        self._counter += 1
        return name

    # -- scopes -------------------------------------------------------------
    def _visit_scope(self, node: ast.AST) -> ast.AST:
        if region_is_dynamic(node):
            self.skipped_scopes += 1
            return node
        is_fn = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        self._scopes.append(bound_names(node))
        if is_fn:
            self._fn_depth += 1
        try:
            return self.generic_visit(node)
        finally:
            self._scopes.pop()
            if is_fn:
                self._fn_depth -= 1

    visit_FunctionDef = _visit_scope
    visit_AsyncFunctionDef = _visit_scope
    visit_Lambda = _visit_scope
    visit_ClassDef = _visit_scope

    # -- loops --------------------------------------------------------------
    def _candidate(self, name: str) -> bool:
        if name.startswith("_pyopt_") or name.startswith("__"):
            return False
        if any(name in scope for scope in self._scopes):
            return False  # a local (or shadowed) name, not a stable global
        if self.jit_mode and name in _JIT_WHITELIST_BUILTINS:
            return False
        binding_idx = self._module_single.get(name)
        if binding_idx is not None:
            return binding_idx < self._top_index
        return name not in self._all_bound and hasattr(builtins, name)

    def _visit_loop(self, loop: ast.AST):
        if self._fn_depth == 0:
            return self.generic_visit(loop)
        per_iteration: list[ast.AST] = list(loop.body)
        if isinstance(loop, ast.While):
            per_iteration.append(loop.test)
        reads: Counter[str] = Counter()
        for node in _region_nodes(per_iteration):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                reads[node.id] += 1
        mapping = {
            name: self._fresh_name()
            for name in sorted(reads)
            if self._candidate(name)
        }
        if not mapping:
            return self.generic_visit(loop)
        renamer = _Renamer(mapping)
        if isinstance(loop, ast.While):
            loop.test = renamer.visit(loop.test)
        else:
            loop.iter = renamer.visit(loop.iter)
        loop.body = [renamer.visit(s) for s in loop.body]
        loop.orelse = [renamer.visit(s) for s in loop.orelse]
        assigns = []
        for name, temp in mapping.items():
            target = ast.copy_location(ast.Name(id=temp, ctx=ast.Store()), loop)
            value = ast.copy_location(ast.Name(id=name, ctx=ast.Load()), loop)
            assigns.append(
                ast.copy_location(ast.Assign(targets=[target], value=value), loop)
            )
            self.changes += 1
        # Recurse afterwards: nested defs get their own treatment, and inner
        # loops find nothing left to do for the names handled here.
        return [*assigns, self.generic_visit(loop)]

    visit_For = _visit_loop
    visit_AsyncFor = _visit_loop
    visit_While = _visit_loop
