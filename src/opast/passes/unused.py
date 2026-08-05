"""Unused-code elimination.

* unused imports (``__future__`` and ``*`` imports are always kept; partially
  used ``import a, b`` keeps only the used aliases) -- note that removing an
  import also skips the imported module's side effects;
* unused module-level function definitions (no decorators; default values and
  annotations must be side-effect free) -- this cleans up helpers whose call
  sites were all inlined;
* unused local assignments inside functions: a pure right-hand side is
  dropped entirely, one with possible side effects is downgraded to a bare
  expression statement (evaluation preserved, only the dead store removed).

Module-level *variable* assignments are deliberately kept: a module's globals
are an observable surface (``import module; module.x``, or the bench harness
reading ``RESULT``).  Strings listed in ``__all__`` count as used; a
non-literal ``__all__`` disables module-level removal entirely.

Removal is scoped: a name is unused only if it is never loaded (nor ``del``-ed
nor read by an augmented assignment) anywhere in the scope's whole subtree,
including nested scopes that could close over it.  The pass runs only on
modules with zero dynamic constructs anywhere -- ``eval``/``exec``/frame
introspection could observe names it removes.

Removed statements are replaced by ``pass`` placeholders (never an empty
block); the next iteration's dead-code pass cleans those up.
"""

from __future__ import annotations

import ast

from ..analysis import bound_names
from ..safety import iter_region, tree_has_dynamic
from .base import ScopedTransformer


def _is_pure(node: ast.AST, pure: frozenset = frozenset()) -> bool:
    """Evaluation can neither raise nor cause side effects.

    *pure* (aggressive ``pure-calls``) additionally admits calls to
    trusted-pure module functions with name/constant/pure arguments;
    dropping such a call is exactly what the option's assumption signs off
    on (its exception or non-termination vanishes with it).
    """
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_is_pure(e, pure) for e in node.elts)
    if isinstance(node, ast.Set):
        # elements must be hashable, so require constants
        return all(isinstance(e, ast.Constant) for e in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            k is not None and isinstance(k, ast.Constant) for k in node.keys
        ) and all(_is_pure(v, pure) for v in node.values)
    if isinstance(node, ast.Lambda):
        return _pure_arguments(node.args)
    if (
        pure
        and isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in pure
    ):
        def _arg(a: ast.AST) -> bool:
            if isinstance(a, ast.Name):
                return isinstance(a.ctx, ast.Load)
            return _is_pure(a, pure)

        return all(_arg(a) for a in node.args) and all(
            k.arg is not None and _arg(k.value) for k in node.keywords
        )
    return False


def _pure_arguments(args: ast.arguments) -> bool:
    for default in list(args.defaults) + list(args.kw_defaults):
        if default is not None and not _is_pure(default):
            return False
    for a in [*args.posonlyargs, *args.args, *args.kwonlyargs,
              args.vararg, args.kwarg]:
        if a is not None and a.annotation is not None and not _is_pure(a.annotation):
            return False
    return True


def _used_names(scope: ast.AST) -> set[str]:
    """Names read anywhere in *scope*'s whole subtree (loads, deletes, and
    the implicit read of an augmented assignment target)."""
    used: set[str] = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Load, ast.Del)):
                used.add(node.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            used.add(node.target.id)
    return used


def _binds_dunder_all(stmt: ast.AST) -> bool:
    if isinstance(stmt, ast.Assign):
        return any(isinstance(t, ast.Name) and t.id == "__all__" for t in stmt.targets)
    if isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
        return isinstance(stmt.target, ast.Name) and stmt.target.id == "__all__"
    return False


#: Import roots the numba whitelist keys on.  Under ``--jit`` these must
#: survive even when nothing in the *current* source reads them: the jit
#: pass runs after the fixpoint and may introduce reads itself (a
#: ``DynArray.zeros(n)`` buffer becomes ``np.zeros(n)`` in the compiled
#: copy), so removing the import here would silently downgrade it.
_JIT_IMPORT_ROOTS = frozenset({"numpy", "math"})


class UnusedElimination(ScopedTransformer):
    name = "unused"

    #: Set by the pipeline when the jit pass will run (see above).
    jit_mode: bool = False

    def __init__(self) -> None:
        super().__init__()
        # (kind, used-names, trusted-pure-callees) per enclosing scope; used
        # is None when elimination is disabled for that scope (class bodies,
        # lambdas, unknown __all__).  The pure set is pre-shadowed for the
        # scope: a local binding of a trusted name evicts it.
        self._stack: list[tuple[str, set[str] | None, frozenset]] = []

    def run(self, tree: ast.Module) -> ast.Module:
        if tree_has_dynamic(tree):
            self.skipped_scopes += 1
            return tree
        return self.visit(tree)

    def _current(self) -> tuple[str, set[str] | None, frozenset]:
        return self._stack[-1] if self._stack else ("opaque", None, frozenset())

    # -- scopes -----------------------------------------------------------
    def visit_Module(self, node: ast.Module) -> ast.AST:
        self._stack.append(("module", self._module_used(node), frozenset()))
        try:
            return self.generic_visit(node)
        finally:
            self._stack.pop()

    def _module_used(self, module: ast.Module) -> set[str] | None:
        used = _used_names(module)
        if self.jit_mode:
            for stmt in module.body:
                if isinstance(stmt, ast.Import):
                    for alias in stmt.names:
                        if alias.name.split(".")[0] in _JIT_IMPORT_ROOTS:
                            used.add(alias.asname or alias.name.split(".")[0])
        for stmt in iter_region(module):
            if not _binds_dunder_all(stmt):
                continue
            value = getattr(stmt, "value", None)
            if isinstance(value, (ast.List, ast.Tuple, ast.Set)) and all(
                isinstance(e, ast.Constant) and isinstance(e.value, str)
                for e in value.elts
            ):
                used.update(e.value for e in value.elts)
            else:
                return None  # unknown export surface
        return used

    def _visit_function(self, node: ast.AST) -> ast.AST:
        kind, used, _ = self._current()
        if (
            used is not None
            and node.name not in used
            and not node.decorator_list
            and _pure_arguments(node.args)
            and (node.returns is None or _is_pure(node.returns))
        ):
            self.changes += 1
            return ast.copy_location(ast.Pass(), node)
        used = _used_names(node)
        # Names declared global/nonlocal in this function's own region are
        # not locals: assigning them escapes the scope and is observable
        # outside, so they always count as used.
        for stmt in iter_region(node):
            if isinstance(stmt, (ast.Global, ast.Nonlocal)):
                used.update(stmt.names)
        # Trusted-pure callees must resolve to the module global: any local
        # binding of the name in this scope shadows it.
        scope_pure = (
            self.pure_calls - bound_names(node) if self.pure_calls else frozenset()
        )
        self._stack.append(("function", used, scope_pure))
        try:
            return self.generic_visit(node)
        finally:
            self._stack.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def _visit_opaque(self, node: ast.AST) -> ast.AST:
        self._stack.append(("opaque", None, frozenset()))
        try:
            return self.generic_visit(node)
        finally:
            self._stack.pop()

    visit_ClassDef = _visit_opaque
    visit_Lambda = _visit_opaque

    # -- imports ----------------------------------------------------------
    @staticmethod
    def _import_binds(alias: ast.alias) -> str:
        return alias.asname or alias.name.split(".")[0]

    def _trim_imports(self, node, bound_name) -> ast.AST:
        kind, used, _ = self._current()
        if used is None:
            return node
        keep = [a for a in node.names if bound_name(a) in used]
        if len(keep) == len(node.names):
            return node
        self.changes += 1
        if keep:
            node.names = keep
            return node
        return ast.copy_location(ast.Pass(), node)

    def visit_Import(self, node: ast.Import) -> ast.AST:
        return self._trim_imports(node, self._import_binds)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST:
        if node.module == "__future__" or any(a.name == "*" for a in node.names):
            return node
        return self._trim_imports(node, lambda a: a.asname or a.name)

    # -- local assignments -------------------------------------------------
    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        node = self.generic_visit(node)
        kind, used, pure = self._current()
        if kind != "function" or used is None:
            return node
        if not all(isinstance(t, ast.Name) for t in node.targets):
            return node
        live = [t for t in node.targets if t.id in used]
        if len(live) == len(node.targets):
            return node
        self.changes += 1
        if live:
            node.targets = live
            return node
        if _is_pure(node.value, pure):
            return ast.copy_location(ast.Pass(), node)
        return ast.copy_location(ast.Expr(value=node.value), node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        node = self.generic_visit(node)
        kind, used, pure = self._current()
        if kind != "function" or used is None:
            return node
        if not isinstance(node.target, ast.Name) or node.target.id in used:
            return node
        # Local annotations are never evaluated or stored at runtime.
        self.changes += 1
        if node.value is None or _is_pure(node.value, pure):
            return ast.copy_location(ast.Pass(), node)
        return ast.copy_location(ast.Expr(value=node.value), node)
