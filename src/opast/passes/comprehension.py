"""Comprehension-to-``map``/``filter`` rewriting.

Recognised shapes (single ``for``, plain-name target ``x``, at most one
``if``; ``f``/``g`` are plain names, called with exactly ``(x)``, ``!= x``):

* ``(f(x) for x in it)``           -> ``map(f, it)``
* ``(x for x in it if g(x))``      -> ``filter(g, it)``
* ``(f(x) for x in it if g(x))``   -> ``map(f, filter(g, it))``
* the same for list/set comprehensions, wrapped in ``list(...)``/``set(...)``

Safety conditions:

* **Name-lookup timing**: a comprehension looks ``f`` up on every iteration,
  ``map`` captures it at construction.  Unobservable only if ``f`` can never
  be rebound: either a builtin callable name bound nowhere in the module, or
  (generator expressions only) a module-level def/class/import/assignment
  bound exactly once, never ``global``-declared, textually before the use
  site, and not shadowed by any enclosing function/class/lambda scope *or
  enclosing comprehension target*.  List/set comprehensions follow the
  stricter builtins-only rule.
* **Object identity** (generator expressions only): a generator has
  ``send``/``throw``/``close``/``gi_frame`` and is ``isinstance``-able; a
  ``map`` object is not.  Generator expressions are therefore rewritten only
  in iteration-consuming positions: the ``iter`` of a ``for`` statement, a
  positional argument of a whitelisted iterating builtin (``sum``, ``list``,
  ``sorted``, ``any``, ...), or the argument of ``"<literal>".join(...)``.
  List/set comprehensions produce the same ``list``/``set`` either way and
  are rewritten anywhere.
* The introduced names (``map``/``filter``/``list``/``set``) and the
  consumer builtins must themselves be unshadowed module-wide, and the
  module must contain no dynamic constructs at all.
"""

from __future__ import annotations

import ast
import builtins
from collections import Counter
from dataclasses import dataclass

from ..analysis import all_bound_names, binding_names, bound_names
from ..safety import iter_region, tree_has_dynamic
from .base import ScopedTransformer

_BUILTIN_CALLABLES = frozenset(
    name
    for name in dir(builtins)
    if not name.startswith("_") and callable(getattr(builtins, name))
)

#: Builtins that only *iterate* their (positional) iterable arguments.
_CONSUMER_NAMES = frozenset({
    "sum", "list", "tuple", "set", "frozenset", "dict", "sorted",
    "min", "max", "any", "all", "map", "filter", "zip", "enumerate",
    "bytes", "bytearray",
})


@dataclass
class _Match:
    func: str | None   # f in map(f, ...); None for the pure-filter form
    pred: str | None   # g in filter(g, ...); None when there is no if
    iter_expr: ast.expr


def _single_call_name(node: ast.AST, target: str) -> str | None:
    """``f(x)`` with plain-name ``f != x`` -> ``'f'``, else None."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and not node.keywords
        and len(node.args) == 1
        and isinstance(node.args[0], ast.Name)
        and isinstance(node.args[0].ctx, ast.Load)
        and node.args[0].id == target
        and node.func.id != target
    ):
        return node.func.id
    return None


def _match(comp) -> _Match | None:
    if len(comp.generators) != 1:
        return None
    gen = comp.generators[0]
    if gen.is_async or not isinstance(gen.target, ast.Name):
        return None
    if len(gen.ifs) > 1:
        return None
    x = gen.target.id
    pred = None
    if gen.ifs:
        pred = _single_call_name(gen.ifs[0], x)
        if pred is None:
            return None
    elt = comp.elt
    if isinstance(elt, ast.Name) and isinstance(elt.ctx, ast.Load) and elt.id == x:
        if pred is None:
            return None  # (x for x in it) alone: nothing to gain
        return _Match(None, pred, gen.iter)
    func = _single_call_name(elt, x)
    if func is None:
        return None
    return _Match(func, pred, gen.iter)


class ComprehensionToMap(ScopedTransformer):
    name = "comp-to-map"

    def run(self, tree: ast.Module) -> ast.Module:
        if tree_has_dynamic(tree):
            self.skipped_scopes += 1
            return tree
        self._bound_everywhere = all_bound_names(tree)
        self._module_fns = self._collect_module_fns(tree)
        self._scopes: list[frozenset[str]] = []
        return self.visit(tree)

    @staticmethod
    def _collect_module_fns(tree: ast.Module) -> dict[str, int]:
        counts: Counter[str] = Counter()
        for n in iter_region(tree):
            counts.update(binding_names(n))
        global_declared: set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Global):
                global_declared.update(n.names)
        fns: dict[str, int] = {}
        for stmt in tree.body:
            names: list[str] = []
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [stmt.name]
            elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
                names = [
                    a.asname or a.name.split(".")[0]
                    for a in stmt.names
                    if a.name != "*"
                ]
            elif isinstance(stmt, ast.Assign):
                names = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            for name in names:
                if counts[name] == 1 and name not in global_declared:
                    fns[name] = stmt.lineno
        return fns

    # -- scope tracking ------------------------------------------------------
    def _visit_scope(self, node: ast.AST) -> ast.AST:
        self._scopes.append(bound_names(node))
        try:
            return self.generic_visit(node)
        finally:
            self._scopes.pop()

    visit_FunctionDef = _visit_scope
    visit_AsyncFunctionDef = _visit_scope
    visit_Lambda = _visit_scope
    visit_ClassDef = _visit_scope

    def _shadowed(self, name: str) -> bool:
        return any(name in scope for scope in self._scopes)

    # -- eligibility ----------------------------------------------------------
    def _builtin_ok(self, name: str) -> bool:
        return name in _BUILTIN_CALLABLES and name not in self._bound_everywhere

    def _stable_module_fn(self, name: str, lineno: int) -> bool:
        defined = self._module_fns.get(name)
        return defined is not None and defined < lineno and not self._shadowed(name)

    def _genexp_name_ok(self, name: str, lineno: int) -> bool:
        return self._builtin_ok(name) or self._stable_module_fn(name, lineno)

    def _build(self, m: _Match, anchor: ast.AST) -> ast.expr:
        expr = m.iter_expr
        if m.pred is not None:
            expr = ast.Call(
                func=ast.Name(id="filter", ctx=ast.Load()),
                args=[ast.Name(id=m.pred, ctx=ast.Load()), expr],
                keywords=[],
            )
        if m.func is not None:
            expr = ast.Call(
                func=ast.Name(id="map", ctx=ast.Load()),
                args=[ast.Name(id=m.func, ctx=ast.Load()), expr],
                keywords=[],
            )
        for node in ast.walk(expr):
            if "lineno" in getattr(node, "_attributes", ()) and not hasattr(
                node, "lineno"
            ):
                ast.copy_location(node, anchor)
        return ast.copy_location(expr, anchor)

    def _helpers_ok(self, m: _Match) -> bool:
        if m.func is not None and not self._builtin_ok("map"):
            return False
        if m.pred is not None and not self._builtin_ok("filter"):
            return False
        return True

    # -- generator expressions (consuming positions only) ---------------------
    def _try_genexp(self, comp: ast.expr) -> ast.expr:
        if not isinstance(comp, ast.GeneratorExp):
            return comp
        m = _match(comp)
        if m is None or not self._helpers_ok(m):
            return comp
        lineno = getattr(comp, "lineno", 0)
        if m.func is not None and not self._genexp_name_ok(m.func, lineno):
            return comp
        if m.pred is not None and not self._genexp_name_ok(m.pred, lineno):
            return comp
        self.changes += 1
        return self._build(m, comp)

    def _is_consumer(self, call: ast.Call) -> bool:
        func = call.func
        if isinstance(func, ast.Name):
            return func.id in _CONSUMER_NAMES and self._builtin_ok(func.id)
        return (
            isinstance(func, ast.Attribute)
            and func.attr == "join"
            and isinstance(func.value, ast.Constant)
            and isinstance(func.value.value, str)
        )

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node, ast.Call) and self._is_consumer(node):
            node.args = [self._try_genexp(a) for a in node.args]
        return node

    def _visit_for(self, node):
        node = self.generic_visit(node)
        node.iter = self._try_genexp(node.iter)
        return node

    visit_For = _visit_for
    visit_AsyncFor = _visit_for

    # -- list/set comprehensions (builtins only, any position) ----------------
    def _visit_comp(self, node):
        # Comprehension targets shadow enclosing names for anything nested.
        names = frozenset(
            n.id
            for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        )
        self._scopes.append(names)
        try:
            node = self.generic_visit(node)
        finally:
            self._scopes.pop()
        if isinstance(node, (ast.ListComp, ast.SetComp)):
            return self._try_eager_comp(node)
        return node

    visit_ListComp = _visit_comp
    visit_SetComp = _visit_comp
    visit_DictComp = _visit_comp
    visit_GeneratorExp = _visit_comp

    def _try_eager_comp(self, comp):
        wrapper = "list" if isinstance(comp, ast.ListComp) else "set"
        m = _match(comp)
        if m is None or not self._helpers_ok(m):
            return comp
        if not self._builtin_ok(wrapper):
            return comp
        if m.func is not None and not self._builtin_ok(m.func):
            return comp
        if m.pred is not None and not self._builtin_ok(m.pred):
            return comp
        self.changes += 1
        inner = self._build(m, comp)
        call = ast.Call(
            func=ast.Name(id=wrapper, ctx=ast.Load()),
            args=[inner],
            keywords=[],
        )
        return ast.copy_location(call, comp)
