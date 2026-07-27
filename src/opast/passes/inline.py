"""Simple function inlining.

Two candidate shapes, both module-level plain ``def``s:

* **expression body** -- a single ``return <expression>`` (optionally after a
  docstring), or a ``pass``/bare ``return`` body which inlines as ``None``.
  The call expression is replaced by the substituted body expression at any
  call position; arguments must be constants or plain names so evaluation
  count and order are preserved.
* **straight-line statement body** -- a docstring plus up to
  ``_STMT_BODY_MAX`` single-target name assignments ending in ``return
  <expression>`` (no branches, loops, or any other statement kind).
  Rewritten call sites, always inside a function:

  - the call as the whole value of an ``x = f(...)``, ``return f(...)`` or
    bare ``f(...)`` statement;
  - the call **buried in an expression** of such a statement, or of an
    ``if`` test -- provided it sits in an *unconditionally evaluated*
    position (never behind ``and``/``or`` short-circuits, ``if/else``
    expression arms, chained-comparison tails, lambdas or comprehensions).
    Every subtree the statement evaluates before the call is bound to its
    own ``_opast_in_<site>_pfx<k>`` temp first, in evaluation order, so
    side effects and exceptions keep their exact original positions --
    single evaluation, not purity, is what a temp preserves.  Constants
    and provably-bound candidate names in call-func position skip the
    temp (the latter so the next fixpoint iteration can inline them too).

  Arguments -- positional and **keyword** -- are evaluated in call-site
  written order into fresh ``_opast_in_<site>_<param>`` locals (missing
  parameters take their constant defaults; unexpected or duplicate
  keywords keep the runtime ``TypeError`` by not rewriting), the body
  assignments follow with all locals renamed, and the call's position gets
  the renamed return expression.  Function locals are invisible to the
  caller by construction, so the renamed temps are the only observable
  difference (tracebacks lose one frame -- a documented opast limitation).

Shared safety requirements (any doubt -> the call is left alone):

* the whole module is free of dynamic constructs (``exec``/``eval`` executed
  anywhere could rebind module globals, including the inlined function);
* the function: no decorators, only plain positional-or-keyword parameters,
  defaults (if any) are constants, not async, body expressions contain no
  ``yield`` / ``await`` / walrus / lambda / comprehension and no name in a
  Store/Del context, no read of a local before its assignment, and no
  self-reference (recursion);
* the function name is bound exactly once at module level (its ``def``) and
  never declared ``global`` in any scope -- so it cannot be rebound;
* the call site appears on a later line than the ``def`` (so the function is
  guaranteed to exist when the call site can execute), and the function name
  and every free name of the body are not shadowed by any enclosing
  non-module scope at the call site.

The ``def`` itself is kept (unused-elimination collects it once idle); call
sites inside candidate bodies are not rewritten (this also prevents growth
through mutually recursive candidates).
"""

from __future__ import annotations

import ast
import copy
from collections import Counter
from dataclasses import dataclass, field

from ..analysis import binding_names, bound_names
from ..safety import iter_region, tree_has_dynamic
from .base import ScopedTransformer

_FORBIDDEN_IN_EXPR = (
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
    ast.NamedExpr,
    ast.Lambda,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)

#: Maximum number of body assignments for a statement-body candidate.
_STMT_BODY_MAX = 6
#: The aggressive ``budgets`` option trades source size for more inlining.
_GENEROUS_STMT_BODY_MAX = 24

_TEMP_PREFIX = "_opast_in"


@dataclass
class _Candidate:
    name: str
    lineno: int
    params: list[str]
    defaults: dict[str, ast.Constant]
    expr: ast.expr
    free_names: frozenset[str]
    usage: dict[str, int]
    node: ast.FunctionDef = field(repr=False)
    #: (target, value) body assignments -- empty for expression bodies.
    assigns: list = field(default_factory=list)


def _simple_body_expr(func: ast.FunctionDef):
    body = list(func.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]  # docstring
    if len(body) != 1:
        return None
    stmt = body[0]
    if isinstance(stmt, ast.Return):
        return stmt.value if stmt.value is not None else ast.Constant(None)
    if isinstance(stmt, ast.Pass):
        return ast.Constant(None)
    return None


def _stmt_body(func: ast.FunctionDef, budget: int = _STMT_BODY_MAX):
    """``([(target, value), ...], return_expr)`` for a straight-line body of
    single-target name assignments ending in ``return <expr>``, else None."""
    body = list(func.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]  # docstring
    if not 2 <= len(body) <= budget + 1:
        return None
    *assign_stmts, last = body
    if not isinstance(last, ast.Return) or last.value is None:
        return None
    pairs = []
    for stmt in assign_stmts:
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
        ):
            return None
        pairs.append((stmt.targets[0].id, stmt.value))
    return pairs, last.value


def _scan_body_names(exprs_in_order, params):
    """Validate expression contents and classify names for a candidate body.

    *exprs_in_order* is ``[(locals_bound_before, expr), ...]``.  Returns
    ``(free_names, usage)`` or None when anything disqualifies: forbidden
    constructs, non-Load names, or a read of a local before its binding
    (would turn UnboundLocalError into NameError after renaming).
    """
    usage: Counter[str] = Counter()
    free: set[str] = set()
    all_locals = set(params)
    for bound_before, _ in exprs_in_order:
        all_locals.update(bound_before)
    for bound_before, expr in exprs_in_order:
        for n in ast.walk(expr):
            if isinstance(n, _FORBIDDEN_IN_EXPR):
                return None
            if isinstance(n, ast.Name):
                if not isinstance(n.ctx, ast.Load):
                    return None
                if n.id in all_locals:
                    if n.id not in bound_before:
                        return None
                    usage[n.id] += 1
                else:
                    free.add(n.id)
    return frozenset(free), dict(usage)


def _collect_candidates(module: ast.Module, stmt_budget: int = _STMT_BODY_MAX):
    module_bindings = Counter()
    for node in iter_region(module):
        module_bindings.update(binding_names(node))
    global_declared: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Global):
            global_declared.update(node.names)

    expr_candidates: dict[str, _Candidate] = {}
    stmt_candidates: dict[str, _Candidate] = {}
    for stmt in module.body:
        if not isinstance(stmt, ast.FunctionDef):
            continue
        if stmt.decorator_list:
            continue
        if module_bindings[stmt.name] != 1 or stmt.name in global_declared:
            continue
        args = stmt.args
        if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs:
            continue
        params = [a.arg for a in args.args]
        defaults: dict[str, ast.Constant] = {}
        if args.defaults:
            if not all(isinstance(d, ast.Constant) for d in args.defaults):
                continue
            for param, default in zip(params[len(params) - len(args.defaults):],
                                      args.defaults):
                defaults[param] = default

        expr = _simple_body_expr(stmt)
        if expr is not None:
            scanned = _scan_body_names([(set(params), expr)], params)
            if scanned is None:
                continue
            free, usage = scanned
            if stmt.name in free:
                continue  # recursion
            expr_candidates[stmt.name] = _Candidate(
                name=stmt.name,
                lineno=stmt.lineno,
                params=params,
                defaults=defaults,
                expr=expr,
                free_names=free,
                usage=usage,
                node=stmt,
            )
            continue

        pairs_ret = _stmt_body(stmt, stmt_budget)
        if pairs_ret is None:
            continue
        pairs, ret = pairs_ret
        exprs_in_order = []
        bound = set(params)
        for target, value in pairs:
            exprs_in_order.append((set(bound), value))
            bound.add(target)
        exprs_in_order.append((set(bound), ret))
        scanned = _scan_body_names(exprs_in_order, params)
        if scanned is None:
            continue
        free, usage = scanned
        if stmt.name in free:
            continue  # recursion
        stmt_candidates[stmt.name] = _Candidate(
            name=stmt.name,
            lineno=stmt.lineno,
            params=params,
            defaults=defaults,
            expr=ret,
            free_names=free,
            usage=usage,
            node=stmt,
            assigns=pairs,
        )
    return expr_candidates, stmt_candidates


def _eval_children(node: ast.AST):
    """Children of *node* in **evaluation order**, each tagged with whether
    it is evaluated unconditionally once *node*'s evaluation starts.  Nodes
    that defer or repeat evaluation (lambdas, comprehensions, f-strings,
    walrus, yield/await) yield nothing, so the call finder never enters
    them; unknown node kinds likewise stay opaque."""
    if isinstance(node, ast.BinOp):
        return [(node.left, True), (node.right, True)]
    if isinstance(node, ast.UnaryOp):
        return [(node.operand, True)]
    if isinstance(node, ast.BoolOp):
        return [(node.values[0], True)] + [(v, False) for v in node.values[1:]]
    if isinstance(node, ast.Compare):
        out = [(node.left, True), (node.comparators[0], True)]
        out += [(c, False) for c in node.comparators[1:]]
        return out
    if isinstance(node, ast.Call):
        out = [(node.func, True)]
        out += [(a, True) for a in node.args]
        out += [(kw.value, True) for kw in node.keywords]
        return out
    if isinstance(node, ast.IfExp):
        return [(node.test, True), (node.body, False), (node.orelse, False)]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return [(e, True) for e in node.elts]
    if isinstance(node, ast.Dict):
        out = []
        for key, value in zip(node.keys, node.values):
            if key is not None:
                out.append((key, True))
            out.append((value, True))
        return out
    if isinstance(node, ast.Subscript):
        return [(node.value, True), (node.slice, True)]
    if isinstance(node, ast.Attribute):
        return [(node.value, True)]
    if isinstance(node, ast.Starred):
        return [(node.value, True)]
    if isinstance(node, ast.Slice):
        return [
            (part, True)
            for part in (node.lower, node.upper, node.step)
            if part is not None
        ]
    return []


def _replace_child(parent: ast.AST, old: ast.AST, new: ast.AST) -> None:
    for name, value in ast.iter_fields(parent):
        if value is old:
            setattr(parent, name, new)
            return
        if isinstance(value, list):
            for k, item in enumerate(value):
                if item is old:
                    value[k] = new
                    return
                # keyword nodes wrap their value one level down
                if isinstance(item, ast.keyword) and item.value is old:
                    item.value = new
                    return
    raise AssertionError("child not found during inline replacement")


class _Substituter(ast.NodeTransformer):
    def __init__(self, mapping: dict[str, ast.expr]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self.mapping:
            return copy.deepcopy(self.mapping[node.id])
        return node


class FunctionInlining(ScopedTransformer):
    name = "inline"

    def run(self, tree: ast.Module) -> ast.Module:
        if tree_has_dynamic(tree):
            self.skipped_scopes += 1
            return tree
        self._candidates, self._stmt_candidates = _collect_candidates(
            tree,
            _GENEROUS_STMT_BODY_MAX
            if "budgets" in self.aggressive
            else _STMT_BODY_MAX,
        )
        if not self._candidates and not self._stmt_candidates:
            return tree
        self._candidate_nodes = {
            id(c.node)
            for c in (*self._candidates.values(), *self._stmt_candidates.values())
        }
        self._scopes: list[frozenset[str]] = []  # enclosing non-module scopes
        self._scope_kinds: list[bool] = []  # True = function scope
        self._site_counter = 0
        for n in ast.walk(tree):
            ident = None
            if isinstance(n, ast.Name):
                ident = n.id
            elif isinstance(n, ast.arg):
                ident = n.arg
            if ident and ident.startswith(_TEMP_PREFIX + "_"):
                site = ident[len(_TEMP_PREFIX) + 1:].split("_", 1)[0]
                if site.isdigit():
                    self._site_counter = max(self._site_counter, int(site) + 1)
        return self.visit(tree)

    def _visit_scope(self, node: ast.AST) -> ast.AST:
        if id(node) in self._candidate_nodes:
            return node  # keep candidate bodies pristine
        is_fn = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        self._scopes.append(bound_names(node))
        self._scope_kinds.append(is_fn)
        try:
            return self.generic_visit(node)
        finally:
            self._scopes.pop()
            self._scope_kinds.pop()

    visit_FunctionDef = _visit_scope
    visit_AsyncFunctionDef = _visit_scope
    visit_Lambda = _visit_scope
    visit_ClassDef = _visit_scope

    def _visit_comp(self, node: ast.AST) -> ast.AST:
        # Comprehension targets are a runtime scope of their own: a free
        # name of an inlined expression must not collide with them, or the
        # substituted expression would read the loop variable instead of
        # the module global.
        names = frozenset(
            n.id
            for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        )
        self._scopes.append(names)
        try:
            return self.generic_visit(node)
        finally:
            self._scopes.pop()

    visit_ListComp = _visit_comp
    visit_SetComp = _visit_comp
    visit_DictComp = _visit_comp
    visit_GeneratorExp = _visit_comp

    def _shadowed(self, name: str) -> bool:
        return any(name in scope for scope in self._scopes)

    def _map_call(self, cand: _Candidate, call: ast.Call):
        if any(isinstance(a, ast.Starred) for a in call.args):
            return None
        if any(kw.arg is None for kw in call.keywords):  # **kwargs
            return None
        if len(call.args) > len(cand.params):
            return None
        values: dict[str, ast.expr] = {}
        for param, arg in zip(cand.params, call.args):
            values[param] = arg
        for kw in call.keywords:
            if kw.arg not in cand.params or kw.arg in values:
                return None
            values[kw.arg] = kw.value
        for param in cand.params:
            if param not in values:
                default = cand.defaults.get(param)
                if default is None:
                    return None
                values[param] = default
        for param, value in values.items():
            if isinstance(value, ast.Constant):
                continue
            if (
                isinstance(value, ast.Name)
                and isinstance(value.ctx, ast.Load)
                and cand.usage.get(param, 0) >= 1
            ):
                continue
            return None
        return values

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        func = node.func
        if not isinstance(func, ast.Name):
            return node
        cand = self._candidates.get(func.id)
        if cand is None:
            return node
        if getattr(node, "lineno", 0) <= cand.lineno:
            return node
        if self._shadowed(cand.name):
            return node
        if any(self._shadowed(name) for name in cand.free_names):
            return node
        mapping = self._map_call(cand, node)
        if mapping is None:
            return node
        new_expr = _Substituter(mapping).visit(copy.deepcopy(cand.expr))
        self.changes += 1
        return ast.copy_location(new_expr, node)

    # -- statement-body inlining -------------------------------------------
    def _match_stmt_call(self, call: ast.expr):
        """``(candidate, ordered)`` for an eligible statement-body call,
        where *ordered* is the ``(param, value)`` list in **call-site
        written order** -- positionals, then keywords as written, then
        constant defaults -- so the argument temps evaluate exactly when
        the call would have evaluated them.  None when ineligible."""
        if not (self._scope_kinds and self._scope_kinds[-1]):
            return None  # only directly inside a function body
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)):
            return None
        cand = self._stmt_candidates.get(call.func.id)
        if cand is None:
            return None
        if getattr(call, "lineno", 0) <= cand.lineno:
            return None
        if self._shadowed(cand.name):
            return None
        if any(self._shadowed(name) for name in cand.free_names):
            return None
        if any(isinstance(a, ast.Starred) for a in call.args):
            return None
        if any(kw.arg is None for kw in call.keywords):  # **kwargs
            return None
        if len(call.args) > len(cand.params):
            return None
        ordered: list[tuple[str, ast.expr]] = list(
            zip(cand.params, call.args)
        )
        seen = {param for param, _ in ordered}
        for kw in call.keywords:
            if kw.arg not in cand.params or kw.arg in seen:
                return None  # unexpected/duplicate: keep the TypeError
            ordered.append((kw.arg, kw.value))
            seen.add(kw.arg)
        for param in cand.params:
            if param not in seen:
                default = cand.defaults.get(param)
                if default is None:
                    return None
                ordered.append((param, copy.deepcopy(default)))
        return cand, ordered

    def _expand(self, cand: _Candidate, ordered, loc: ast.stmt):
        """Argument temps (in call order) plus renamed body assignments;
        returns ``(prelude_statements, renamed_return_expr)``."""
        site = self._site_counter
        self._site_counter += 1
        rename = {
            local: f"{_TEMP_PREFIX}_{site}_{local}"
            for local in (*cand.params, *(t for t, _ in cand.assigns))
        }
        sub = _Substituter(
            {old: ast.Name(id=new, ctx=ast.Load()) for old, new in rename.items()}
        )
        prelude: list[ast.stmt] = []
        for param, value in ordered:
            target = ast.Name(id=rename[param], ctx=ast.Store())
            prelude.append(
                ast.copy_location(ast.Assign(targets=[target], value=value), loc)
            )
        for tname, texpr in cand.assigns:
            target = ast.Name(id=rename[tname], ctx=ast.Store())
            value = sub.visit(copy.deepcopy(texpr))
            prelude.append(
                ast.copy_location(ast.Assign(targets=[target], value=value), loc)
            )
        return prelude, sub.visit(copy.deepcopy(cand.expr))

    def _stmt_inline(self, stmt: ast.stmt, call: ast.expr):
        """Inline a statement-body candidate when *call* (the whole value of
        *stmt*) is an eligible call.  Returns the replacement statement list
        or None."""
        matched = self._match_stmt_call(call)
        if matched is None:
            return None
        cand, ordered = matched
        prelude, ret = self._expand(cand, ordered, stmt)
        stmt.value = ast.copy_location(ret, call)
        out = [*prelude, stmt]
        ast.fix_missing_locations(ast.Module(body=out, type_ignores=[]))
        self.changes += 1
        return out

    # -- expression-position statement-body inlining ------------------------
    def _find_expr_call(self, expr: ast.expr):
        """First eligible statement-candidate call in *expr*'s evaluation
        order, restricted to unconditionally-evaluated positions.  Returns
        ``(path, call, cand, ordered)`` where *path* is the list of
        ``(parent, child)`` edges from *expr* down to the call."""
        result = None

        def walk(node, path):
            nonlocal result
            if result is not None:
                return
            if isinstance(node, ast.Call):
                matched = self._match_stmt_call(node)
                if matched is not None:
                    result = (list(path), node, *matched)
                    return
            for child, unconditional in _eval_children(node):
                if result is not None:
                    return
                if not unconditional:
                    continue
                path.append((node, child))
                walk(child, path)
                path.pop()

        walk(expr, [])
        return result

    def _provably_bound_func(self, name_node, parent) -> bool:
        """A bare candidate name in a call's func slot is provably bound
        (module-level def before this line, unshadowed), so its load can
        safely move after the expansion without a NameError reordering."""
        if not (
            isinstance(parent, ast.Call)
            and parent.func is name_node
            and isinstance(name_node, ast.Name)
        ):
            return False
        cand = self._candidates.get(name_node.id) or self._stmt_candidates.get(
            name_node.id
        )
        return (
            cand is not None
            and getattr(name_node, "lineno", 0) > cand.lineno
            and not self._shadowed(cand.name)
        )

    def _expr_inline(self, stmt: ast.stmt, expr: ast.expr, setter):
        """Inline one candidate call buried inside *expr*.

        Every subtree the statement evaluates *before* the call becomes its
        own temp, in evaluation order, so side effects and exceptions keep
        their exact original positions -- purity is not required of the
        prefix, only single evaluation, which a temp preserves.  Constants
        need no temp, and neither does a provably-bound candidate name in
        a func slot (keeping it a plain name lets the next fixpoint
        iteration inline that call too)."""
        if not (self._scope_kinds and self._scope_kinds[-1]):
            return None
        found = self._find_expr_call(expr)
        if found is None:
            return None
        path, call, cand, ordered = found
        # Prefix temps get their own site number so they can never collide
        # with the parameter renames of the expansion's site.
        site = self._site_counter
        self._site_counter += 1
        prefix: list[ast.stmt] = []
        for parent, child in path:
            for sibling, _ in _eval_children(parent):
                if sibling is child:
                    break
                if isinstance(sibling, ast.Constant):
                    continue
                if self._provably_bound_func(sibling, parent):
                    continue
                temp = f"{_TEMP_PREFIX}_{site}_pfx{len(prefix)}"
                prefix.append(
                    ast.copy_location(
                        ast.Assign(
                            targets=[ast.Name(id=temp, ctx=ast.Store())],
                            value=sibling,
                        ),
                        stmt,
                    )
                )
                _replace_child(
                    parent, sibling, ast.Name(id=temp, ctx=ast.Load())
                )
        prelude, ret = self._expand(cand, ordered, stmt)
        ret = ast.copy_location(ret, call)
        if path:
            _replace_child(path[-1][0], call, ret)
        else:
            setter(ret)
        out = [*prefix, *prelude, stmt]
        ast.fix_missing_locations(ast.Module(body=out, type_ignores=[]))
        self.changes += 1
        return out

    def _visit_value_stmt(self, node: ast.stmt):
        node = self.generic_visit(node)  # expression-inline inner calls first
        if node.value is None:
            return node
        replacement = self._stmt_inline(node, node.value)
        if replacement is not None:
            return replacement
        replacement = self._expr_inline(
            node, node.value, lambda new: setattr(node, "value", new)
        )
        return replacement if replacement is not None else node

    def visit_Return(self, node: ast.Return):
        return self._visit_value_stmt(node)

    def visit_Expr(self, node: ast.Expr):
        return self._visit_value_stmt(node)

    def visit_Assign(self, node: ast.Assign):
        node = self.generic_visit(node)
        replacement = self._stmt_inline(node, node.value)
        if replacement is not None:
            return replacement
        replacement = self._expr_inline(
            node, node.value, lambda new: setattr(node, "value", new)
        )
        return replacement if replacement is not None else node

    def visit_If(self, node: ast.If):
        node = self.generic_visit(node)
        replacement = self._expr_inline(
            node, node.test, lambda new: setattr(node, "test", new)
        )
        if replacement is not None:
            return replacement
        return node
