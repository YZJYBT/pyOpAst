"""Scalar replacement of small local aggregates.

A container that is only ever indexed by constants never needs to exist:

    t = (x * 2, y + 1)           _opast_sr0_t_0 = x * 2
    a = t[0]              ->     _opast_sr0_t_1 = y + 1
    b = t[1]                     a = _opast_sr0_t_0
                                 b = _opast_sr0_t_1

Each element becomes its own local: the allocation disappears and every
``BINARY_SUBSCR`` becomes a ``LOAD_FAST``, after which const-prop / CSE /
unused treat the elements like any other scalars.  The same applies to an
instance of a *simple class* -- a module-level class whose whole body is one
plain ``__init__`` of ``self.attr = expr`` assignments:

    p = Point(i, j)              _opast_sr1_a_x = i
    s += p.x * p.y        ->     _opast_sr1_a_y = j
                                 _opast_sr1_p_x = _opast_sr1_a_x
                                 _opast_sr1_p_y = _opast_sr1_a_y
                                 s += _opast_sr1_p_x * _opast_sr1_p_y

Soundness rests on the aggregate being provably unobservable:

* the local is bound exactly once in the whole function subtree, by a plain
  ``t = <tuple/list display>`` (no starred elements) or ``p = C(<plain
  positional args>)``, and is never declared ``global``/``nonlocal``;
* the binding **dominates** every use: only occurrences inside statements
  *after* the binding in its own statement list (at any depth) qualify.  A
  use in a sibling branch, before the binding in a loop body (that reads
  the previous iteration's value), in an ``except``/``finally`` block or
  in the binding's own right-hand side rejects -- otherwise a rewrite
  could erase ``UnboundLocalError`` (a folded ``len(t)`` leaves no read
  behind to raise, and a list-element store would create a temp where the
  original raised);
* **every** other occurrence of the name -- anywhere in the function
  subtree, nested scopes included -- is an allowed use: a constant, in-bounds
  subscript (``Load`` for tuples; ``Load``/``Store`` for lists), ``len(t)``
  under the module-wide ``len`` gate, or ``p.attr`` (``Load``/``Store``) for
  an attribute ``__init__`` assigns.  A bare read, ``del``, slice,
  out-of-bounds or non-constant index, unknown attribute, or any use inside
  a nested scope rejects the candidate, so escapes, aliases, iteration and
  runtime ``IndexError``/``AttributeError``/``TypeError`` are all preserved
  (``t[0] += 1`` on a tuple stays a ``TypeError``);
* element / argument expressions are evaluated in the same order at the
  same program point (displays and calls evaluate operands left to right,
  and the emitted assignments repeat exactly that order), so effects and
  exceptions keep their positions -- purity is not required;
* a simple class is trusted only when the module has zero dynamic
  constructs and the class name is bound exactly once module-wide, the
  class has no bases / decorators / keywords and no method besides a plain
  ``__init__`` (positional parameters, constant defaults, body of
  ``self.attr = expr`` only, no ``self`` reads / walrus / nested scopes in
  the exprs), the call site uses plain positional arguments, and the
  function's own top-level statement comes *after* the class definition
  (calling before the class exists must keep its ``NameError``).  Free
  names in ``__init__`` exprs must not be shadowed by the caller or any
  enclosing function scope: they resolve to the same module/builtin scope
  in both worlds;
* one binding does not freeze the class *object*: ``P.__init__ = fn``
  monkey-patches without rebinding the name, and ``Q = P`` / ``g(P)`` hand
  the object to code that can.  Every Load of the class name module-wide
  must therefore be the func of a call -- any attribute store/read, alias,
  argument position, base-class use or ``except`` clause disqualifies the
  class;
* generated temp names avoid every identifier appearing anywhere in the
  function subtree (a user local named like a temp would otherwise be
  silently overwritten).

Function scopes only (module globals are observable by every callee), and
element counts are budgeted (8 by default; the aggressive ``budgets``
option raises the cap -- a resource knob, not a semantic bet).
"""

from __future__ import annotations

import ast
import copy
from collections import Counter

from ..analysis import binding_names, bound_names, builtin_gate
from ..safety import region_is_dynamic

_DEFAULT_MAX = 8
_GENEROUS_MAX = 64


class _ClassInfo:
    __slots__ = ("params", "defaults", "attr_assigns", "attrs", "free", "index")

    def __init__(self, params, defaults, attr_assigns, free) -> None:
        self.params = params            # parameter names after ``self``
        self.defaults = defaults        # trailing Constant defaults
        self.attr_assigns = attr_assigns  # ordered (attr, expr) pairs
        self.attrs = frozenset(a for a, _ in attr_assigns)
        self.free = free                # non-parameter names the exprs read
        self.index = -1                 # module-body position of the class


def _docstring_stripped(body: list) -> list:
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        return body[1:]
    return list(body)


def _simple_class(cd: ast.ClassDef, budget: int) -> _ClassInfo | None:
    """Shape-check *cd*; None unless it is a plain data holder."""
    if cd.decorator_list or cd.bases or cd.keywords:
        return None
    body = _docstring_stripped(cd.body)
    if len(body) != 1 or not isinstance(body[0], ast.FunctionDef):
        return None
    init = body[0]
    if init.name != "__init__" or init.decorator_list:
        return None
    a = init.args
    if a.posonlyargs or a.vararg or a.kwonlyargs or a.kw_defaults or a.kwarg:
        return None
    if not a.args or a.args[0].arg != "self":
        return None
    params = [arg.arg for arg in a.args[1:]]
    if len(params) > budget or len(set(params)) != len(params):
        return None
    if not all(isinstance(d, ast.Constant) for d in a.defaults):
        return None
    attr_assigns: list[tuple[str, ast.expr]] = []
    free: set[str] = set()
    for stmt in _docstring_stripped(init.body):
        if not (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Attribute)
            and isinstance(stmt.targets[0].value, ast.Name)
            and stmt.targets[0].value.id == "self"
            and isinstance(stmt.targets[0].ctx, ast.Store)
        ):
            return None
        expr = stmt.value
        for node in ast.walk(expr):
            if isinstance(node, ast.Name):
                # ``self`` reads (earlier attrs) and walrus targets would
                # need their own renaming story -- keep v1 flat.
                if node.id == "self":
                    return None
                if isinstance(node.ctx, ast.Load):
                    free.add(node.id)
                else:
                    return None
            elif isinstance(
                node,
                (
                    ast.Lambda,
                    ast.NamedExpr,
                    ast.Await,
                    ast.Yield,
                    ast.YieldFrom,
                    ast.ListComp,
                    ast.SetComp,
                    ast.DictComp,
                    ast.GeneratorExp,
                ),
            ):
                return None
        attr_assigns.append((stmt.targets[0].attr, expr))
    if not attr_assigns or len({a for a, _ in attr_assigns}) > budget:
        return None
    free -= set(params)
    return _ClassInfo(params, list(a.defaults), attr_assigns, frozenset(free))


class _Candidate:
    __slots__ = (
        "name", "kind", "assign", "size", "info", "call", "temps", "site",
        "dominated",
    )

    def __init__(self, name, kind, assign, size, info=None, call=None) -> None:
        self.name = name
        self.kind = kind        # "tuple" | "list" | "class"
        self.assign = assign    # the binding ast.Assign
        self.size = size        # element count (containers only)
        self.info = info        # _ClassInfo (class kind)
        self.call = call        # the ast.Call (class kind)
        self.temps = {}         # normalized index / attr -> temp name
        self.site = -1
        self.dominated = frozenset()  # id()s of nodes the binding dominates


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _region_blocks(node: ast.AST):
    """Yields every statement list in *node*'s region (never crossing into
    nested scopes)."""
    for field, value in ast.iter_fields(node):
        if (
            isinstance(value, list)
            and value
            and all(isinstance(s, ast.stmt) for s in value)
        ):
            yield value
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, _SCOPES):
            yield from _region_blocks(child)


def _declared_names(func: ast.AST) -> set[str]:
    """Names declared ``global``/``nonlocal`` anywhere in *func*'s subtree
    (nested scopes included: calling a closure that declares the name can
    rebind it)."""
    out: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            out.update(node.names)
    return out


def _const_index(sl: ast.expr, size: int):
    """Normalized in-bounds index for a constant subscript, else None."""
    if isinstance(sl, ast.Constant) and type(sl.value) is int:
        i = sl.value
        if -size <= i < size:
            return i if i >= 0 else i + size
    return None


class _UseChecker:
    """Visits every node of the function subtree exactly once.  Allowed use
    shapes pre-authorize their base ``Name``; any candidate name reached
    without authorization (bare read, bad shape, nested scope, ``del``,
    non-constant or out-of-bounds index, unknown attribute) rejects the
    candidate.  Inside nested scopes nothing is ever authorized: the
    rewriter does not descend there, so any use at all must reject."""

    def __init__(self, cands: dict, len_ok: bool) -> None:
        self.cands = cands
        self.len_ok = len_ok
        self.bad: set[str] = set()
        self.allowed = {id(c.assign.targets[0]) for c in cands.values()}

    def walk(self, node: ast.AST, nested: bool = False) -> None:
        for child in ast.iter_child_nodes(node):
            self._inspect(child, nested)
            self.walk(child, nested or isinstance(child, _SCOPES))

    def _inspect(self, node: ast.AST, nested: bool) -> None:
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id in self.cands
        ):
            cand = self.cands[node.value.id]
            idx = _const_index(node.slice, cand.size)
            if (
                not nested
                and id(node.value) in cand.dominated
                and cand.kind in ("tuple", "list")
                and idx is not None
                and (
                    isinstance(node.ctx, ast.Load)
                    or (cand.kind == "list" and isinstance(node.ctx, ast.Store))
                )
            ):
                self.allowed.add(id(node.value))
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in self.cands
        ):
            cand = self.cands[node.value.id]
            if (
                not nested
                and id(node.value) in cand.dominated
                and cand.kind == "class"
                and node.attr in cand.info.attrs
                and isinstance(node.ctx, (ast.Load, ast.Store))
            ):
                self.allowed.add(id(node.value))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in self.cands
        ):
            cand = self.cands[node.args[0].id]
            if (
                not nested
                and id(node.args[0]) in cand.dominated
                and self.len_ok
                and cand.kind in ("tuple", "list")
            ):
                self.allowed.add(id(node.args[0]))
        elif isinstance(node, ast.Name) and node.id in self.cands:
            if id(node) not in self.allowed:
                self.bad.add(node.id)


class _ParamRename(ast.NodeTransformer):
    """Rewrites ``__init__`` parameter reads to the argument temps.  The
    exprs were checked flat (no nested scopes, no stores), so a plain Name
    substitution is exact."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        new = self.mapping.get(node.id)
        if new is not None and isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Name(id=new, ctx=ast.Load()), node)
        return node


class _Rewriter(ast.NodeTransformer):
    """Expands candidate bindings and retargets their uses.  Never descends
    into nested scopes: a candidate with nested-scope uses was rejected."""

    def __init__(self, cands: dict, owner) -> None:
        self.cands = cands
        self.owner = owner

    def _skip(self, node: ast.AST) -> ast.AST:
        return node

    visit_FunctionDef = _skip
    visit_AsyncFunctionDef = _skip
    visit_Lambda = _skip
    visit_ClassDef = _skip

    def visit_Assign(self, node: ast.Assign):
        node = self.generic_visit(node)
        for cand in self.cands.values():
            if node is cand.assign:
                return self.owner._expand(cand) or ast.copy_location(
                    ast.Pass(), node
                )
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id in self.cands:
            cand = self.cands[node.value.id]
            idx = _const_index(node.slice, cand.size)
            if idx is not None and cand.kind in ("tuple", "list"):
                return ast.copy_location(
                    ast.Name(id=cand.temps[idx], ctx=node.ctx), node
                )
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id in self.cands:
            cand = self.cands[node.value.id]
            if cand.kind == "class" and node.attr in cand.info.attrs:
                return ast.copy_location(
                    ast.Name(id=cand.temps[node.attr], ctx=node.ctx), node
                )
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Name)
            and node.args[0].id in self.cands
        ):
            cand = self.cands[node.args[0].id]
            if cand.kind in ("tuple", "list"):
                return ast.copy_location(ast.Constant(cand.size), node)
        return self.generic_visit(node)


class ScalarReplacement:
    """Pass driver."""

    name = "scalar-repl"

    #: Set by the pipeline.
    aggressive: frozenset = frozenset()

    def __init__(self) -> None:
        self.changes = 0
        self.skipped_scopes = 0

    def run(self, tree: ast.Module) -> ast.Module:
        if region_is_dynamic(tree):
            self.skipped_scopes += 1
            return tree
        self._budget = (
            _GENEROUS_MAX if "budgets" in self.aggressive else _DEFAULT_MAX
        )
        self._len_ok = builtin_gate(tree, "len")
        self._site = 0
        counts: Counter = Counter()
        for node in ast.walk(tree):
            counts.update(binding_names(node))
        # A single name binding does not make the class object immutable:
        # ``P.__init__ = replacement`` rebinds an attribute, ``Q = P`` /
        # ``g(P)`` let unknown code do the same later.  Trust therefore
        # additionally requires that every Load of the class name anywhere
        # in the module is the func of a call -- the class object itself
        # never escapes a ``P(...)`` position.
        call_funcs = {
            id(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        escaping_loads: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and id(node) not in call_funcs
            ):
                escaping_loads.add(node.id)
        self._classes: dict[str, _ClassInfo] = {}
        for idx, stmt in enumerate(tree.body):
            if (
                isinstance(stmt, ast.ClassDef)
                and counts[stmt.name] == 1
                and stmt.name not in escaping_loads
            ):
                info = _simple_class(stmt, self._budget)
                if info is not None:
                    info.index = idx
                    self._classes[stmt.name] = info
        for idx, stmt in enumerate(tree.body):
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._function(stmt, frozenset(), idx)
            else:
                self._walk(stmt, frozenset(), idx)
        return tree

    def _walk(self, node: ast.AST, ancestor_bound: frozenset, module_index: int) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._function(child, ancestor_bound, module_index)
            else:
                self._walk(child, ancestor_bound, module_index)

    def _function(self, func, ancestor_bound: frozenset, module_index: int) -> None:
        if region_is_dynamic(func):
            self.skipped_scopes += 1
            return
        own = bound_names(func)
        inner = ancestor_bound | own
        for child in ast.iter_child_nodes(func):
            self._walk(child, inner, module_index)
        self._rewrite(func, ancestor_bound | own, module_index)

    def _rewrite(self, func, shadowing: frozenset, module_index: int) -> None:
        declared = _declared_names(func)
        counts: Counter = Counter()
        for node in ast.walk(func):
            counts.update(binding_names(node))
        cands: dict[str, _Candidate] = {}
        for block in _region_blocks(func):
            for i, stmt in enumerate(block):
                if not (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                ):
                    continue
                name = stmt.targets[0].id
                if counts[name] != 1 or name in declared:
                    continue
                cand = self._classify(
                    name, stmt, shadowing, declared, module_index
                )
                if cand is not None:
                    # The binding dominates exactly the statements after it
                    # in its own block (any depth): a use anywhere else --
                    # sibling branch, prior loop-body position (previous
                    # iteration's value!), except/finally, or the binding's
                    # own right-hand side -- must reject, or a rewrite
                    # could erase UnboundLocalError (``len(t)`` folds to a
                    # constant and leaves no read behind to raise).
                    dominated: set[int] = set()
                    for later in block[i + 1 :]:
                        for n in ast.walk(later):
                            dominated.add(id(n))
                    cand.dominated = dominated
                    cands[name] = cand
        if not cands:
            return
        checker = _UseChecker(cands, self._len_ok)
        checker.walk(func)
        for name in checker.bad:
            del cands[name]
        if not cands:
            return
        # Every identifier appearing anywhere in the function subtree
        # (loads included -- a user global named like a temp would be
        # captured by the new local): generated temps must avoid them all.
        avoid: set[str] = set()
        for node in ast.walk(func):
            if isinstance(node, ast.Name):
                avoid.add(node.id)
            elif isinstance(node, ast.arg):
                avoid.add(node.arg)
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                avoid.update(node.names)
        for cand in cands.values():
            self._assign_temps(cand, avoid)
        # generic_visit: enter through the function's children -- visiting
        # the function node itself would hit the nested-scope skip.
        _Rewriter(cands, self).generic_visit(func)
        self.changes += len(cands)

    def _classify(self, name, assign, shadowing, declared, module_index):
        value = assign.value
        if isinstance(value, (ast.Tuple, ast.List)) and isinstance(
            value.ctx, ast.Load
        ):
            if len(value.elts) > self._budget or any(
                isinstance(e, ast.Starred) for e in value.elts
            ):
                return None
            kind = "tuple" if isinstance(value, ast.Tuple) else "list"
            return _Candidate(name, kind, assign, len(value.elts))
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in self._classes
            and not value.keywords
            and not any(isinstance(a, ast.Starred) for a in value.args)
        ):
            cls = value.func.id
            info = self._classes[cls]
            if cls in shadowing or cls in declared:
                return None
            if info.index >= module_index:
                # Calling before the class statement ran must keep its
                # NameError.
                return None
            required = len(info.params) - len(info.defaults)
            if not required <= len(value.args) <= len(info.params):
                return None
            if info.free & shadowing:
                # A free name in ``__init__`` would resolve differently at
                # the call site.
                return None
            return _Candidate(name, "class", assign, 0, info=info, call=value)
        return None

    def _assign_temps(self, cand: _Candidate, avoid: set) -> None:
        while True:
            site = self._site
            self._site += 1
            if cand.kind in ("tuple", "list"):
                temps = {
                    i: f"_opast_sr{site}_{cand.name}_{i}"
                    for i in range(cand.size)
                }
                generated = set(temps.values())
            else:
                temps = {
                    attr: f"_opast_sr{site}_{cand.name}_{attr}"
                    for attr in cand.info.attrs
                }
                generated = set(temps.values()) | {
                    f"_opast_sr{site}_a_{param}" for param in cand.info.params
                }
            if not generated & avoid:
                break
        cand.site = site
        cand.temps = temps

    def _expand(self, cand: _Candidate) -> list:
        loc = cand.assign
        out: list[ast.stmt] = []

        def emit(name: str, value: ast.expr) -> None:
            stmt = ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())], value=value
            )
            out.append(ast.copy_location(stmt, loc))

        if cand.kind in ("tuple", "list"):
            for i, elt in enumerate(cand.assign.value.elts):
                emit(cand.temps[i], elt)
            return out
        info = cand.info
        site = cand.site
        arg_temps = {
            param: f"_opast_sr{site}_a_{param}" for param in info.params
        }
        args = cand.call.args
        for param, arg in zip(info.params, args):
            emit(arg_temps[param], arg)
        missing = info.params[len(args):]
        defaults = info.defaults[len(info.defaults) - len(missing):]
        for param, dflt in zip(missing, defaults):
            emit(arg_temps[param], ast.Constant(dflt.value))
        renamer = _ParamRename(arg_temps)
        for attr, expr in info.attr_assigns:
            emit(cand.temps[attr], renamer.visit(copy.deepcopy(expr)))
        return out
