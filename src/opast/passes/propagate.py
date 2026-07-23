"""Constant propagation.

Substitutes uses of a variable with its constant value when the variable is
provably never modified.  Requirements (any doubt -> no propagation):

* the name is bound exactly once in the whole scope region, by a plain
  ``x = <constant>`` (or annotated) assignment -- or a flat constant tuple
  unpacking ``x, y = 1, 2`` (equal lengths, plain names, no starred
  targets; each name qualifies independently) -- that is a *direct* child
  of the scope body, so it dominates everything after it;
* only uses in statements *after* the assignment are substituted (a use
  before it keeps its NameError/UnboundLocalError behaviour);
* within one scope, nested scopes are never entered (a closure keeps reading
  the name; the assignment itself is left in place -- the unused-code pass
  removes it once nothing reads it);
* module scope additionally requires the whole module to be free of dynamic
  constructs (a called function could rebind module globals through ``exec``)
  and the name never declared ``global`` anywhere; function scope requires
  the name never declared ``nonlocal`` in any nested scope;
* strings/bytes longer than 128 characters are not duplicated.

**Cross-scope**: a module constant additionally propagates into functions,
lambdas and classes whose *top-level* statement comes after the binding.
The dominance argument mirrors the localize pass: code inside a scope can
only run after its enclosing top-level ``def``/``class`` statement executed,
which is after the binding executed -- and a single-binding name that is
never declared ``global`` can never be rebound, so every read inside the
scope sees the constant.  Scopes that bind the name themselves (parameters,
assignments, comprehension targets -- ``bound_names`` is conservative) are
excluded along with everything nested inside them, since their reads no
longer resolve to the module global; dynamic scopes are skipped as always.
This is what lets a function-level ``if DEBUG:`` die when ``DEBUG = False``
is a module constant.

**Span propagation** additionally handles names bound more than once.
Within one statement block, statements run strictly in order, so from an
``x = <constant>`` assignment up to (but excluding) the first following
statement whose subtree can rebind ``x``, every load of ``x`` sees that
constant and is substituted.  "Can rebind" is judged conservatively per
statement: any binding of the name at this scope level (assignment
targets, ``del``, imports, ``def``/``class``, walrus, comprehension
targets) plus any ``global``/``nonlocal`` declaration of it inside a
nested scope (calling such a closure rebinds our binding).  A compound
statement that nowhere rebinds ``x`` is substituted throughout (loops
included: the constant holds on every iteration); its inner blocks are
then walked recursively so assignments inside them open their own local
spans.  Names declared ``global``/``nonlocal`` anywhere in the scope's
subtree (for the module scope: declared ``global`` anywhere in the tree)
are never tracked, and the module scope keeps its no-dynamic-constructs
requirement.  Nested scopes are still never entered -- a closure keeps
reading the variable.

**Copy propagation** rides the same span machinery: after ``y = x`` where
both names are plain locals of the scope, loads of ``y`` are replaced by
``x`` while *neither* name has been rebound.  ``x`` must be bound at this
scope level -- replacing a load of frozen-at-copy-time ``y`` with a live
read of a global/builtin ``x`` could observe a later rebinding.  The
``y = x`` assignment itself stays; the unused pass removes it once
nothing reads ``y`` (inlining temporaries die this way).  Function scopes
only: that is where the inlining temporaries live, and module-level
aliases are frequently *intentional* renames (``getattr = my_func``)
whose textual survival other tooling may rely on.
"""

from __future__ import annotations

import ast
from collections import Counter

from ..analysis import binding_names, bound_names
from ..safety import (
    SCOPE_NODES,
    _boundary_children,
    iter_region,
    region_is_dynamic,
    tree_has_dynamic,
)
from .base import ScopedTransformer

_PROPAGATE_TYPES = (int, float, complex, bool, str, bytes, type(None), type(...))
_MAX_SEQ = 128


def _direct_nested_scopes(node: ast.AST):
    """Scope nodes within *node* reachable without crossing another scope
    boundary (yields *node* itself when it is a scope node)."""
    if isinstance(node, SCOPE_NODES):
        yield node
        return
    stack = list(ast.iter_child_nodes(node))
    while stack:
        n = stack.pop()
        if isinstance(n, SCOPE_NODES):
            yield n
        else:
            stack.extend(ast.iter_child_nodes(n))


def _constant_pairs(stmt: ast.stmt) -> list[tuple[str, ast.Constant]]:
    """(name, constant) bindings produced by *stmt* if it is a plain constant
    assignment -- including flat tuple/list unpacking of constants
    (``x, y = 1, 2``: equal lengths, plain names, no starred targets)."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target, value = stmt.targets[0], stmt.value
        if isinstance(target, ast.Name) and isinstance(value, ast.Constant):
            return [(target.id, value)]
        if (
            isinstance(target, (ast.Tuple, ast.List))
            and isinstance(value, (ast.Tuple, ast.List))
            and len(target.elts) == len(value.elts)
            and all(isinstance(e, ast.Name) for e in target.elts)
            and all(isinstance(e, ast.Constant) for e in value.elts)
        ):
            return [(t.id, v) for t, v in zip(target.elts, value.elts)]
    elif (
        isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.target, ast.Name)
        and isinstance(stmt.value, ast.Constant)
    ):
        return [(stmt.target.id, stmt.value)]
    return []


def _stmt_kills(node: ast.AST) -> set[str]:
    """Names whose tracked value may change once *node* has (even partially)
    executed, seen from the enclosing scope: names bound at this scope level
    (assignment targets, ``del``, imports, ``def``/``class``, walrus --
    comprehension targets conservatively included) plus any name declared
    ``global``/``nonlocal`` inside a nested scope, since calling that
    closure rebinds the enclosing binding."""
    kills: set[str] = set()
    stack: list[ast.AST] = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, SCOPE_NODES):
            kills.update(binding_names(n))  # the def/class name itself
            for sub in ast.walk(n):
                if isinstance(sub, (ast.Global, ast.Nonlocal)):
                    kills.update(sub.names)
            # Decorators/defaults/annotations evaluate in this scope and may
            # contain walrus bindings.
            stack.extend(_boundary_children(n))
            continue
        kills.update(binding_names(n))
        stack.extend(ast.iter_child_nodes(n))
    return kills


def _declared_names(root: ast.AST, kinds) -> set[str]:
    return {
        name
        for n in ast.walk(root)
        if isinstance(n, kinds)
        for name in n.names
    }


def _sub_blocks(stmt: ast.stmt):
    """Statement lists nested directly inside a compound statement."""
    if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While, ast.If)):
        yield stmt.body
        yield stmt.orelse
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        yield stmt.body
    elif isinstance(stmt, ast.Try) or (
        hasattr(ast, "TryStar") and isinstance(stmt, ast.TryStar)
    ):
        yield stmt.body
        for handler in stmt.handlers:
            yield handler.body
        yield stmt.orelse
        yield stmt.finalbody
    elif isinstance(stmt, ast.Match):
        for case in stmt.cases:
            yield case.body


_COMPOUND = (
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.If,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    *((ast.TryStar,) if hasattr(ast, "TryStar") else ()),
    ast.Match,
)


class _SpanSubstituter(ast.NodeTransformer):
    """Replaces Name loads with constants (span propagation) or with the
    copied-from name (copy propagation); never enters nested scopes."""

    def __init__(self, consts: dict, copies: dict, owner) -> None:
        self.consts = consts
        self.copies = copies
        self.owner = owner

    def _skip(self, node: ast.AST) -> ast.AST:
        return node

    visit_FunctionDef = _skip
    visit_AsyncFunctionDef = _skip
    visit_Lambda = _skip
    visit_ClassDef = _skip

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load):
            const = self.consts.get(node.id)
            if const is not None:
                self.owner.changes += 1
                return ast.copy_location(ast.Constant(const.value), node)
            source = self.copies.get(node.id)
            if source is not None:
                self.owner.changes += 1
                return ast.copy_location(
                    ast.Name(id=source, ctx=ast.Load()), node
                )
        return node


class _NameSubstituter(ast.NodeTransformer):
    """Replaces Name loads with constants; never enters nested scopes."""

    def __init__(self, mapping: dict[str, ast.Constant], owner) -> None:
        self.mapping = mapping
        self.owner = owner

    def _skip(self, node: ast.AST) -> ast.AST:
        return node

    visit_FunctionDef = _skip
    visit_AsyncFunctionDef = _skip
    visit_Lambda = _skip
    visit_ClassDef = _skip

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load):
            const = self.mapping.get(node.id)
            if const is not None:
                self.owner.changes += 1
                return ast.copy_location(ast.Constant(const.value), node)
        return node


class ConstantPropagation(ScopedTransformer):
    name = "const-prop"

    def run(self, tree: ast.Module) -> ast.Module:
        if region_is_dynamic(tree):
            self.skipped_scopes += 1
            return tree
        self._module_dynamic = tree_has_dynamic(tree)
        return self.visit(tree)

    def visit_Module(self, node: ast.Module) -> ast.AST:
        if not self._module_dynamic:
            candidates = self._propagate(node, forbidden_decl=ast.Global)
            if candidates:
                self._cross_scope(node, candidates)
            # Nonlocal can never target a module global.
            self._span_propagate(
                node, _declared_names(node, ast.Global), allow_copies=False
            )
        return self.generic_visit(node)

    def _cross_scope(self, module: ast.Module, candidates: dict) -> None:
        """Propagate module constants into scopes whose enclosing top-level
        statement comes after the binding (see module docstring)."""
        for j, stmt in enumerate(module.body):
            avail = {
                name: const
                for name, (idx, const) in candidates.items()
                if idx < j
            }
            if not avail:
                continue
            for scope in _direct_nested_scopes(stmt):
                self._propagate_into(scope, avail)

    def _propagate_into(self, scope: ast.AST, avail: dict) -> None:
        if region_is_dynamic(scope):
            return
        effective = {
            name: const
            for name, const in avail.items()
            if name not in bound_names(scope)
        }
        if not effective:
            return
        sub = _NameSubstituter(effective, self)
        if isinstance(scope, ast.Lambda):
            scope.body = sub.visit(scope.body)
            roots = [scope.body]
        else:
            scope.body = [sub.visit(s) for s in scope.body]
            roots = scope.body
        for root in roots:
            for nested in _direct_nested_scopes(root):
                self._propagate_into(nested, effective)

    def _visit_function(self, node: ast.AST) -> ast.AST:
        if region_is_dynamic(node):
            self.skipped_scopes += 1
            return node
        self._propagate(node, forbidden_decl=ast.Nonlocal)
        self._span_propagate(
            node, _declared_names(node, (ast.Global, ast.Nonlocal))
        )
        return self.generic_visit(node)

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function
    # Lambda has no assignments; class-body names are attributes readable
    # from outside -- both inherit the base descend-only behaviour.

    def _propagate(self, scope: ast.AST, forbidden_decl: type) -> dict:
        counts: Counter[str] = Counter()
        for n in iter_region(scope):
            counts.update(binding_names(n))
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = scope.args
            for a in [*args.posonlyargs, *args.args, *args.kwonlyargs,
                      args.vararg, args.kwarg]:
                if a is not None:
                    counts[a.arg] += 1

        candidates: dict[str, tuple[int, ast.Constant]] = {}
        for idx, stmt in enumerate(scope.body):
            for name, value in _constant_pairs(stmt):
                if type(value.value) not in _PROPAGATE_TYPES:
                    continue
                if isinstance(value.value, (str, bytes)) and len(value.value) > _MAX_SEQ:
                    continue
                if counts[name] != 1:
                    continue
                candidates[name] = (idx, value)
        if not candidates:
            return {}

        declared: set[str] = set()
        for n in ast.walk(scope):
            if isinstance(n, forbidden_decl):
                declared.update(n.names)
        for name in declared:
            candidates.pop(name, None)
        if not candidates:
            return {}

        for j, stmt in enumerate(scope.body):
            active = {
                name: const
                for name, (idx, const) in candidates.items()
                if idx < j
            }
            if active:
                scope.body[j] = _NameSubstituter(active, self).visit(stmt)
        return candidates

    # -- span / copy propagation (multi-binding names) ----------------------
    def _span_propagate(
        self, scope: ast.AST, forbidden: set[str], allow_copies: bool = True
    ) -> None:
        """See module docstring.  *forbidden* holds the names that external
        code can rebind (``global``/``nonlocal`` declarations)."""
        self._forbidden = forbidden
        self._allow_copies = allow_copies
        self._scope_locals = {
            name
            for n in iter_region(scope)
            for name in binding_names(n)
        }
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            args = scope.args
            for a in [*args.posonlyargs, *args.args, *args.kwonlyargs,
                      args.vararg, args.kwarg]:
                if a is not None:
                    self._scope_locals.add(a.arg)
        self._walk_span_block(scope.body, {}, {})

    def _walk_span_block(
        self, stmts: list, consts: dict, copies: dict
    ) -> None:
        for i, stmt in enumerate(stmts):
            if isinstance(stmt, SCOPE_NODES):
                # def/class statement: the substituter would skip it whole
                # anyway; just account for what it can rebind.
                self._kill(_stmt_kills(stmt), consts, copies)
                continue
            kills = _stmt_kills(stmt)
            if isinstance(stmt, _COMPOUND):
                avail_c, avail_p = self._narrow(consts, copies, kills)
                if avail_c or avail_p:
                    stmts[i] = _SpanSubstituter(avail_c, avail_p, self).visit(
                        stmt
                    )
                for block in _sub_blocks(stmt):
                    self._walk_span_block(block, dict(avail_c), dict(avail_p))
                self._kill(kills, consts, copies)
                continue
            # Simple statement.  For single-name assignments the value
            # evaluates before the store, so the target's own kill does not
            # apply to the value expression (only bindings *inside* the
            # value -- walrus -- do); this keeps ``x = x + 1`` substitutable.
            target_name = None
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
            ):
                target_name = stmt.targets[0].id
                value_kills = _stmt_kills(stmt.value)
                avail_c, avail_p = self._narrow(consts, copies, value_kills)
                if avail_c or avail_p:
                    stmt.value = _SpanSubstituter(avail_c, avail_p, self).visit(
                        stmt.value
                    )
            elif isinstance(stmt, ast.AugAssign) and isinstance(
                stmt.target, ast.Name
            ):
                value_kills = _stmt_kills(stmt.value)
                avail_c, avail_p = self._narrow(consts, copies, value_kills)
                if avail_c or avail_p:
                    stmt.value = _SpanSubstituter(avail_c, avail_p, self).visit(
                        stmt.value
                    )
            elif (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.value is not None
            ):
                target_name = stmt.target.id
                value_kills = _stmt_kills(stmt.value)
                avail_c, avail_p = self._narrow(consts, copies, value_kills)
                if avail_c or avail_p:
                    stmt.value = _SpanSubstituter(avail_c, avail_p, self).visit(
                        stmt.value
                    )
            else:
                avail_c, avail_p = self._narrow(consts, copies, kills)
                if avail_c or avail_p:
                    stmts[i] = _SpanSubstituter(avail_c, avail_p, self).visit(
                        stmt
                    )
            self._kill(kills, consts, copies)
            # Gen: a fresh constant binding opens a new span ...
            for name, const in _constant_pairs(stmt):
                if name in self._forbidden:
                    continue
                if type(const.value) not in _PROPAGATE_TYPES:
                    continue
                if (
                    isinstance(const.value, (str, bytes))
                    and len(const.value) > _MAX_SEQ
                ):
                    continue
                consts[name] = const
                copies.pop(name, None)
            # ... and ``y = x`` between plain locals opens a copy span (if x
            # held a tracked constant the substitution above already turned
            # this into a constant assignment handled by _constant_pairs).
            if (
                self._allow_copies
                and target_name is not None
                and isinstance(stmt, ast.Assign)
                and isinstance(stmt.value, ast.Name)
                and isinstance(stmt.value.ctx, ast.Load)
                and stmt.value.id != target_name
                and target_name not in self._forbidden
                and stmt.value.id not in self._forbidden
                and stmt.value.id in self._scope_locals
            ):
                copies[target_name] = stmt.value.id

    def _narrow(self, consts: dict, copies: dict, kills: set[str]):
        avail_c = {k: v for k, v in consts.items() if k not in kills}
        avail_p = {
            k: v
            for k, v in copies.items()
            if k not in kills and v not in kills
        }
        return avail_c, avail_p

    @staticmethod
    def _kill(names: set[str], consts: dict, copies: dict) -> None:
        for n in names:
            consts.pop(n, None)
            copies.pop(n, None)
        for y in [y for y, src in copies.items() if src in names]:
            copies.pop(y, None)
