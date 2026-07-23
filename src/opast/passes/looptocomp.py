"""Append-loop to comprehension rewriting.

Rewrites the accumulation idiom

    x = []                      x = set()
    for v in it:                for v in it:
        if cond:                    x.add(f(v))
            x.append(f(v))

into ``x = [f(v) for v in it if cond]`` / ``x = {f(v) for v in it}``.  A
chain of nested ``for``/guard-``if`` statements with a single innermost
``append``/``add`` call maps to the equivalent multi-generator
comprehension.  The comprehension form runs the dedicated LIST_APPEND/
SET_ADD bytecode instead of an attribute lookup plus a method call per
iteration, and feeds the comp-to-map pass.

Safety conditions (any doubt -> no rewrite):

* the ``x = []`` (or ``x = set()``, requiring ``set`` unbound module-wide)
  seed assignment *immediately* precedes the loop, binds a plain name, and
  that name's **only** occurrence in the whole loop subtree is as the
  receiver of the single innermost ``append``/``add`` call -- so the loop
  can neither observe nor rebind ``x``, and the fresh-literal receiver
  guarantees the method is the real ``list.append``/``set.add``;
* the name is never declared ``global``/``nonlocal`` anywhere in the module
  and the module has no dynamic constructs: nothing that runs during the
  loop (calls in the element/condition/iter expressions included) can
  rebind ``x`` behind our back;
* the loop-target names become comprehension-local, so they must be dead
  afterwards: they may not occur anywhere else in the enclosing scope's
  subtree (nested scopes included -- a later closure read, a later loop
  reusing the name, a parameter, or a ``global``/``nonlocal`` declaration
  all disqualify);
* the loop subtree contains no nested scope (``def``/``lambda``/``class``),
  no ``yield``/``await`` and no walrus -- their interaction with the
  comprehension's own scope is not worth reasoning about;
* no ``else`` on any ``for`` and no rewriting inside ``try``/``with``: on a
  mid-iteration exception the statement form leaves a partially-filled
  ``x`` observable by the handler (or by ``__exit__``), the comprehension
  form does not.  Elsewhere the partial list is unobservable -- the
  exception leaves the frame and frame introspection is gated off with the
  rest of the dynamic constructs.

Zero iterations produce an empty list/set either way (the seed literal
never escapes, so object identity cannot be compared).  Class bodies are
left alone: a metaclass ``__prepare__`` mapping would observe the
per-iteration reads of ``x`` that the comprehension form removes.
"""

from __future__ import annotations

import ast

from ..analysis import all_bound_names
from ..safety import tree_has_dynamic
from .base import ScopedTransformer

#: Node types whose presence anywhere in the loop subtree disqualifies it.
_FORBIDDEN = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
    ast.Await,
    ast.Yield,
    ast.YieldFrom,
    ast.NamedExpr,
)

_METHODS = {"list": "append", "set": "add"}


def _flat_target_names(target: ast.expr) -> list[str] | None:
    """Names of a plain comprehension-compatible target (a name or a flat
    tuple/list of names), else None."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)) and all(
        isinstance(e, ast.Name) for e in target.elts
    ):
        return [e.id for e in target.elts]
    return None


class LoopToComprehension(ScopedTransformer):
    name = "loop-to-comp"

    def run(self, tree: ast.Module) -> ast.Module:
        # Whole-module gate, like inlining: exec/eval anywhere could rebind
        # the accumulator mid-loop, and frame introspection could observe
        # the loop variables this rewrite removes.
        if tree_has_dynamic(tree):
            self.skipped_scopes += 1
            return tree
        bound = all_bound_names(tree)
        self._set_ok = "set" not in bound
        self._declared: set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.Global, ast.Nonlocal)):
                self._declared.update(n.names)
        return self.visit(tree)

    def visit_Module(self, node: ast.Module) -> ast.AST:
        node.body = self._process_block(node.body, node, guarded=False)
        return self.generic_visit(node)

    def _visit_function(self, node: ast.AST) -> ast.AST:
        node.body = self._process_block(node.body, node, guarded=False)
        return self.generic_visit(node)

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function
    # ClassDef keeps the base behaviour: descend into methods, never rewrite
    # the class body itself (see module docstring).

    # -- block walk ---------------------------------------------------------
    def _process_block(self, stmts: list, scope: ast.AST, guarded: bool) -> list:
        out: list[ast.stmt] = []
        i = 0
        while i < len(stmts):
            stmt = stmts[i]
            nxt = stmts[i + 1] if i + 1 < len(stmts) else None
            if not guarded and nxt is not None:
                replacement = self._try_rewrite(stmt, nxt, scope)
                if replacement is not None:
                    out.append(replacement)
                    self.changes += 1
                    i += 2
                    continue
            self._descend(stmt, scope, guarded)
            out.append(stmt)
            i += 1
        return out

    def _descend(self, stmt: ast.stmt, scope: ast.AST, guarded: bool) -> None:
        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            stmt.body = self._process_block(stmt.body, scope, guarded)
            stmt.orelse = self._process_block(stmt.orelse, scope, guarded)
        elif isinstance(stmt, ast.If):
            stmt.body = self._process_block(stmt.body, scope, guarded)
            stmt.orelse = self._process_block(stmt.orelse, scope, guarded)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            stmt.body = self._process_block(stmt.body, scope, True)
        elif isinstance(stmt, ast.Try) or (
            hasattr(ast, "TryStar") and isinstance(stmt, ast.TryStar)
        ):
            stmt.body = self._process_block(stmt.body, scope, True)
            for handler in stmt.handlers:
                handler.body = self._process_block(handler.body, scope, True)
            stmt.orelse = self._process_block(stmt.orelse, scope, True)
            stmt.finalbody = self._process_block(stmt.finalbody, scope, True)
        elif isinstance(stmt, ast.Match):
            for case in stmt.cases:
                case.body = self._process_block(case.body, scope, guarded)
        # Nested scope statements are handled by the visitor dispatch.

    # -- the rewrite --------------------------------------------------------
    def _try_rewrite(self, seed: ast.stmt, loop: ast.stmt, scope: ast.AST):
        if not (
            isinstance(seed, ast.Assign)
            and len(seed.targets) == 1
            and isinstance(seed.targets[0], ast.Name)
        ):
            return None
        acc = seed.targets[0].id
        if acc in self._declared:
            return None
        value = seed.value
        if isinstance(value, ast.List) and not value.elts:
            kind = "list"
        elif (
            self._set_ok
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "set"
            and not value.args
            and not value.keywords
        ):
            kind = "set"
        else:
            return None
        if not isinstance(loop, ast.For):
            return None

        matched = self._match_chain(loop, acc, _METHODS[kind])
        if matched is None:
            return None
        generators, elt = matched

        for n in ast.walk(loop):
            if isinstance(n, _FORBIDDEN):
                return None
        # The accumulator's only occurrence in the loop must be the matched
        # call receiver -- one Name node total.
        if sum(
            isinstance(n, ast.Name) and n.id == acc for n in ast.walk(loop)
        ) != 1:
            return None

        targets: set[str] = set()
        for gen in generators:
            names = _flat_target_names(gen.target)
            if names is None:
                return None
            targets.update(names)
        if targets & self._declared or not self._targets_dead(scope, loop, targets):
            return None

        comp_cls = ast.ListComp if kind == "list" else ast.SetComp
        comp = ast.copy_location(
            comp_cls(elt=elt, generators=generators), loop
        )
        new = ast.Assign(targets=[seed.targets[0]], value=comp)
        return ast.copy_location(new, seed)

    def _match_chain(self, loop: ast.For, acc: str, method: str):
        """Match a for/if chain ending in a single ``acc.<method>(elt)``
        call; returns ``(generators, elt)`` or None."""
        generators: list[ast.comprehension] = []
        node: ast.stmt = loop
        while True:
            if not (isinstance(node, ast.For) and not node.orelse):
                return None
            gen = ast.comprehension(
                target=node.target, iter=node.iter, ifs=[], is_async=0
            )
            generators.append(gen)
            body = node.body
            while (
                len(body) == 1
                and isinstance(body[0], ast.If)
                and not body[0].orelse
            ):
                gen.ifs.append(body[0].test)
                body = body[0].body
            if len(body) != 1:
                return None
            inner = body[0]
            if isinstance(inner, ast.For):
                node = inner
                continue
            call = inner.value if isinstance(inner, ast.Expr) else None
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == method
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == acc
                and len(call.args) == 1
                and not isinstance(call.args[0], ast.Starred)
                and not call.keywords
            ):
                return None
            return generators, call.args[0]

    @staticmethod
    def _targets_dead(scope: ast.AST, loop: ast.For, targets: set[str]) -> bool:
        """True if none of *targets* occurs anywhere in *scope*'s subtree
        outside the loop (nested scopes included)."""
        stack = list(ast.iter_child_nodes(scope))
        while stack:
            n = stack.pop()
            if n is loop:
                continue
            if isinstance(n, ast.Name) and n.id in targets:
                return False
            if isinstance(n, ast.arg) and n.arg in targets:
                return False
            if isinstance(
                n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ) and n.name in targets:
                return False
            stack.extend(ast.iter_child_nodes(n))
        return True
