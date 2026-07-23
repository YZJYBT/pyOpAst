"""Index loops over fresh sequences become direct iteration (range-to-iter).

Rewrites

    for i in range(len(x)):          for _pyopt_elem_N in x:
        use(x[i])               ->       use(_pyopt_elem_N)

when the index is used *only* to subscript ``x``, and otherwise

    for i in range(len(x)):          for i, _pyopt_elem_N in enumerate(x):
        use(i, x[i])            ->       use(i, _pyopt_elem_N)

which keeps ``i`` bound exactly as before (same per-iteration values, same
final value, still unbound after zero iterations).  Both forms drop the
per-iteration ``BINARY_SUBSCR`` index lookups.

Safety conditions (any doubt -> no rewrite):

* ``range``/``len`` (and ``enumerate`` for the second form) are trusted:
  no dynamic constructs module-wide and the names never bound anywhere --
  the fresh-container gate of the ``len()`` machinery plus the ``range``
  builtin gate;
* ``x`` qualifies under :func:`pyopt.analysis.fresh_container_names`
  (bound exactly once, never escaping, never mutated, never rebound), so
  its length and elements cannot change during the loop -- which is what
  makes ``x[i]``-at-use-time equal to element-at-iteration-time -- **and**
  its binding is provably a *sequence* (list/tuple display, list
  comprehension, or a ``list``/``tuple``/``sorted``/``range``/``str``/
  ``bytes`` constructor call), so iteration order matches ``x[0..n-1]``
  (sets/dicts are excluded: their iteration order is not indexable);
* the loop target is a plain name, bound nowhere inside the loop subtree
  (nested ``global``/``nonlocal`` declarations included);
* only exact ``x[i]`` *loads* at this scope level inside the loop body are
  replaced (at least one, or the rewrite is pointless); ``x[i]`` inside
  nested scopes, in the ``else`` block or in compound index expressions
  (``x[i+1]``) stays as written -- under the enumerate form both ``x`` and
  ``i`` remain bound, so leftovers are still correct;
* the direct form (which stops binding ``i``) additionally requires that
  ``i`` occurs nowhere else: not outside the loop anywhere in the
  enclosing scope's subtree (nested scopes included), and inside the loop
  only as the target and the replaced indices.

Function and module scopes (module-global ``x`` rebinding by called code
would need a ``global`` declaration or dynamic constructs, both gated);
class bodies are left alone as usual.  Runs before loop-to-comp, whose
chain matcher accepts both rewritten forms, so an index-append loop can
cascade all the way to a comprehension in the same iteration.
"""

from __future__ import annotations

import ast

from ..analysis import builtin_gate, fresh_container_names
from ..safety import iter_region, tree_has_dynamic
from .base import ScopedTransformer
from .licm import container_gate, subtree_bindings

_TEMP_PREFIX = "_pyopt_elem"

#: Binding value shapes proving *sequence* semantics (iteration order is
#: exactly ``x[0..len(x)-1]``).
_SEQUENCE_DISPLAYS = (ast.List, ast.Tuple, ast.ListComp)
_SEQUENCE_CALLS = frozenset({"list", "tuple", "sorted", "range", "str", "bytes"})


def _name_occurs(root: ast.AST, name: str, skip: ast.AST | None = None) -> int:
    """Occurrences of *name* in *root*'s subtree (Name nodes, parameters,
    def/class names, global/nonlocal declarations), skipping the *skip*
    subtree."""
    count = 0
    stack = [root] if root is not skip else []
    while stack:
        n = stack.pop()
        if isinstance(n, ast.Name) and n.id == name:
            count += 1
        elif isinstance(n, ast.arg) and n.arg == name:
            count += 1
        elif isinstance(
            n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ) and n.name == name:
            count += 1
        elif isinstance(n, (ast.Global, ast.Nonlocal)) and name in n.names:
            count += 1
        stack.extend(c for c in ast.iter_child_nodes(n) if c is not skip)
    return count


class _IndexSubstituter(ast.NodeTransformer):
    """Replaces exact ``x[i]`` loads with the element temp (or, with
    ``count_only``, just counts them); never enters nested scopes."""

    def __init__(self, container: str, index: str, temp: str,
                 count_only: bool = False) -> None:
        self.container = container
        self.index = index
        self.temp = temp
        self.count = 0
        self.count_only = count_only

    def _skip(self, node: ast.AST) -> ast.AST:
        return node

    visit_FunctionDef = _skip
    visit_AsyncFunctionDef = _skip
    visit_Lambda = _skip
    visit_ClassDef = _skip

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if (
            isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id == self.container
            and isinstance(node.value.ctx, ast.Load)
            and isinstance(node.slice, ast.Name)
            and node.slice.id == self.index
            and isinstance(node.slice.ctx, ast.Load)
        ):
            self.count += 1
            if not self.count_only:
                return ast.copy_location(
                    ast.Name(id=self.temp, ctx=ast.Load()), node
                )
            return node
        return self.generic_visit(node)


class RangeToIteration(ScopedTransformer):
    name = "range-to-iter"

    def run(self, tree: ast.Module) -> ast.Module:
        if (
            tree_has_dynamic(tree)
            or not container_gate(tree)
            or not builtin_gate(tree, "range")
        ):
            self.skipped_scopes += 1
            return tree
        self._enum_ok = builtin_gate(tree, "enumerate")
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

    def _visit_scope_body(self, node: ast.AST) -> ast.AST:
        sequences = self._sequence_names(node)
        if sequences:
            # Region-only walk: loops inside nested scopes are handled when
            # their own scope is visited.
            loops = [n for n in iter_region(node) if isinstance(n, ast.For)]
            for loop in loops:
                self._try_rewrite(node, loop, sequences)
        return self.generic_visit(node)

    visit_Module = _visit_scope_body
    visit_FunctionDef = _visit_scope_body
    visit_AsyncFunctionDef = _visit_scope_body
    # ClassDef keeps the base behaviour: descend into methods only.

    @staticmethod
    def _sequence_names(scope: ast.AST) -> frozenset[str]:
        fresh = fresh_container_names(scope)
        if not fresh:
            return frozenset()
        sequences = set()
        for n in iter_region(scope):
            if not (
                isinstance(n, ast.Assign)
                and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id in fresh
            ):
                continue
            value = n.value
            if isinstance(value, _SEQUENCE_DISPLAYS) or (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id in _SEQUENCE_CALLS
            ):
                sequences.add(n.targets[0].id)
        return frozenset(sequences)

    def _try_rewrite(
        self, scope: ast.AST, loop: ast.For, sequences: frozenset[str]
    ) -> None:
        if not isinstance(loop.target, ast.Name):
            return
        index = loop.target.id
        it = loop.iter
        if not (
            isinstance(it, ast.Call)
            and isinstance(it.func, ast.Name)
            and it.func.id == "range"
            and len(it.args) == 1
            and not it.keywords
            and isinstance(it.args[0], ast.Call)
            and isinstance(it.args[0].func, ast.Name)
            and it.args[0].func.id == "len"
            and len(it.args[0].args) == 1
            and not it.args[0].keywords
            and isinstance(it.args[0].args[0], ast.Name)
        ):
            return
        container = it.args[0].args[0].id
        if container not in sequences or container == index:
            return
        inner_bound: set[str] = set()
        for stmt in [*loop.body, *loop.orelse]:
            inner_bound |= subtree_bindings(stmt)
        if index in inner_bound:
            return

        counter = _IndexSubstituter(container, index, "", count_only=True)
        for stmt in loop.body:
            counter.visit(stmt)
        if counter.count == 0:
            return

        # Decide the form before touching anything.  Inside the loop the
        # index legitimately occurs as the target (1) plus once per exact
        # ``x[i]`` (each contributing one Name for ``i``); anything beyond
        # that -- or any use outside the loop -- keeps ``i`` alive.
        index_dead = (
            _name_occurs(scope, index, skip=loop) == 0
            and _name_occurs(loop, index) == 1 + counter.count
        )
        if not index_dead and not self._enum_ok:
            return

        temp = self._fresh_name()
        sub = _IndexSubstituter(container, index, temp)
        loop.body = [sub.visit(stmt) for stmt in loop.body]
        if index_dead:
            loop.target = ast.copy_location(
                ast.Name(id=temp, ctx=ast.Store()), loop.target
            )
            loop.iter = ast.copy_location(
                ast.Name(id=container, ctx=ast.Load()), loop.iter
            )
        else:
            loop.target = ast.copy_location(
                ast.Tuple(
                    elts=[loop.target, ast.Name(id=temp, ctx=ast.Store())],
                    ctx=ast.Store(),
                ),
                loop.target,
            )
            loop.iter = ast.copy_location(
                ast.Call(
                    func=ast.Name(id="enumerate", ctx=ast.Load()),
                    args=[ast.Name(id=container, ctx=ast.Load())],
                    keywords=[],
                ),
                loop.iter,
            )
        self.changes += 1
