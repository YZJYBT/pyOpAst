"""Tail-recursion elimination.

A self-call that *is* the whole return value can reuse the current frame:
rebind the parameters and jump back to the top instead of recursing.

    def total(n, acc):          def total(n, acc):
        if n == 0:                  while True:
            return acc      ->          if n == 0:
        return total(               ->      return acc
            n - 1, acc + n)             n, acc = n - 1, acc + n
                                        continue

Measured on CPython 3.14 (900-deep accumulator): the loop form runs
**4.6x** faster than the recursion.  Python does not do this itself
because the recursion limit is observable -- eliminating the frames makes
``RecursionError`` disappear -- so the rewrite **keeps a depth counter**
and raises the same error at the same approximate point, retaining 2.6x
of the win.  The limit is read once per call, not per iteration.

That counter is only an *approximation*, which is why the whole pass is
opt-in behind the aggressive ``tail-calls`` option: the real trigger point
depends on how deep the *caller* already is (finding that out needs frame
introspection, exactly what the dynamic gate forbids), and the limit comes
from ``sys.getrecursionlimit()`` rather than the interpreter's own stack,
so the two diverge if that function is monkeypatched.  The refining
``unbounded-recursion`` option drops the counter entirely (back to the
full 4.6x) for callers who want deep tail recursion to simply work.

**Partial elimination** is supported and is where divide-and-conquer code
benefits: only the calls in tail position become jumps, any others stay
real recursion, which is the textbook way to halve quicksort's stack::

    def qsort(lo, hi):
        if lo >= hi:
            return
        p = partition(lo, hi)
        qsort(lo, p - 1)        # not a tail call: left as recursion
        return qsort(p + 1, hi) # tail call: becomes a jump

Conditions (any doubt -> untouched):

* module-level ``def``, bound exactly once, no decorators, never declared
  ``global``, and the name is not shadowed by a local;
* plain positional parameters only, and every self-call passes exactly one
  argument per parameter (no ``*``/``**``, no relying on defaults, whose
  once-only evaluation a loop would not reproduce);
* the tail call is not inside ``try``/``with`` -- ``finally`` and
  ``__exit__`` run per frame, and a loop would run them per iteration --
  and not inside a nested loop, where the emitted ``continue`` would bind
  to the wrong loop;
* no ``yield``/``await`` (generators and coroutines are not plain calls),
  and no nested function or lambda that reads a parameter, since it would
  capture the mutated variable rather than the value of its own frame.
"""

from __future__ import annotations

import ast

from ..analysis import all_bound_names, binding_names, builtin_gate
from ..safety import SCOPE_NODES, iter_region, region_is_dynamic
from .base import ScopedTransformer

_DEPTH = "_opast_depth"
_LIMIT = "_opast_limit"
_SYS = "_opast_sys"

_LOOPS = (ast.For, ast.AsyncFor, ast.While)
_GUARDS = (ast.Try, ast.With, ast.AsyncWith)
if hasattr(ast, "TryStar"):  # 3.11+
    _GUARDS = _GUARDS + (ast.TryStar,)


def _self_call(node: ast.AST, name: str):
    """``node`` is exactly ``name(a, b, ...)`` with plain positional args."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
        and not node.keywords
        and not any(isinstance(a, ast.Starred) for a in node.args)
    )


class TailRecursion(ScopedTransformer):
    name = "tail-recursion"

    def run(self, tree: ast.Module) -> ast.Module:
        # Opt-in: the depth counter only *approximates* the recursion limit
        # (see the module docstring), so this is a stated assumption rather
        # than a proof.
        if "tail-calls" not in self.aggressive:
            return tree
        if region_is_dynamic(tree):
            self.skipped_scopes += 1
            return tree
        self._counted = "unbounded-recursion" not in self.aggressive
        if self._counted and not builtin_gate(tree, "RecursionError"):
            # A module that binds the name would make the synthesized raise
            # throw something else than a real stack overflow does.
            self.skipped_scopes += 1
            return tree
        counts: dict[str, int] = {}
        for node in iter_region(tree):
            for bound in binding_names(node):
                counts[bound] = counts.get(bound, 0) + 1
        declared: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                declared.update(node.names)

        taken = all_bound_names(tree) | {
            n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
        }
        self._depth = _fresh(_DEPTH, taken)
        self._limit = _fresh(_LIMIT, taken)
        self._sys = _fresh(_SYS, taken)

        needs_sys = False
        for stmt in tree.body:
            if not isinstance(stmt, ast.FunctionDef) or stmt.decorator_list:
                continue
            if counts.get(stmt.name, 0) != 1 or stmt.name in declared:
                continue
            if self._rewrite(stmt):
                needs_sys = needs_sys or self._counted
        if needs_sys:
            tree.body.insert(
                _import_position(tree),
                ast.Import(names=[ast.alias(name="sys", asname=self._sys)]),
            )
            # Later passes in this very iteration read lineno; the pipeline
            # only fixes locations once the whole round is done.
            ast.fix_missing_locations(tree)
        return tree

    # -- one function -------------------------------------------------------
    def _rewrite(self, func: ast.FunctionDef) -> bool:
        args = func.args
        if (
            args.vararg
            or args.kwarg
            or args.kwonlyargs
            or args.posonlyargs
        ):
            return False
        params = [a.arg for a in args.args]
        if not params:
            return False  # a no-argument tail call is an infinite loop
        name = func.name

        for node in ast.walk(func):
            if isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await)):
                return False
            # A local rebinding of the function's own name would make the
            # "self-call" call something else.
            if (
                isinstance(node, ast.Name)
                and node.id == name
                and not isinstance(node.ctx, ast.Load)
            ):
                return False
            # A closure over a parameter would see the loop's mutations.
            if isinstance(node, SCOPE_NODES) and node is not func:
                inner = {
                    n.id for n in ast.walk(node) if isinstance(n, ast.Name)
                }
                if inner & set(params):
                    return False
            if isinstance(node, ast.arg) and node.arg == name:
                return False

        # The docstring must stay the first statement of the *function*:
        # moving it inside the loop would blank out ``__doc__``.
        body = list(func.body)
        prologue = []
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            prologue.append(body.pop(0))

        sites = self._tail_sites(body, name, len(params))
        if sites is None or not sites:
            return False
        # A reused frame keeps the previous iteration's locals alive, so a
        # read that the original would have met with UnboundLocalError could
        # silently see a stale value.  Require every non-parameter local to
        # be definitely assigned earlier in the same iteration.
        if not _locals_safe(body, params):
            return False

        func.body = prologue + [
            ast.While(
                test=ast.Constant(True),
                body=self._transform(body, sites, params)
                + [ast.Return(value=None)],
                orelse=[],
            )
        ]
        if self._counted:
            func.body[len(prologue):len(prologue)] = [
                ast.Assign(
                    targets=[ast.Name(id=self._depth, ctx=ast.Store())],
                    value=ast.Constant(0),
                ),
                ast.Assign(
                    targets=[ast.Name(id=self._limit, ctx=ast.Store())],
                    value=ast.Call(
                        func=ast.Attribute(
                            value=ast.Name(id=self._sys, ctx=ast.Load()),
                            attr="getrecursionlimit",
                            ctx=ast.Load(),
                        ),
                        args=[],
                        keywords=[],
                    ),
                ),
            ]
        ast.fix_missing_locations(func)
        self.changes += 1
        return True

    def _tail_sites(self, body, name: str, arity: int):
        """Return statements whose value is a self-call, reachable by a
        ``continue`` from the function's top-level loop.  None when a site
        is unusable (wrong arity), which aborts the whole rewrite."""
        found = []

        def walk(stmts, jumpable: bool):
            for stmt in stmts:
                if isinstance(stmt, ast.Return) and _self_call(stmt.value, name):
                    if not jumpable or len(stmt.value.args) != arity:
                        raise _Unusable
                    found.append(stmt)
                    continue
                if isinstance(stmt, ast.If):
                    walk(stmt.body, jumpable)
                    walk(stmt.orelse, jumpable)
                elif isinstance(stmt, _LOOPS):
                    # ``continue`` inside would bind to this loop instead.
                    walk(stmt.body, False)
                    walk(stmt.orelse, False)
                elif isinstance(stmt, _GUARDS):
                    # finally/__exit__ run once per frame, not per iteration.
                    for sub in _guard_blocks(stmt):
                        walk(sub, False)
                elif isinstance(stmt, ast.Match):
                    for case in stmt.cases:
                        walk(case.body, jumpable)

        try:
            walk(body, True)
        except _Unusable:
            return None
        return found

    def _transform(self, body, sites, params):
        replace = {id(stmt) for stmt in sites}

        def convert(stmts):
            out = []
            for stmt in stmts:
                if id(stmt) in replace:
                    out.extend(self._jump(stmt.value.args, params))
                    continue
                if isinstance(stmt, ast.If):
                    stmt.body = convert(stmt.body)
                    stmt.orelse = convert(stmt.orelse)
                elif isinstance(stmt, ast.Match):
                    for case in stmt.cases:
                        case.body = convert(case.body)
                out.append(stmt)
            return out

        return convert(body)

    def _jump(self, call_args, params):
        """Rebind the parameters simultaneously, then loop."""
        stmts = []
        if self._counted:
            stmts.append(
                ast.AugAssign(
                    target=ast.Name(id=self._depth, ctx=ast.Store()),
                    op=ast.Add(),
                    value=ast.Constant(1),
                )
            )
            stmts.append(
                ast.If(
                    test=ast.Compare(
                        left=ast.Name(id=self._depth, ctx=ast.Load()),
                        ops=[ast.GtE()],
                        comparators=[ast.Name(id=self._limit, ctx=ast.Load())],
                    ),
                    body=[
                        ast.Raise(
                            exc=ast.Call(
                                func=ast.Name(id="RecursionError", ctx=ast.Load()),
                                args=[
                                    ast.Constant(
                                        "maximum recursion depth exceeded"
                                    )
                                ],
                                keywords=[],
                            ),
                            cause=None,
                        )
                    ],
                    orelse=[],
                )
            )
        if len(params) == 1:
            target = ast.Name(id=params[0], ctx=ast.Store())
            value = call_args[0]
        else:
            target = ast.Tuple(
                elts=[ast.Name(id=p, ctx=ast.Store()) for p in params],
                ctx=ast.Store(),
            )
            value = ast.Tuple(elts=list(call_args), ctx=ast.Load())
        stmts.append(ast.Assign(targets=[target], value=value))
        stmts.append(ast.Continue())
        return stmts


class _Unusable(Exception):
    """A self-call in a position the rewrite cannot express."""


def _locals_safe(body, params) -> bool:
    """True when every non-parameter local is definitely assigned before it
    is read, within a single iteration.

    Without this the loop's reused frame changes behaviour: the original
    gets a fresh frame per call, so reading a local the current call has not
    assigned is an ``UnboundLocalError``; the loop would hand back whatever
    the previous iteration left there.  Compound statements contribute their
    reads but never count as assignments (they may not run), which is the
    conservative direction.
    """
    # Every binding form counts, not just ``Name`` stores: ``import x``,
    # ``def x``, ``class x``, ``with ... as x`` and ``except ... as x`` all
    # create per-frame locals that a reused frame would carry over.
    region = ast.Module(body=list(body), type_ignores=[])
    tracked: set[str] = set()
    for node in iter_region(region):
        tracked.update(binding_names(node))
    tracked -= set(params)
    if not tracked:
        return True

    def reads_ok(node, defined) -> bool:
        return not any(
            n.id in tracked and n.id not in defined
            for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        )

    def check(stmts, defined):
        defined = set(defined)
        for stmt in stmts:
            if isinstance(stmt, ast.If):
                if not reads_ok(stmt.test, defined):
                    return None
                left = check(stmt.body, defined)
                right = check(stmt.orelse, defined)
                if left is None or right is None:
                    return None
                defined = left & right
                continue
            if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                if isinstance(stmt, ast.AugAssign):
                    # ``x += e`` reads x first.
                    if not reads_ok(stmt.target, defined):
                        return None
                if stmt.value is not None and not reads_ok(stmt.value, defined):
                    return None
                targets = (
                    stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                )
                for target in targets:
                    for n in ast.walk(target):
                        if isinstance(n, ast.Name):
                            if isinstance(n.ctx, ast.Load):
                                if not reads_ok(n, defined):
                                    return None
                            else:
                                defined.add(n.id)
                continue
            if isinstance(
                stmt,
                (ast.Import, ast.ImportFrom, ast.FunctionDef,
                 ast.AsyncFunctionDef, ast.ClassDef),
            ):
                # Straight-line binders: they always run at this level, so
                # the names they create are definitely assigned afterwards.
                if not reads_ok(stmt, defined):
                    return None
                defined.update(binding_names(stmt))
                continue
            if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
                head = stmt.iter if isinstance(stmt, (ast.For, ast.AsyncFor)) else stmt.test
                if not reads_ok(head, defined):
                    return None
                inner = set(defined)
                if isinstance(stmt, (ast.For, ast.AsyncFor)):
                    for n in ast.walk(stmt.target):
                        if isinstance(n, ast.Name):
                            inner.add(n.id)
                if check(stmt.body, inner) is None:
                    return None
                if check(stmt.orelse, defined) is None:
                    return None
                continue  # the loop may not run: nothing becomes definite
            if isinstance(stmt, (ast.Try, *(
                (ast.TryStar,) if hasattr(ast, "TryStar") else ()
            ))):
                for block in _guard_blocks(stmt):
                    if check(block, defined) is None:
                        return None
                continue
            if not reads_ok(stmt, defined):
                return None
        return defined

    return check(body, set(params)) is not None


def _guard_blocks(stmt):
    yield stmt.body
    for handler in getattr(stmt, "handlers", []):
        yield handler.body
    yield getattr(stmt, "orelse", [])
    yield getattr(stmt, "finalbody", [])


def _fresh(base: str, taken: set[str]) -> str:
    name = base
    while name in taken:
        name += "_"
    taken.add(name)
    return name


def _import_position(tree: ast.Module) -> int:
    index = 0
    for i, stmt in enumerate(tree.body):
        if (
            i == 0
            and isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            index = i + 1
            continue
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
            index = i + 1
            continue
        break
    return index
