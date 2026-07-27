"""Condition fact narrowing (path-sensitive interval folding).

Decides comparisons between provably plain-``int`` expressions whose value
ranges make the outcome certain, and replaces them with ``True``/``False``
so the dead-code pass can delete the branch:

    for i in range(n):        for i in range(n):
        if i < 0:        ->       if False:      -> block deleted
            ...                       ...

Two sources of interval knowledge are combined:

* the **flow-insensitive base** from :func:`opast.analysis.infer_int_ranges`
  (join over every binding, with widening) -- this is what proves a
  ``range()`` target non-negative;
* **path-sensitive narrowing**: inside ``if c:`` the condition holds, so a
  name compared against a bound there carries a tighter interval for the
  duration of that block (and the negated fact holds in ``else``).  This is
  what decides the classic re-test ``if k > 10: ... if k > 5:``.

Soundness rules for the narrowed environment:

* a name is narrowed only while nothing rebinds it.  After any statement,
  every name bound anywhere in that statement's subtree is **reset to its
  base interval** -- which over-approximates every value the name can ever
  hold, so resetting is always sound;
* entering a loop body resets every name bound in the loop's subtree first
  (the body also runs after earlier iterations mutated them), and only then
  applies the loop test as a fact -- the test does hold at every body entry;
* ``and`` narrows in the true branch (all operands hold), ``or`` narrows in
  the false branch (all operands fail), ``not`` swaps the two;
* only *pure* int expressions are ever folded (names, int constants,
  arithmetic and ``len()`` of a fresh container -- :func:`is_int_expr`), so
  removing an evaluation can never remove a side effect;
* pure is not *total*: ``10 // x`` raises when ``x`` can be 0.  Any operand
  a rewrite would stop evaluating must additionally be
  :func:`hoistable_int_expr` (provably never raises); operands the original
  short-circuit already skipped are exempt.

Function scopes only, like the other interval consumers: module globals may
be rebound by any called function, so nothing is provably ranged there.
Bare truth tests (``if count:``) participate as well: a name whose interval
excludes ``0`` is always truthy, one pinned to ``[0, 0]`` always falsy.
"""

from __future__ import annotations

import ast

from ..analysis import (
    INT_BIN_OPS,
    NEG_INF,
    POS_INF,
    AnnotationTrust,
    _binop_range,
    builtin_gate,
    fresh_container_names,
    hoistable_int_expr,
    infer_int_names,
    infer_int_ranges,
    int_expr_range,
    is_int_expr,
)
from ..safety import region_is_dynamic
from .licm import container_gate, subtree_bindings

_LOOPS = (ast.While, ast.For, ast.AsyncFor)

#: ``op`` -> (op that must hold for True, op that must hold for False).
_NEGATE = {
    ast.Lt: ast.GtE,
    ast.LtE: ast.Gt,
    ast.Gt: ast.LtE,
    ast.GtE: ast.Lt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}


def _decide(op, left, right):
    """True/False when the intervals settle ``left <op> right``, else None.
    Intervals are inclusive ``(lo, hi)``; infinities compare correctly."""
    if left is None or right is None:
        return None
    l0, l1 = left
    r0, r1 = right
    if isinstance(op, ast.Lt):
        if l1 < r0:
            return True
        if l0 >= r1:
            return False
    elif isinstance(op, ast.LtE):
        if l1 <= r0:
            return True
        if l0 > r1:
            return False
    elif isinstance(op, ast.Gt):
        if l0 > r1:
            return True
        if l1 <= r0:
            return False
    elif isinstance(op, ast.GtE):
        if l0 >= r1:
            return True
        if l1 < r0:
            return False
    elif isinstance(op, ast.Eq):
        if l0 == l1 == r0 == r1:
            return True
        if l1 < r0 or r1 < l0:
            return False
    elif isinstance(op, ast.NotEq):
        if l0 == l1 == r0 == r1:
            return False
        if l1 < r0 or r1 < l0:
            return True
    return None


def _narrow_one(current, op, bound):
    """The interval of a name known to satisfy ``name <op> bound`` (bound
    given as an interval), or None when nothing tightens."""
    if current is None or bound is None:
        return current
    lo, hi = current
    blo, bhi = bound
    if isinstance(op, ast.Lt):
        hi = min(hi, bhi - 1 if bhi != POS_INF else POS_INF)
    elif isinstance(op, ast.LtE):
        hi = min(hi, bhi)
    elif isinstance(op, ast.Gt):
        lo = max(lo, blo + 1 if blo != NEG_INF else NEG_INF)
    elif isinstance(op, ast.GtE):
        lo = max(lo, blo)
    elif isinstance(op, ast.Eq):
        if blo == bhi:
            lo, hi = max(lo, blo), min(hi, bhi)
    else:
        return current
    if lo > hi:
        # Contradiction: the branch is unreachable.  Keep the original
        # interval rather than inventing an empty one -- dead-code removal
        # of unreachable blocks is not this pass's job.
        return current
    return (lo, hi)


class _Folder(ast.NodeTransformer):
    """Folds decidable comparisons under the current environment; never
    descends into nested scopes."""

    def __init__(self, env: dict, ints, containers, owner) -> None:
        self.env = env
        self.ints = ints
        self.containers = containers
        self.owner = owner

    def _skip(self, node: ast.AST) -> ast.AST:
        return node

    visit_FunctionDef = _skip
    visit_AsyncFunctionDef = _skip
    visit_Lambda = _skip
    visit_ClassDef = _skip

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        node = self.generic_visit(node)
        operands = [node.left, *node.comparators]
        if not all(
            is_int_expr(o, self.ints, self.containers) for o in operands
        ):
            return node
        # A chain is the conjunction of its legs, with each middle operand
        # evaluated once.  Operands are pure (``is_int_expr``) but not
        # necessarily *total* -- ``10 // x`` raises when ``x`` can be 0 --
        # so an operand may only be DROPPED when it is also hoistable
        # (provably never raises).  Within that limit the short-circuit is
        # unobservable: one provably-false leg makes the whole chain False
        # (operands past that leg never ran in the original either), all
        # legs True makes it True.  Otherwise provably-true legs are peeled
        # from the FRONT and BACK only, so the remainder stays one
        # contiguous chain (``0 <= i < n`` becomes ``i < n``); an undecided
        # leg surrounded by decided ones keeps the original chain rather
        # than duplicating shared operands.
        ranges = [
            int_expr_range(o, self.env, self.containers) for o in operands
        ]
        verdicts = [
            _decide(op, ranges[k], ranges[k + 1])
            for k, op in enumerate(node.ops)
        ]
        droppable = [
            hoistable_int_expr(o, self.ints, self.containers)
            for o in operands
        ]
        for k, v in enumerate(verdicts):
            if v is False:
                # The original evaluates operands 0..k+1 at most (the chain
                # stops at leg *k*); replacing with False deletes those
                # evaluations, so every one of them must be total.
                if all(droppable[: k + 2]):
                    self.owner.changes += 1
                    return ast.copy_location(ast.Constant(False), node)
                return node
        if all(v is True for v in verdicts) and all(droppable):
            self.owner.changes += 1
            return ast.copy_location(ast.Constant(True), node)
        # Peeling a front leg drops its left operand; peeling a back leg
        # drops its right operand.  Either drop needs totality.  At least
        # one leg always remains: reaching here means some operand is not
        # droppable (the all-True/all-droppable fold did not fire), and a
        # fully peeled chain would delete that operand's evaluation.
        lo = 0
        hi = len(verdicts)
        while lo < hi - 1 and verdicts[lo] is True and droppable[lo]:
            lo += 1
        while hi - lo > 1 and verdicts[hi - 1] is True and droppable[hi]:
            hi -= 1
        if lo == 0 and hi == len(verdicts):
            return node
        self.owner.changes += 1
        trimmed = ast.Compare(
            left=operands[lo],
            ops=node.ops[lo:hi],
            comparators=operands[lo + 1 : hi + 1],
        )
        return ast.copy_location(trimmed, node)


class ConditionNarrowing:
    """Pass driver: walks blocks maintaining narrowed intervals."""

    name = "cond-narrow"

    #: Set by the pipeline (this pass is not a ScopedTransformer).
    aggressive: frozenset = frozenset()

    def __init__(self) -> None:
        self.changes = 0
        self.skipped_scopes = 0

    def run(self, tree: ast.Module) -> ast.Module:
        if region_is_dynamic(tree):
            self.skipped_scopes += 1
            return tree
        self._len_ok = container_gate(tree)
        self._range_ok = builtin_gate(tree, "range")
        self._trust = AnnotationTrust(
            tree if "annotations" in self.aggressive else None
        )
        self._visit(tree)
        return tree

    def _visit(self, node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._function(child)
            else:
                self._visit(child)

    def _function(self, node: ast.AST) -> None:
        if region_is_dynamic(node):
            self.skipped_scopes += 1
            return
        for child in ast.iter_child_nodes(node):
            self._visit(child)  # nested scopes handled independently
        containers = fresh_container_names(node) if self._len_ok else frozenset()
        trusted = self._trust.ints(node)
        ints = infer_int_names(
            node,
            containers,
            range_ok=self._range_ok,
            trusted=trusted,
            typed_calls=self._trust.int_returns,
        )
        if not ints and not containers:
            return
        base = infer_int_ranges(
            node,
            ints,
            containers,
            range_ok=self._range_ok,
            trusted=trusted,
        )
        self._base = base
        self._ints = ints
        self._containers = containers
        self._walk_block(node.body, dict(base))

    # -- environment helpers ------------------------------------------------
    def _reset(self, env: dict, names) -> None:
        for name in names:
            if name in self._base:
                env[name] = self._base[name]

    def _fold(self, expr: ast.expr, env: dict) -> ast.expr:
        return _Folder(env, self._ints, self._containers, self).visit(expr)

    def _apply_fact(self, env: dict, test: ast.expr, truth: bool) -> dict:
        """Environment in which *test* is known to evaluate to *truth*."""
        env = dict(env)
        self._apply_into(env, test, truth)
        return env

    def _apply_into(self, env: dict, test: ast.expr, truth: bool) -> None:
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            self._apply_into(env, test.operand, not truth)
            return
        if isinstance(test, ast.BoolOp):
            # ``and`` gives facts when true, ``or`` when false.
            wants = isinstance(test.op, ast.And)
            if truth == wants:
                for value in test.values:
                    self._apply_into(env, value, truth)
            return
        if isinstance(test, ast.Name) and test.id in self._ints:
            current = env.get(test.id)
            if current is None:
                return
            lo, hi = current
            if not truth:
                env[test.id] = (max(lo, 0), min(hi, 0)) if lo <= 0 <= hi else current
            elif lo == 0:
                env[test.id] = (1, hi) if hi >= 1 else current
            elif hi == 0:
                env[test.id] = (lo, -1) if lo <= -1 else current
            return
        if not isinstance(test, ast.Compare):
            return
        if len(test.ops) > 1:
            # A true chain means every leg holds; a false chain only means
            # *some* leg failed, which narrows nothing.
            if truth:
                operands = [test.left, *test.comparators]
                for k, op in enumerate(test.ops):
                    self._apply_leg(env, operands[k], op, operands[k + 1])
            return
        op = test.ops[0]
        if not truth:
            negated = _NEGATE.get(type(op))
            if negated is None:
                return
            op = negated()
        self._apply_leg(env, test.left, op, test.comparators[0])

    def _apply_leg(self, env: dict, left, op, right) -> None:
        # ``name <op> expr`` and the mirrored ``expr <op> name``.
        for target, other, effective in (
            (left, right, op),
            (right, left, _mirror(op)),
        ):
            if (
                effective is not None
                and isinstance(target, ast.Name)
                and target.id in self._ints
                and is_int_expr(other, self._ints, self._containers)
            ):
                bound = int_expr_range(other, env, self._containers)
                env[target.id] = _narrow_one(
                    env.get(target.id), effective, bound
                )

    # -- block walk ---------------------------------------------------------
    def _walk_block(self, stmts: list, env: dict) -> None:
        for i, stmt in enumerate(stmts):
            pending_fact = None
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef)):
                self._reset(env, subtree_bindings(stmt))
                continue
            if isinstance(stmt, ast.If):
                stmt.test = self._fold(stmt.test, env)
                body_env = self._apply_fact(env, stmt.test, True)
                else_env = self._apply_fact(env, stmt.test, False)
                self._walk_block(stmt.body, body_env)
                self._walk_block(stmt.orelse, else_env)
                # Guard clause: when one branch always leaves the block,
                # reaching the code below proves the *other* side held.
                fell_through = None
                if _always_exits(stmt.body) and not _always_exits(stmt.orelse):
                    fell_through = False
                elif _always_exits(stmt.orelse) and not _always_exits(stmt.body):
                    fell_through = True
                if fell_through is not None:
                    # Applied after the reset below, so it survives it.
                    pending_fact = (stmt.test, fell_through)
            elif isinstance(stmt, _LOOPS):
                # The body may run after earlier iterations rebound names.
                inner = dict(env)
                self._reset(inner, subtree_bindings(stmt))
                if isinstance(stmt, ast.While):
                    stmt.test = self._fold(stmt.test, inner)
                    inner = self._apply_fact(inner, stmt.test, True)
                else:
                    stmt.iter = self._fold(stmt.iter, env)
                self._walk_block(stmt.body, inner)
                self._walk_block(stmt.orelse, dict(env))
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                self._walk_block(stmt.body, env)
            elif isinstance(stmt, ast.Try) or (
                hasattr(ast, "TryStar") and isinstance(stmt, ast.TryStar)
            ):
                self._walk_block(stmt.body, dict(env))
                for handler in stmt.handlers:
                    self._walk_block(handler.body, dict(env))
                self._walk_block(stmt.orelse, dict(env))
                self._walk_block(stmt.finalbody, dict(env))
            elif isinstance(stmt, ast.Match):
                for case in stmt.cases:
                    self._walk_block(case.body, dict(env))
            else:
                stmts[i] = self._fold(stmt, env)
                stmt = stmts[i]
            # Straight-line assignment transfer: the stored value's interval
            # under the *pre-statement* environment (the right-hand side is
            # evaluated before the store), which is strictly sharper than
            # falling back to the flow-insensitive base.
            transferred = self._transfer(stmt, env)
            self._reset(env, subtree_bindings(stmt))
            env.update(transferred)
            if pending_fact is not None:
                self._apply_into(env, pending_fact[0], pending_fact[1])

    def _transfer(self, stmt: ast.stmt, env: dict) -> dict:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id in self._ints
            and is_int_expr(stmt.value, self._ints, self._containers)
        ):
            new = int_expr_range(stmt.value, env, self._containers)
            if new is not None:
                return {stmt.targets[0].id: new}
        elif (
            isinstance(stmt, ast.AugAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id in self._ints
            and isinstance(stmt.op, INT_BIN_OPS)
            and is_int_expr(stmt.value, self._ints, self._containers)
        ):
            current = env.get(stmt.target.id)
            value = int_expr_range(stmt.value, env, self._containers)
            new = _binop_range(stmt.op, current, value)
            if new is not None:
                return {stmt.target.id: new}
        return {}


def _always_exits(stmts) -> bool:
    """True if control can never fall out of the bottom of *stmts* (the
    block ends in return/raise/break/continue, or in an ``if`` whose two
    sides both always exit)."""
    if not stmts:
        return False
    last = stmts[-1]
    if isinstance(last, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
        return True
    if isinstance(last, ast.If):
        return _always_exits(last.body) and _always_exits(last.orelse)
    return False


def _mirror(op):
    """``a <op> b`` rewritten as ``b <mirror> a`` (None when not needed)."""
    table = {
        ast.Lt: ast.Gt,
        ast.LtE: ast.GtE,
        ast.Gt: ast.Lt,
        ast.GtE: ast.LtE,
        ast.Eq: ast.Eq,
        ast.NotEq: ast.NotEq,
    }
    mirrored = table.get(type(op))
    return mirrored() if mirrored is not None else None
