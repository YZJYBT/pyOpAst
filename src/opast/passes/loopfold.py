"""Bounded closed-form loop evaluation (loop folding).

A ``for <name> in range(<constants>):`` loop whose body is pure int
arithmetic -- assignments, augmented assignments and ``if``/``elif``/
``else`` over int/bool constants, previously-established constants and the
loop counter, with no calls and no other constructs -- is *simulated* at
optimisation time and replaced by direct assignments of the final values:

    total = 0                    total = 0
    for i in range(1000):   ->   i = 999
        total += i * 2           total = 999000

The simulation is exact: it performs the very computation the runtime
would, so results (including bool-ness, floor-division sign behaviour,
chained comparisons, short-circuit values) are identical, and any
simulated exception (``ZeroDivisionError``, negative shifts) aborts the
fold so the runtime error is preserved.  Because simulation is exact and
deterministic, a successful fold also proves the loop cannot raise.

Initial values come from the run of plain ``<name> = <int/bool constant>``
assignments immediately preceding the loop (those stay in place; the
unused pass removes the dead stores in function scopes).  Reading any
other name aborts.  The loop is replaced by one constant assignment per
name the simulation actually stored, plus the loop target's final value
when the trip count is >= 1 (zero-trip loops disappear entirely -- their
``range()`` evaluation is pure).  Assignment order follows first-store
order, so rebinding effects are preserved exactly.

Guardrails mirror the folding pass: the optimiser never does unbounded
work (total simulated steps bounded by :data:`_MAX_STEPS`, pre-checked
via trip count x body size) and never produces giant literals (every
intermediate and final value must stay within :data:`_MAX_BITS` bits;
``**`` exponents are capped).  Exceeding a guard keeps the loop intact.

Safety notes:

* the whole pass is gated on the ``range`` builtin gate (no dynamic
  constructs module-wide, ``range`` unbound everywhere) -- frame
  introspection that could observe the removed loop variables is gated
  off with the rest of the dynamic constructs;
* the body contains no calls and no expression that can run user code, so
  nothing can observe the per-iteration intermediate values the rewrite
  collapses -- this is why module scope and ``global``-declared names are
  fine, and why folding inside ``try``/``with`` is sound (the loop
  provably cannot raise).  As elsewhere in opast, another *thread*
  polling the bindings mid-loop is not part of the semantics contract;
* class bodies are left alone: a metaclass ``__prepare__`` mapping would
  observe every per-iteration store this rewrite removes.
"""

from __future__ import annotations

import ast
import operator

from ..analysis import builtin_gate, for_range_binding
from .base import ScopedTransformer

_MAX_STEPS = 200_000
_MAX_BITS = 4096
_MAX_POW_EXP = 64
_MAX_SHIFT = 256

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
}

_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Invert: operator.invert,
    ast.Not: operator.not_,
}

_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


class _Abort(Exception):
    """Fold abandoned: guard exceeded, unknown name, runtime error, ..."""


def _validate_stmt(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Assign):
        return (
            len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and _validate_expr(stmt.value)
        )
    if isinstance(stmt, ast.AugAssign):
        return (
            isinstance(stmt.target, ast.Name)
            and type(stmt.op) in _BIN_OPS
            and _validate_expr(stmt.value)
        )
    if isinstance(stmt, ast.If):
        return (
            _validate_expr(stmt.test)
            and all(_validate_stmt(s) for s in stmt.body)
            and all(_validate_stmt(s) for s in stmt.orelse)
        )
    if isinstance(stmt, ast.Pass):
        return True
    return False


def _validate_expr(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return type(node.value) in (int, bool)
    if isinstance(node, ast.Name):
        return isinstance(node.ctx, ast.Load)
    if isinstance(node, ast.BinOp):
        return (
            type(node.op) in _BIN_OPS
            and _validate_expr(node.left)
            and _validate_expr(node.right)
        )
    if isinstance(node, ast.UnaryOp):
        return type(node.op) in _UNARY_OPS and _validate_expr(node.operand)
    if isinstance(node, ast.Compare):
        return (
            all(type(op) in _CMP_OPS for op in node.ops)
            and _validate_expr(node.left)
            and all(_validate_expr(c) for c in node.comparators)
        )
    if isinstance(node, ast.BoolOp):
        return all(_validate_expr(v) for v in node.values)
    if isinstance(node, ast.IfExp):
        return (
            _validate_expr(node.test)
            and _validate_expr(node.body)
            and _validate_expr(node.orelse)
        )
    return False


class LoopFolding(ScopedTransformer):
    name = "loop-fold"

    def run(self, tree: ast.Module) -> ast.Module:
        if not builtin_gate(tree, "range"):
            self.skipped_scopes += 1
            return tree
        return self.visit(tree)

    def visit_Module(self, node: ast.Module) -> ast.AST:
        node.body = self._process_block(node.body)
        return self.generic_visit(node)

    def _visit_function(self, node: ast.AST) -> ast.AST:
        node.body = self._process_block(node.body)
        return self.generic_visit(node)

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function
    # ClassDef keeps the base behaviour: descend into methods only.

    # -- block walk (bottom-up: inner loops fold first, which can make the
    # enclosing loop body pure-arithmetic on the next pipeline iteration) ---
    def _process_block(self, stmts: list) -> list:
        out: list[ast.stmt] = []
        for stmt in stmts:
            self._descend(stmt)
            if isinstance(stmt, ast.For):
                replacement = self._try_fold(out, stmt)
                if replacement is not None:
                    out.extend(replacement)
                    self.changes += 1
                    continue
            out.append(stmt)
        return out

    def _descend(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            stmt.body = self._process_block(stmt.body)
            stmt.orelse = self._process_block(stmt.orelse)
        elif isinstance(stmt, ast.If):
            stmt.body = self._process_block(stmt.body)
            stmt.orelse = self._process_block(stmt.orelse)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            stmt.body = self._process_block(stmt.body)
        elif isinstance(stmt, ast.Try) or (
            hasattr(ast, "TryStar") and isinstance(stmt, ast.TryStar)
        ):
            stmt.body = self._process_block(stmt.body)
            for handler in stmt.handlers:
                handler.body = self._process_block(handler.body)
            stmt.orelse = self._process_block(stmt.orelse)
            stmt.finalbody = self._process_block(stmt.finalbody)
        elif isinstance(stmt, ast.Match):
            for case in stmt.cases:
                case.body = self._process_block(case.body)
        # Nested scope statements are handled by the visitor dispatch.

    # -- the fold -----------------------------------------------------------
    def _try_fold(self, prev: list, loop: ast.For):
        info = for_range_binding(loop)
        if info is None or loop.orelse:
            return None
        target, args = info
        if not all(
            isinstance(a, ast.Constant) and type(a.value) is int for a in args
        ):
            return None
        try:
            rng = range(*[a.value for a in args])
        except ValueError:
            return None  # step 0: the runtime error is preserved
        if not all(_validate_stmt(s) for s in loop.body):
            return None
        # Cheap upper bound before simulating: every simulated step consumes
        # at most one body node per iteration.
        weight = sum(1 for _ in ast.walk(loop))
        if len(rng) * weight > _MAX_STEPS:
            return None

        # Initial values: the run of plain int/bool constant assignments
        # immediately preceding the loop (nearest binding wins).
        env: dict[str, int] = {}
        for stmt in reversed(prev):
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Constant)
                and type(stmt.value.value) in (int, bool)
            ):
                env.setdefault(stmt.targets[0].id, stmt.value.value)
            else:
                break

        self._budget = _MAX_STEPS
        write_order: list[str] = []
        try:
            for value in rng:
                env[target] = value
                for stmt in loop.body:
                    self._exec(stmt, env, write_order)
        except _Abort:
            return None

        out = []
        names = ([target] if len(rng) else []) + [
            n for n in write_order if n != target
        ]
        for name in names:
            assign = ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=ast.Constant(env[name]),
            )
            out.append(ast.copy_location(assign, loop))
        return out

    # -- exact mini-interpreter --------------------------------------------
    def _step(self) -> None:
        self._budget -= 1
        if self._budget < 0:
            raise _Abort

    def _exec(self, stmt: ast.stmt, env: dict, write_order: list) -> None:
        self._step()
        if isinstance(stmt, ast.Assign):
            value = self._eval(stmt.value, env)
            self._store(stmt.targets[0].id, value, env, write_order)
        elif isinstance(stmt, ast.AugAssign):
            name = stmt.target.id
            if name not in env:
                raise _Abort  # would be UnboundLocalError/NameError
            value = self._binop(
                stmt.op, env[name], self._eval(stmt.value, env)
            )
            self._store(name, value, env, write_order)
        elif isinstance(stmt, ast.If):
            branch = stmt.body if self._eval(stmt.test, env) else stmt.orelse
            for s in branch:
                self._exec(s, env, write_order)
        # ast.Pass: nothing

    @staticmethod
    def _store(name: str, value, env: dict, write_order: list) -> None:
        if name not in write_order:
            write_order.append(name)
        env[name] = value

    def _eval(self, node: ast.expr, env: dict):
        self._step()
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in env:
                raise _Abort
            return env[node.id]
        if isinstance(node, ast.BinOp):
            return self._binop(
                node.op, self._eval(node.left, env), self._eval(node.right, env)
            )
        if isinstance(node, ast.UnaryOp):
            result = _UNARY_OPS[type(node.op)](self._eval(node.operand, env))
            return self._check(result)
        if isinstance(node, ast.Compare):
            left = self._eval(node.left, env)
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval(comparator, env)
                if not _CMP_OPS[type(op)](left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.BoolOp):
            is_and = isinstance(node.op, ast.And)
            value = None
            for i, sub in enumerate(node.values):
                value = self._eval(sub, env)
                if i < len(node.values) - 1 and bool(value) != is_and:
                    return value  # short-circuit: this operand is the result
            return value
        if isinstance(node, ast.IfExp):
            chosen = node.body if self._eval(node.test, env) else node.orelse
            return self._eval(chosen, env)
        raise _Abort  # unreachable after validation

    def _binop(self, op: ast.operator, left, right):
        if isinstance(op, ast.Pow):
            if abs(right) > _MAX_POW_EXP:
                raise _Abort
            if right > 0 and left.bit_length() * right > _MAX_BITS:
                raise _Abort
        if isinstance(op, ast.LShift) and right > _MAX_SHIFT:
            raise _Abort
        try:
            result = _BIN_OPS[type(op)](left, right)
        except Exception:
            raise _Abort from None  # runtime would raise: keep the loop
        return self._check(result)

    @staticmethod
    def _check(value):
        # ``2 ** -1`` is a float; anything non-int leaves our exact world.
        if type(value) not in (int, bool):
            raise _Abort
        if type(value) is int and value.bit_length() > _MAX_BITS:
            raise _Abort
        return value
