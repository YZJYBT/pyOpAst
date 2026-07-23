"""Constant folding.

Folds arithmetic / string / comparison / boolean expressions whose operands
are all literal constants.  Anything that raises at evaluation time (e.g.
``1 / 0``, ``1 < "a"``) is left untouched so the runtime error is preserved.
Results that would blow up the source (huge ints, long strings) are skipped.
"""

from __future__ import annotations

import ast
import operator

from .base import ScopedTransformer

_MAX_SEQ_LEN = 4096
_MAX_POW_EXP = 64
_MAX_SHIFT = 256
_MAX_INT_BITS = 4096

_SKIP = object()

_FOLDABLE_TYPES = (int, float, complex, bool, str, bytes)
_COMPARE_TYPES = _FOLDABLE_TYPES + (type(None),)

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
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

# ``is`` / ``is not`` on literals is implementation defined -- never folded.
_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}


def _foldable(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and type(node.value) in _FOLDABLE_TYPES


def _check_result(value):
    if type(value) not in _FOLDABLE_TYPES:
        return _SKIP
    if isinstance(value, (str, bytes)) and len(value) > _MAX_SEQ_LEN:
        return _SKIP
    if type(value) is int and value.bit_length() > _MAX_INT_BITS:
        return _SKIP
    return value


def _fold_binop(op: ast.operator, left, right):
    # Pre-guards against expressions that are cheap to write but expensive
    # to evaluate (they must not hang or exhaust memory at optimise time).
    if isinstance(op, ast.Pow):
        if type(right) is int and abs(right) > _MAX_POW_EXP:
            return _SKIP
        if type(left) is int and type(right) is int and right > 0:
            if left.bit_length() * right > _MAX_INT_BITS:
                return _SKIP
    if isinstance(op, ast.LShift) and type(right) is int and right > _MAX_SHIFT:
        return _SKIP
    if isinstance(op, ast.Mult):
        if isinstance(left, (str, bytes)) and type(right) is int:
            if len(left) * max(right, 0) > _MAX_SEQ_LEN:
                return _SKIP
        if isinstance(right, (str, bytes)) and type(left) is int:
            if len(right) * max(left, 0) > _MAX_SEQ_LEN:
                return _SKIP
    fn = _BIN_OPS.get(type(op))
    if fn is None:
        return _SKIP
    try:
        value = fn(left, right)
    except Exception:
        return _SKIP
    return _check_result(value)


class ConstantFolding(ScopedTransformer):
    name = "constant-folding"

    def _const(self, value, node: ast.AST) -> ast.Constant:
        self.changes += 1
        return ast.copy_location(ast.Constant(value), node)

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node = self.generic_visit(node)
        if _foldable(node.left) and _foldable(node.right):
            value = _fold_binop(node.op, node.left.value, node.right.value)
            if value is not _SKIP:
                return self._const(value, node)
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        node = self.generic_visit(node)
        operand = node.operand
        if isinstance(operand, ast.Constant):
            if isinstance(node.op, ast.Not):
                allowed = type(operand.value) in _COMPARE_TYPES
            else:
                allowed = type(operand.value) in (int, float, complex, bool)
            fn = _UNARY_OPS.get(type(node.op))
            if allowed and fn is not None:
                try:
                    value = fn(operand.value)
                except Exception:
                    return node
                value = _check_result(value)
                if value is not _SKIP:
                    return self._const(value, node)
        return node

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        node = self.generic_visit(node)
        is_and = isinstance(node.op, ast.And)
        last = len(node.values) - 1
        out: list[ast.expr] = []
        changed = False
        for i, value in enumerate(node.values):
            if isinstance(value, ast.Constant):
                truthy = bool(value.value)
                decided = (not truthy) if is_and else truthy
                if decided:
                    # Short-circuit: this constant is the result, the rest
                    # is never evaluated.
                    out.append(value)
                    if i != last:
                        changed = True
                    break
                if i == last:
                    out.append(value)
                else:
                    changed = True  # pass-through constant dropped
            else:
                out.append(value)
        if not changed:
            return node
        self.changes += 1
        if len(out) == 1:
            return ast.copy_location(out[0], node)
        new = ast.BoolOp(op=node.op, values=out)
        return ast.copy_location(new, node)

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        node = self.generic_visit(node)
        operands = [node.left] + list(node.comparators)
        if not all(
            isinstance(o, ast.Constant) and type(o.value) in _COMPARE_TYPES
            for o in operands
        ):
            return node
        if not all(type(op) in _CMP_OPS for op in node.ops):
            return node
        try:
            result = True
            left = node.left.value
            for op, comparator in zip(node.ops, node.comparators):
                right = comparator.value
                if not _CMP_OPS[type(op)](left, right):
                    result = False
                    break
                left = right
        except Exception:
            return node
        return self._const(bool(result), node)

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            self.changes += 1
            chosen = node.body if node.test.value else node.orelse
            return ast.copy_location(chosen, node)
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        node = self.generic_visit(node)
        if not isinstance(node.ctx, ast.Load):
            return node
        value = node.value
        if not (isinstance(value, ast.Constant) and isinstance(value.value, (str, bytes))):
            return node
        sl = node.slice
        if isinstance(sl, ast.Constant) and type(sl.value) is int:
            try:
                result = value.value[sl.value]
            except Exception:
                return node
        elif isinstance(sl, ast.Slice):
            parts = []
            for bound in (sl.lower, sl.upper, sl.step):
                if bound is None:
                    parts.append(None)
                elif isinstance(bound, ast.Constant) and type(bound.value) is int:
                    parts.append(bound.value)
                else:
                    return node
            try:
                result = value.value[slice(*parts)]
            except Exception:
                return node
        else:
            return node
        return self._const(result, node)
