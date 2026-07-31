"""Constant folding.

Folds arithmetic / string / comparison / boolean expressions whose operands
are all literal constants.  Anything that raises at evaluation time (e.g.
``1 / 0``, ``1 < "a"``) is left untouched so the runtime error is preserved.
Results that would blow up the source (huge ints, long strings) are skipped.

Constant *containers* participate too: a tuple display of constants becomes
one constant tuple (which const-prop can then propagate), constant
subscripts / slices / ``len()`` / membership tests on it fold, and call
arguments unpacking a literal display or constant tuple flatten into plain
arguments -- ``f(*(a, b))`` -> ``f(a, b)``, ``f(**{'k': v})`` -> ``f(k=v)``
(evaluation order is the display's own order, so this is exact even for
impure elements; ``**`` keys must be unique valid identifiers that do not
collide with other keywords, keeping every runtime ``TypeError``).
"""

from __future__ import annotations

import ast
import keyword as _keyword
import operator

from ..analysis import builtin_gate
from .base import ScopedTransformer

#: Guardrails on optimisation-time work and emitted literal size.  The
#: aggressive ``budgets`` option swaps in the generous set: nothing about
#: the *semantics* changes, only how much folding is considered worthwhile
#: (bigger literals in the output, more work at optimise time).
class _Limits:
    __slots__ = ("seq_len", "pow_exp", "shift", "int_bits")

    def __init__(self, seq_len, pow_exp, shift, int_bits) -> None:
        self.seq_len = seq_len
        self.pow_exp = pow_exp
        self.shift = shift
        self.int_bits = int_bits


DEFAULT_LIMITS = _Limits(seq_len=4096, pow_exp=64, shift=256, int_bits=4096)
GENEROUS_LIMITS = _Limits(
    seq_len=1 << 20, pow_exp=4096, shift=1 << 16, int_bits=1 << 20
)

_SKIP = object()

_FOLDABLE_TYPES = (int, float, complex, bool, str, bytes)
_COMPARE_TYPES = _FOLDABLE_TYPES + (type(None), tuple)
#: Values allowed inside a folded constant tuple / as a fold result.
_CONST_ELEMENT_TYPES = _FOLDABLE_TYPES + (type(None), type(...), tuple)


def _const_size(value) -> int:
    """Rough emitted-source footprint of a constant (chars for sequences,
    element count plus nesting for tuples) -- guards against pasting a huge
    literal into every use site."""
    if isinstance(value, (str, bytes)):
        return len(value)
    if type(value) is tuple:
        return len(value) + sum(_const_size(v) for v in value)
    return 1

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


def _check_result(value, limits=DEFAULT_LIMITS):
    if type(value) is tuple:
        if any(type(v) not in _CONST_ELEMENT_TYPES for v in value):
            return _SKIP
        return value if _const_size(value) <= limits.seq_len else _SKIP
    if value is None or value is Ellipsis:
        return value
    if type(value) not in _FOLDABLE_TYPES:
        return _SKIP
    if isinstance(value, (str, bytes)) and len(value) > limits.seq_len:
        return _SKIP
    if type(value) is int and value.bit_length() > limits.int_bits:
        return _SKIP
    return value


def _fold_binop(op: ast.operator, left, right, limits=DEFAULT_LIMITS):
    # Pre-guards against expressions that are cheap to write but expensive
    # to evaluate (they must not hang or exhaust memory at optimise time).
    if isinstance(op, ast.Pow):
        if type(right) is int and abs(right) > limits.pow_exp:
            return _SKIP
        if type(left) is int and type(right) is int and right > 0:
            if left.bit_length() * right > limits.int_bits:
                return _SKIP
    if isinstance(op, ast.LShift) and type(right) is int and right > limits.shift:
        return _SKIP
    if isinstance(op, ast.Mult):
        if isinstance(left, (str, bytes)) and type(right) is int:
            if len(left) * max(right, 0) > limits.seq_len:
                return _SKIP
        if isinstance(right, (str, bytes)) and type(left) is int:
            if len(right) * max(left, 0) > limits.seq_len:
                return _SKIP
    fn = _BIN_OPS.get(type(op))
    if fn is None:
        return _SKIP
    try:
        value = fn(left, right)
    except Exception:
        return _SKIP
    return _check_result(value, limits)


class ConstantFolding(ScopedTransformer):
    name = "constant-folding"

    @property
    def _limits(self):
        return (
            GENEROUS_LIMITS if "budgets" in self.aggressive else DEFAULT_LIMITS
        )

    def run(self, tree: ast.Module) -> ast.Module:
        self._len_ok = builtin_gate(tree, "len")
        return super().run(tree)

    def _const(self, value, node: ast.AST) -> ast.Constant:
        self.changes += 1
        return ast.copy_location(ast.Constant(value), node)

    def visit_Tuple(self, node: ast.Tuple) -> ast.AST:
        node = self.generic_visit(node)
        if not isinstance(node.ctx, ast.Load):
            return node
        if not all(
            isinstance(e, ast.Constant)
            and type(e.value) in _CONST_ELEMENT_TYPES
            for e in node.elts
        ):
            return node
        value = tuple(e.value for e in node.elts)
        if _const_size(value) > self._limits.seq_len:
            return node
        return self._const(value, node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if (
            self._len_ok
            and isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and len(node.args) == 1
            and not node.keywords
            and isinstance(node.args[0], ast.Constant)
            and type(node.args[0].value) in (str, bytes, tuple)
        ):
            return self._const(len(node.args[0].value), node)
        return self._expand_call(node)

    def _expand_call(self, node: ast.Call) -> ast.Call:
        """Flattens ``*<display/constant tuple>`` and ``**<literal dict>``
        arguments.  Displays evaluate their elements left to right at the
        argument's own position, so splicing the elements in changes
        nothing about order or effects; a rejected ``**`` entry (non-str /
        non-identifier / duplicate / colliding key) is left alone so the
        runtime behaviour -- including ``TypeError`` -- is preserved."""
        changed = False
        new_args: list[ast.expr] = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                v = arg.value
                if (
                    isinstance(v, (ast.Tuple, ast.List))
                    and isinstance(v.ctx, ast.Load)
                    and not any(isinstance(e, ast.Starred) for e in v.elts)
                ):
                    new_args.extend(v.elts)
                    changed = True
                    continue
                if isinstance(v, ast.Constant) and type(v.value) is tuple:
                    new_args.extend(
                        ast.copy_location(ast.Constant(x), arg)
                        for x in v.value
                    )
                    changed = True
                    continue
            new_args.append(arg)
        new_keywords: list[ast.keyword] = []
        taken = {kw.arg for kw in node.keywords if kw.arg is not None}
        for kw in node.keywords:
            if kw.arg is None and isinstance(kw.value, ast.Dict):
                display = kw.value
                names: list[str] = []
                ok = True
                for key in display.keys:
                    if (
                        isinstance(key, ast.Constant)
                        and type(key.value) is str
                        and key.value.isidentifier()
                        and not _keyword.iskeyword(key.value)
                        and key.value not in names
                        and key.value not in taken
                    ):
                        names.append(key.value)
                    else:
                        ok = False
                        break
                if ok:
                    for name, value in zip(names, display.values):
                        new_keywords.append(
                            ast.copy_location(
                                ast.keyword(arg=name, value=value), kw
                            )
                        )
                    taken.update(names)
                    changed = True
                    continue
            new_keywords.append(kw)
        if changed:
            self.changes += 1
            node.args = new_args
            node.keywords = new_keywords
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node = self.generic_visit(node)
        if _foldable(node.left) and _foldable(node.right):
            value = _fold_binop(
                node.op, node.left.value, node.right.value, self._limits
            )
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
                value = _check_result(value, self._limits)
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
        if not (
            isinstance(value, ast.Constant)
            and type(value.value) in (str, bytes, tuple)
        ):
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
