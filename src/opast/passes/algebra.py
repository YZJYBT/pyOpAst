"""Algebraic simplification and integer strength reduction.

Identities like ``x + 0 -> x`` are *not* valid for arbitrary Python objects
(custom ``__add__``, ``-0.0 + 0 == 0.0``, ``True * 1 == 1`` ...), so this pass
only rewrites expressions that are provably plain ``int``-valued, using the
per-function inference from :mod:`opast.analysis` (which, when ``range`` is
trustworthy module-wide, also proves ``for i in range(...)`` targets).
Applied identities (``E`` a proven int expression, constants exact ``int``):

* ``E + 0``, ``0 + E``, ``E - 0``  -> ``E``;   ``0 - E`` -> ``-E``
* ``E * 1``, ``1 * E``, ``E ** 1``, ``E // 1`` -> ``E``
* ``E << 0``, ``E >> 0``, ``E | 0``, ``0 | E``, ``E ^ 0``, ``0 ^ E`` -> ``E``
* ``+E`` -> ``E``;  ``-(-E)`` -> ``E``;  ``~~E`` -> ``E``

Strength reduction (``c`` a constant power of two >= 2; both rewrites hold
for *all* Python ints because ``//`` is floor division and ints behave as
infinite two's complement):

* ``E % c``  -> ``E & (c-1)``
* ``E // c`` -> ``E >> log2(c)``
* ``E ** 2`` -> ``E * E`` for small pure ``E`` (evaluating a pure, total int
  expression twice is unobservable; ``pow`` costs far more than the re-eval)

Interval-based (using :func:`opast.analysis.infer_int_ranges`):

* ``abs(E)`` -> ``E`` when ``E``'s interval proves it non-negative (gated
  module-wide on ``abs`` being the untouched builtin)

Proven ``float`` expressions (:func:`opast.analysis.infer_float_names`) get
only the identities that are **bit-exact** for every float input --
``-0.0``, the infinities and ``NaN`` included.  Verified empirically:

* safe:   ``F - 0``, ``F * 1``, ``1 * F``, ``F / 1``, ``F ** 1``, ``+F``,
  ``-(-F)`` (with an ``int`` *or* ``float`` constant)
* unsafe, deliberately not applied: ``F + 0`` and ``0 + F`` (they turn
  ``-0.0`` into ``0.0``), ``F // 1`` (floors: ``1.5 // 1 == 1.0``), and
  every bitwise identity (a ``TypeError`` on floats)

Strength reduction stays int-only: ``%``/``//`` by a power of two mean
something different for floats, and ``F ** 2 -> F * F`` would not be
bit-exact for subnormals.

``E`` itself is always kept, never dropped, so side effects and exceptions
inside ``E`` are preserved.  ``E ** 2`` is the one deliberate duplication:
allowed only because ``E`` is provably pure and deterministic.
"""

from __future__ import annotations

import ast
import copy
import math

from ..analysis import (
    AnnotationTrust,
    builtin_gate,
    hoistable_int_expr,
    infer_float_names,
    infer_int_names,
    infer_int_ranges,
    int_expr_range,
    is_float_expr,
    is_int_expr,
)
from ..safety import region_is_dynamic
from .base import ScopedTransformer

#: Node-count budget for the ``E ** 2 -> E * E`` duplication.
_POW_DUP_MAX_NODES = 5
_GENEROUS_POW_DUP_MAX_NODES = 24

#: Smallest positive normal double: a reciprocal at or below this loses the
#: exactness that makes the division-to-multiplication rewrite bit-safe.
_MIN_NORMAL = 2.2250738585072014e-308


def _as_float_constant(node: ast.AST):
    """The value of a non-zero, finite numeric constant as a float, or None.

    An ``int`` literal too large for a double (``10 ** 400``) raises
    ``OverflowError`` on conversion -- the runtime raises there too, so the
    rewrite is simply declined and the original division kept.
    """
    if not (isinstance(node, ast.Constant) and type(node.value) in (int, float)):
        return None
    try:
        value = float(node.value)
    except OverflowError:
        return None
    if not math.isfinite(value) or value == 0.0:
        return None
    return value


def _exact_reciprocal(node: ast.AST):
    """``1/c`` when *c* is a constant for which ``x / c`` and ``x * (1/c)``
    are **bit-identical** for every finite ``x`` -- that is, a power of two
    whose reciprocal is still a normal double.  Verified empirically over
    subnormals, huge values and signed zeros.  Non-powers of two only agree
    to within an ulp, so they are left to the ``fastmath`` option."""
    value = _as_float_constant(node)
    if value is None:
        return None
    mantissa, _ = math.frexp(value)
    if abs(mantissa) != 0.5:  # not a power of two
        return None
    recip = 1.0 / value
    if not math.isfinite(recip) or abs(recip) < _MIN_NORMAL:
        return None
    return recip


def _int_const(node: ast.AST, value: int) -> bool:
    return (
        isinstance(node, ast.Constant)
        and type(node.value) is int
        and node.value == value
    )


def _num_const(node: ast.AST, value: int) -> bool:
    """An exact ``int`` *or* ``float`` constant equal to *value* -- both
    forms are bit-exact in the float identities this pass applies."""
    return (
        isinstance(node, ast.Constant)
        and type(node.value) in (int, float)
        and node.value == value
    )


def _pow2_const(node: ast.AST):
    """The value of a constant power-of-two int >= 2, else None."""
    if (
        isinstance(node, ast.Constant)
        and type(node.value) is int
        and node.value >= 2
        and node.value & (node.value - 1) == 0
    ):
        return node.value
    return None


class AlgebraicSimplification(ScopedTransformer):
    name = "algebraic"

    def __init__(self) -> None:
        super().__init__()
        # Module level uses an empty set: module globals can be rebound by
        # called functions, so nothing is provably int there.
        self._proven: list[frozenset[str]] = [frozenset()]
        self._floats: list[frozenset[str]] = [frozenset()]
        self._ranges: list[dict] = [{}]
        self._range_ok = False
        self._abs_ok = False

    def run(self, tree: ast.Module) -> ast.Module:
        if region_is_dynamic(tree):
            self.skipped_scopes += 1
            return tree
        self._range_ok = builtin_gate(tree, "range")
        self._abs_ok = builtin_gate(tree, "abs")
        self._min_ok = builtin_gate(tree, "min")
        self._max_ok = builtin_gate(tree, "max")
        self._trust = AnnotationTrust(
            tree if "annotations" in self.aggressive else None
        )
        return self.visit(tree)

    def _visit_function(self, node: ast.AST) -> ast.AST:
        if region_is_dynamic(node):
            self.skipped_scopes += 1
            return node
        trusted_ints = self._trust.ints(node)
        proven = infer_int_names(
            node,
            range_ok=self._range_ok,
            trusted=trusted_ints,
            typed_calls=self._trust.int_returns,
        )
        self._proven.append(proven)
        self._floats.append(
            infer_float_names(
                node,
                ints=proven,
                trusted=self._trust.floats(node),
                typed_calls=self._trust.float_returns,
            )
        )
        self._ranges.append(
            infer_int_ranges(
                node, proven, range_ok=self._range_ok, trusted=trusted_ints
            )
        )
        try:
            return self.generic_visit(node)
        finally:
            self._proven.pop()
            self._floats.pop()
            self._ranges.pop()

    visit_FunctionDef = _visit_function
    visit_AsyncFunctionDef = _visit_function

    def _visit_opaque(self, node: ast.AST) -> ast.AST:
        if region_is_dynamic(node):
            self.skipped_scopes += 1
            return node
        self._proven.append(frozenset())
        self._floats.append(frozenset())
        self._ranges.append({})
        try:
            return self.generic_visit(node)
        finally:
            self._proven.pop()
            self._floats.pop()
            self._ranges.pop()

    visit_Lambda = _visit_opaque
    visit_ClassDef = _visit_opaque

    def _is_float(self, node: ast.AST) -> bool:
        return is_float_expr(node, self._floats[-1], self._proven[-1])

    @property
    def _fast_math(self) -> bool:
        return "fastmath" in self.aggressive

    @property
    def _pow_dup_budget(self) -> int:
        return (
            _GENEROUS_POW_DUP_MAX_NODES
            if "budgets" in self.aggressive
            else _POW_DUP_MAX_NODES
        )

    @staticmethod
    def _inexact_reciprocal(node: ast.AST):
        """``1/c`` for any non-zero finite constant -- accurate to within an
        ulp rather than exact, hence fastmath only."""
        value = _as_float_constant(node)
        if value is None:
            return None
        recip = 1.0 / value
        if not math.isfinite(recip) or abs(recip) < _MIN_NORMAL:
            return None
        return recip

    def _fastmath_binop(self, op, left, right):
        """Float rewrites that are *not* bit-exact (see module docstring)."""
        # ``F + 0`` / ``0 + F``: only ``-0.0`` observes the difference, and
        # dropping the operation removes a whole dispatch.
        if isinstance(op, ast.Add):
            if _num_const(right, 0) and self._is_float(left):
                return left
            if _num_const(left, 0) and self._is_float(right):
                return right
        # ``F ** 2 -> F * F``: about 3x faster, but ``1e308 ** 2`` raises
        # OverflowError where the multiplication yields ``inf``.
        if (
            isinstance(op, ast.Pow)
            and _num_const(right, 2)
            and self._is_float(left)
            and sum(1 for _ in ast.walk(left)) <= self._pow_dup_budget
        ):
            return ast.BinOp(
                left=left, op=ast.Mult(), right=copy.deepcopy(left)
            )
        return None

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node = self.generic_visit(node)
        proven = self._proven[-1]
        left, right, op = node.left, node.right, node.op
        replacement = None

        # Float identities first: only the bit-exact ones (see docstring).
        # ``F + 0`` and ``F // 1`` are deliberately absent.
        if self._floats[-1]:
            if isinstance(op, ast.Sub) and _num_const(right, 0):
                if self._is_float(left):
                    replacement = left
            elif isinstance(op, ast.Mult):
                if _num_const(right, 1) and self._is_float(left):
                    replacement = left
                elif _num_const(left, 1) and self._is_float(right):
                    replacement = right
            elif (
                isinstance(op, ast.Div)
                and _num_const(right, 1)
                # PEP 484 lets an ``int`` argument satisfy a ``float``
                # annotation, and ``5 / 1`` is ``5.0`` while ``5`` is not --
                # so this identity is off whenever annotations are trusted.
                and not self._trust.enabled
            ):
                if self._is_float(left):
                    replacement = left
            elif isinstance(op, ast.Pow) and _num_const(right, 1):
                if self._is_float(left):
                    replacement = left
            if replacement is None and isinstance(op, ast.Div):
                # ``F / 2**k -> F * 2**-k`` is bit-identical (the reciprocal
                # of a power of two is exact); other divisors only agree to
                # within an ulp and need the fastmath option.
                recip = _exact_reciprocal(right)
                if recip is None and self._fast_math:
                    recip = self._inexact_reciprocal(right)
                if recip is not None and self._is_float(left):
                    replacement = ast.BinOp(
                        left=left, op=ast.Mult(), right=ast.Constant(recip)
                    )
            if replacement is None and self._fast_math:
                replacement = self._fastmath_binop(op, left, right)
            if replacement is not None:
                self.changes += 1
                return ast.copy_location(replacement, node)

        if isinstance(op, ast.Add):
            if _int_const(right, 0) and is_int_expr(left, proven):
                replacement = left
            elif _int_const(left, 0) and is_int_expr(right, proven):
                replacement = right
        elif isinstance(op, ast.Sub):
            if _int_const(right, 0) and is_int_expr(left, proven):
                replacement = left
            elif _int_const(left, 0) and is_int_expr(right, proven):
                replacement = ast.UnaryOp(op=ast.USub(), operand=right)
        elif isinstance(op, ast.Mult):
            if _int_const(right, 1) and is_int_expr(left, proven):
                replacement = left
            elif _int_const(left, 1) and is_int_expr(right, proven):
                replacement = right
        elif isinstance(op, ast.Pow):
            if _int_const(right, 1) and is_int_expr(left, proven):
                replacement = left
            elif (
                _int_const(right, 2)
                and is_int_expr(left, proven)
                and sum(1 for _ in ast.walk(left)) <= self._pow_dup_budget
            ):
                replacement = ast.BinOp(
                    left=left, op=ast.Mult(), right=copy.deepcopy(left)
                )
        elif isinstance(op, ast.FloorDiv):
            if _int_const(right, 1) and is_int_expr(left, proven):
                replacement = left
            else:
                c = _pow2_const(right)
                if c is not None and is_int_expr(left, proven):
                    replacement = ast.BinOp(
                        left=left,
                        op=ast.RShift(),
                        right=ast.Constant(c.bit_length() - 1),
                    )
        elif isinstance(op, ast.Mod):
            c = _pow2_const(right)
            if c is not None and is_int_expr(left, proven):
                replacement = ast.BinOp(
                    left=left, op=ast.BitAnd(), right=ast.Constant(c - 1)
                )
        elif isinstance(op, (ast.LShift, ast.RShift)):
            if _int_const(right, 0) and is_int_expr(left, proven):
                replacement = left
        elif isinstance(op, (ast.BitOr, ast.BitXor)):
            if _int_const(right, 0) and is_int_expr(left, proven):
                replacement = left
            elif _int_const(left, 0) and is_int_expr(right, proven):
                replacement = right

        if replacement is None:
            return node
        self.changes += 1
        return ast.copy_location(replacement, node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        node = self.generic_visit(node)
        proven = self._proven[-1]
        operand = node.operand
        if isinstance(node.op, ast.UAdd) and (
            is_int_expr(operand, proven) or self._is_float(operand)
        ):
            self.changes += 1
            return ast.copy_location(operand, node)
        for double in (ast.USub, ast.Invert):
            if (
                isinstance(node.op, double)
                and isinstance(operand, ast.UnaryOp)
                and isinstance(operand.op, double)
                and (
                    is_int_expr(operand.operand, proven)
                    # ``~`` is a TypeError on floats, so only ``-(-F)``.
                    or (
                        double is ast.USub
                        and self._is_float(operand.operand)
                    )
                )
            ):
                self.changes += 1
                return ast.copy_location(operand.operand, node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if not (isinstance(node.func, ast.Name) and not node.keywords):
            return node
        name = node.func.id
        if self._abs_ok and name == "abs" and len(node.args) == 1:
            arg = node.args[0]
            if is_int_expr(arg, self._proven[-1]):
                interval = int_expr_range(arg, self._ranges[-1])
                if interval is not None and interval[0] >= 0:
                    self.changes += 1
                    return ast.copy_location(arg, node)
        if (
            name in ("min", "max")
            and len(node.args) == 2
            and (self._min_ok if name == "min" else self._max_ok)
        ):
            replacement = self._fold_clamp(name, node.args[0], node.args[1])
            if replacement is not None:
                self.changes += 1
                return ast.copy_location(replacement, node)
        return node

    def _fold_clamp(self, name: str, a: ast.expr, b: ast.expr):
        """Interval-decided two-argument ``min``/``max``.

        The kept operand only needs to be proven int (its evaluation
        stays); the *dropped* one must be pure AND total
        (:func:`hoistable_int_expr`), since it will no longer run.  Ties:
        both builtins return the **first** argument, so keeping ``a``
        tolerates equality while keeping ``b`` needs strict dominance --
        which object of two equal ints comes back is observable through
        ``is``, however unspecified that is for user code.
        """
        proven = self._proven[-1]
        if not (is_int_expr(a, proven) and is_int_expr(b, proven)):
            return None
        ra = int_expr_range(a, self._ranges[-1])
        rb = int_expr_range(b, self._ranges[-1])
        if ra is None or rb is None:
            return None
        if name == "min":
            keep_a = ra[1] <= rb[0]   # a <= b always; ties return a anyway
            keep_b = rb[1] < ra[0]    # b < a strictly
        else:
            keep_a = ra[0] >= rb[1]   # a >= b always; ties return a anyway
            keep_b = rb[0] > ra[1]    # b > a strictly
        if keep_a and hoistable_int_expr(b, proven, frozenset()):
            return a
        if keep_b and hoistable_int_expr(a, proven, frozenset()):
            return b
        return None
