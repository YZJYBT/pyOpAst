"""Static analyses shared by the optimisation passes.

Everything here errs on the conservative side: when in doubt a name is
reported as bound (for shadowing checks) or as *not* provably ``int`` (for
algebraic identities).
"""

from __future__ import annotations

import ast
from collections import Counter

from .safety import iter_region, tree_has_dynamic

#: Binary operators that map ``int x int -> int`` (no ``/``, no ``**``:
#: true division always yields float and ``int ** negative_int`` is a float).
INT_BIN_OPS = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.FloorDiv,
    ast.Mod,
    ast.LShift,
    ast.RShift,
    ast.BitOr,
    ast.BitAnd,
    ast.BitXor,
)

INT_UNARY_OPS = (ast.USub, ast.UAdd, ast.Invert)

#: Builtin calls whose result is a *fresh* container object.
BUILTIN_CONTAINER_CALLS = frozenset(
    {"list", "tuple", "set", "dict", "sorted", "frozenset", "str", "bytes", "range"}
)

#: Names that must be unbound module-wide (with no dynamic constructs) for
#: the fresh-container ``len()`` machinery to trust the builtins.
LEN_GATE_NAMES = frozenset({"len"}) | BUILTIN_CONTAINER_CALLS

#: Display/comprehension expressions that always build a fresh container.
_FRESH_VALUE_TYPES = (
    ast.List,
    ast.Tuple,
    ast.Set,
    ast.Dict,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
)

# Guards shared by LICM/CSE hoistability (see hoistable_int_expr).
_HOIST_SAFE_OPS = (ast.Add, ast.Sub, ast.Mult, ast.BitOr, ast.BitAnd, ast.BitXor)
_HOIST_DIV_OPS = (ast.FloorDiv, ast.Mod)
_HOIST_SHIFT_OPS = (ast.LShift, ast.RShift)
_HOIST_MAX_SHIFT = 256


def binding_names(node: ast.AST):
    """Names bound by a single node, for non-assignment binding kinds.

    ``Name(Store/Del)`` is intentionally included so that a plain region walk
    picks up assignment/deletion targets as well.
    """
    if isinstance(node, ast.Name):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            yield node.id
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        yield node.name
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            if alias.name == "*":
                continue
            yield alias.asname or alias.name.split(".")[0]
    elif isinstance(node, (ast.Global, ast.Nonlocal)):
        yield from node.names
    elif isinstance(node, ast.ExceptHandler):
        if node.name:
            yield node.name
    elif isinstance(node, ast.MatchAs):
        if node.name:
            yield node.name
    elif isinstance(node, ast.MatchStar):
        if node.name:
            yield node.name
    elif isinstance(node, ast.MatchMapping):
        if node.rest:
            yield node.rest


def _own_params(scope: ast.AST) -> set[str]:
    if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return set()
    args = scope.args
    names = set()
    for a in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
        names.add(a.arg)
    if args.vararg is not None:
        names.add(args.vararg.arg)
    if args.kwarg is not None:
        names.add(args.kwarg.arg)
    return names


def bound_names(scope: ast.AST) -> frozenset[str]:
    """Over-approximation of every name bound anywhere in *scope*'s region
    (including its own parameters).  Used for shadowing checks."""
    names = _own_params(scope)
    for node in iter_region(scope):
        names.update(binding_names(node))
    return frozenset(names)


def all_bound_names(tree: ast.AST) -> set[str]:
    """Every name bound anywhere in the whole tree, any scope (including
    parameters).  Used to check that builtin names are never shadowed."""
    names: set[str] = set()
    for node in ast.walk(tree):
        names.update(binding_names(node))
        if isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def is_len_call(node: ast.AST, containers: frozenset[str] | set[str]) -> bool:
    """``len(x)`` where *x* is a stable fresh-container name."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "len"
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], ast.Name)
        and isinstance(node.args[0].ctx, ast.Load)
        and node.args[0].id in containers
    )


def hoistable_int_expr(node, ints, containers) -> bool:
    """Pure, total int expression: built from int constants, names in *ints*
    (proven plain-int) and ``len()`` of names in *containers* (fresh,
    never-escaping builtin containers).  ``//``/``%`` need a non-zero int
    constant divisor and shifts a 0..256 constant amount, so evaluation can
    never raise -- which is what makes speculative evaluation (LICM before a
    zero-iteration loop, CSE ahead of a conditional use) sound."""
    if isinstance(node, ast.Constant):
        return type(node.value) is int
    if isinstance(node, ast.Name):
        return isinstance(node.ctx, ast.Load) and node.id in ints
    if is_len_call(node, containers):
        return True
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, INT_UNARY_OPS) and hoistable_int_expr(
            node.operand, ints, containers
        )
    if isinstance(node, ast.BinOp):
        if isinstance(node.op, _HOIST_SAFE_OPS):
            return hoistable_int_expr(node.left, ints, containers) and (
                hoistable_int_expr(node.right, ints, containers)
            )
        if isinstance(node.op, _HOIST_DIV_OPS):
            return (
                hoistable_int_expr(node.left, ints, containers)
                and isinstance(node.right, ast.Constant)
                and type(node.right.value) is int
                and node.right.value != 0
            )
        if isinstance(node.op, _HOIST_SHIFT_OPS):
            return (
                hoistable_int_expr(node.left, ints, containers)
                and isinstance(node.right, ast.Constant)
                and type(node.right.value) is int
                and 0 <= node.right.value <= _HOIST_MAX_SHIFT
            )
    return False


def contains_name(node: ast.AST) -> bool:
    return any(isinstance(n, ast.Name) for n in ast.walk(node))


def fresh_container_names(scope: ast.AST) -> frozenset[str]:
    """Names in *scope* provably bound to a fresh builtin container whose
    length can never change.

    Requirements: bound exactly once, by ``x = <display/comprehension>`` or
    ``x = list(...)``-style builtin constructor call; and every other use of
    the name -- anywhere in the subtree, nested scopes included -- is one of
    the non-escaping, non-mutating whitelist forms: ``len(x)``, subscript
    *read* ``x[...]``, ``for ... in x``, or a bare ``if x:`` / ``while x:``
    test.  Anything else (argument passing, aliasing assignments, attribute
    or method access, subscript writes, containment in displays, returns,
    comparisons, ...) disqualifies: an escaped reference could let any later
    call mutate the container.

    Caller must separately verify the module-wide gate: no dynamic
    constructs and no binding of LEN_GATE_NAMES anywhere.
    """
    counts: "Counter[str]" = Counter()
    for n in iter_region(scope):
        counts.update(binding_names(n))
    for p in _own_params(scope):
        counts[p] += 1

    fresh: set[str] = set()
    for n in iter_region(scope):
        if not (
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
        ):
            continue
        name = n.targets[0].id
        if counts[name] != 1:
            continue
        value = n.value
        if isinstance(value, _FRESH_VALUE_TYPES):
            fresh.add(name)
        elif (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in BUILTIN_CONTAINER_CALLS
        ):
            fresh.add(name)
    if not fresh:
        return frozenset()

    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(scope):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent

    disqualified: set[str] = set()
    for n in ast.walk(scope):
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            # A nested closure declaring the name global/nonlocal can rebind
            # it on any call (the region-level counts never see that Store).
            disqualified.update(n.names)
            continue
        if not (isinstance(n, ast.Name) and n.id in fresh):
            continue
        if isinstance(n.ctx, (ast.Store, ast.Del)):
            continue  # the single binding (dels/rebinds already failed counts)
        parent = parents.get(id(n))
        allowed = False
        if (
            isinstance(parent, ast.Call)
            and isinstance(parent.func, ast.Name)
            and parent.func.id == "len"
            and list(parent.args) == [n]
            and not parent.keywords
        ):
            allowed = True
        elif (
            isinstance(parent, ast.Subscript)
            and parent.value is n
            and isinstance(parent.ctx, ast.Load)
        ):
            allowed = True
        elif isinstance(parent, (ast.For, ast.AsyncFor)) and parent.iter is n:
            allowed = True
        elif isinstance(parent, (ast.If, ast.While)) and parent.test is n:
            allowed = True
        if not allowed:
            disqualified.add(n.id)
    return frozenset(fresh - disqualified)


def for_range_binding(node: ast.AST):
    """``(target_name, range_args)`` for ``for <name> in range(...):`` loops
    (1-3 plain positional args), else None."""
    if (
        isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "range"
        and 1 <= len(node.iter.args) <= 3
        and not node.iter.keywords
        and not any(isinstance(a, ast.Starred) for a in node.iter.args)
    ):
        return node.target.id, list(node.iter.args)
    return None


def builtin_gate(tree: ast.Module, name: str) -> bool:
    """Module-wide precondition for trusting builtin *name*: no dynamic
    constructs and the name never bound anywhere in the tree."""
    return not tree_has_dynamic(tree) and name not in all_bound_names(tree)


def is_int_expr(
    node: ast.AST,
    proven: frozenset[str] | set[str],
    containers: frozenset[str] | set[str] = frozenset(),
) -> bool:
    """True if *node* provably evaluates to a plain ``int`` (never ``bool``),
    given that every name in *proven* is a plain int and every name in
    *containers* is a stable fresh builtin container (``len()`` of one is an
    int)."""
    if isinstance(node, ast.Constant):
        return type(node.value) is int
    if isinstance(node, ast.Name):
        return isinstance(node.ctx, ast.Load) and node.id in proven
    if containers and is_len_call(node, containers):
        return True
    if isinstance(node, ast.BinOp):
        return (
            isinstance(node.op, INT_BIN_OPS)
            and is_int_expr(node.left, proven, containers)
            and is_int_expr(node.right, proven, containers)
        )
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.op, INT_UNARY_OPS) and is_int_expr(
            node.operand, proven, containers
        )
    return False


def infer_int_names(
    scope: ast.AST,
    containers: frozenset[str] | set[str] = frozenset(),
    range_ok: bool = False,
) -> frozenset[str]:
    """Names local to *scope* whose every binding provably produces a plain
    ``int`` (``len()`` of a name in *containers* counts as int).

    With *range_ok* (caller must have verified :func:`builtin_gate` for
    ``"range"``) a ``for i in range(...):`` target is proven int with no
    constraint on the argument expressions: a successful builtin ``range()``
    call yields plain ints whatever its arguments are (bool/``__index__``
    inputs included), and a bad argument raises before the target is ever
    bound.
    """
    bindings: dict[str, list[ast.AST]] = {}
    disqualified: set[str] = set(_own_params(scope))
    recorded: set[int] = set()  # id() of Name nodes that are handled targets

    for node in iter_region(scope):
        if range_ok and (rng := for_range_binding(node)) is not None:
            recorded.add(id(node.target))
            bindings.setdefault(rng[0], [])  # int regardless of the args
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    recorded.add(id(target))
                    bindings.setdefault(target.id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.value is not None:
                recorded.add(id(node.target))
                bindings.setdefault(node.target.id, []).append(node.value)
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name):
                recorded.add(id(node.target))
                if isinstance(node.op, INT_BIN_OPS):
                    bindings.setdefault(node.target.id, []).append(node.value)
                else:
                    disqualified.add(node.target.id)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)) and id(node) not in recorded:
                # for/with/except/walrus/comprehension targets, tuple
                # unpacking, del, ... -- all disqualify.
                disqualified.add(node.id)
        else:
            disqualified.update(binding_names(node))

    # Greatest fixpoint: assume every candidate is int, then discard any name
    # with a binding that is not int-valued under the current assumption.
    # Sound also for self-referential accumulators (``acc = acc + i``): a
    # surviving name can only ever be assigned from int-closed operations
    # over surviving names (reading a not-yet-bound name raises instead of
    # producing a value), so by induction over completed assignments it never
    # holds a non-int.
    proven: set[str] = {n for n in bindings if n not in disqualified}
    while True:
        removed = False
        for name in list(proven):
            if not all(is_int_expr(v, proven, containers) for v in bindings[name]):
                proven.discard(name)
                removed = True
        if not removed:
            break
    return frozenset(proven)


# -- interval (value-range) analysis ----------------------------------------
#
# Flow-insensitive: a name's interval is the join over the intervals of all
# its bindings, iterated to a fixpoint from bottom with widening (a bound
# still moving after a few sweeps goes to +-inf).  Bounds are exact Python
# ints; infinities are the only floats, so no precision is ever lost on the
# finite side.  Everything over-approximates: consumers may only rely on
# "value is certainly inside [lo, hi]".

NEG_INF = float("-inf")
POS_INF = float("inf")
_TOP = (NEG_INF, POS_INF)

_RANGE_MAX_SHIFT = 256  # mirror _HOIST_MAX_SHIFT: never build giant literals


def _mul_bound(a, b):
    if a == 0 or b == 0:
        return 0
    if isinstance(a, float) or isinstance(b, float):
        return POS_INF if (a > 0) == (b > 0) else NEG_INF
    return a * b


def _hull(values):
    values = list(values)
    return (min(values), max(values))


def interval_join(x, y):
    return (min(x[0], y[0]), max(x[1], y[1]))


def _binop_range(op, left, right):
    """Interval of ``left <op> right`` (both operands proven int).  Returns
    None (bottom) when either side is bottom."""
    if left is None or right is None:
        return None
    l0, l1 = left
    r0, r1 = right
    if isinstance(op, ast.Add):
        return (l0 + r0, l1 + r1)
    if isinstance(op, ast.Sub):
        return (l0 - r1, l1 - r0)
    if isinstance(op, ast.Mult):
        return _hull(_mul_bound(a, b) for a in (l0, l1) for b in (r0, r1))
    if isinstance(op, ast.FloorDiv):
        # Exact only for finite divisor intervals that exclude 0.
        if isinstance(r0, float) or isinstance(r1, float):
            return _TOP
        if r0 >= 1 or r1 <= -1:
            candidates = []
            for a in (l0, l1):
                for d in (r0, r1):
                    if isinstance(a, float):
                        candidates.append(a if d > 0 else -a)
                    else:
                        candidates.append(a // d)
            return _hull(candidates)
        return _TOP
    if isinstance(op, ast.Mod):
        if r0 >= 1:
            return (0, r1 - 1)
        if r1 <= -1:
            return (r0 + 1, 0)
        return _TOP
    if isinstance(op, ast.LShift):
        if r0 >= 0 and not isinstance(r1, float) and r1 <= _RANGE_MAX_SHIFT:
            return _hull(
                _mul_bound(a, 2 ** d) for a in (l0, l1) for d in (r0, r1)
            )
        return _TOP
    if isinstance(op, ast.RShift):
        if r0 >= 0 and not isinstance(r1, float) and r1 <= _RANGE_MAX_SHIFT:
            candidates = []
            for a in (l0, l1):
                for d in (r0, r1):
                    if isinstance(a, float):
                        candidates.append(a)
                    else:
                        candidates.append(a >> d)
            return _hull(candidates)
        return _TOP
    if isinstance(op, ast.BitAnd):
        # x & m for m >= 0 lies in [0, m] whatever x is (and vice versa).
        if l0 >= 0 and r0 >= 0:
            return (0, min(l1, r1))
        if r0 >= 0:
            return (0, r1)
        if l0 >= 0:
            return (0, l1)
        return _TOP
    if isinstance(op, (ast.BitOr, ast.BitXor)):
        if l0 >= 0 and r0 >= 0:
            if isinstance(l1, float) or isinstance(r1, float):
                return (0, POS_INF)
            bits = max(int(l1).bit_length(), int(r1).bit_length())
            return (0, (1 << bits) - 1)
        return _TOP
    return _TOP


def int_expr_range(node, ranges, containers=frozenset()):
    """Interval of a proven-int expression under *ranges* (name -> interval).

    Returns None (bottom) when the expression reads a name that has no
    completed binding yet; unknown shapes yield the unbounded interval.
    Sound only for expressions that :func:`is_int_expr` accepts (callers
    check that first); anything else simply comes back unbounded.
    """
    if isinstance(node, ast.Constant):
        if type(node.value) is int:
            return (node.value, node.value)
        return _TOP
    if isinstance(node, ast.Name):
        if node.id in ranges:
            return ranges[node.id]  # may be None (bottom)
        return _TOP
    if is_len_call(node, containers):
        return (0, POS_INF)
    if isinstance(node, ast.UnaryOp):
        r = int_expr_range(node.operand, ranges, containers)
        if r is None:
            return None
        if isinstance(node.op, ast.UAdd):
            return r
        if isinstance(node.op, ast.USub):
            return (-r[1], -r[0])
        if isinstance(node.op, ast.Invert):
            return (-r[1] - 1, -r[0] - 1)
        return _TOP
    if isinstance(node, ast.BinOp):
        left = int_expr_range(node.left, ranges, containers)
        right = int_expr_range(node.right, ranges, containers)
        return _binop_range(node.op, left, right)
    return _TOP


def infer_int_ranges(
    scope: ast.AST,
    proven: frozenset[str] | set[str],
    containers: frozenset[str] | set[str] = frozenset(),
    range_ok: bool = False,
) -> dict[str, tuple | None]:
    """Interval for every name in *proven* (which must come from
    :func:`infer_int_names` on the same scope with the same options).

    A None entry means no completed binding was found (e.g. a name only ever
    augmented, which can never bind successfully at runtime) -- consumers
    must treat it as unknown.
    """
    if not proven:
        return {}
    bindings: dict[str, list[tuple]] = {n: [] for n in proven}
    for node in iter_region(scope):
        if range_ok and (rng := for_range_binding(node)) is not None:
            if rng[0] in bindings:
                bindings[rng[0]].append(("range", rng[1]))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in bindings:
                    bindings[t.id].append(("expr", node.value))
        elif isinstance(node, ast.AnnAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.value is not None
                and node.target.id in bindings
            ):
                bindings[node.target.id].append(("expr", node.value))
        elif isinstance(node, ast.AugAssign):
            if isinstance(node.target, ast.Name) and node.target.id in bindings:
                bindings[node.target.id].append(("aug", node.op, node.value))

    ranges: dict[str, tuple | None] = {n: None for n in proven}
    sweeps = 0
    while True:
        sweeps += 1
        changed = False
        for name, entries in bindings.items():
            parts = []
            for entry in entries:
                if entry[0] == "expr":
                    r = int_expr_range(entry[1], ranges, containers)
                elif entry[0] == "aug":
                    current = ranges[name]
                    value = int_expr_range(entry[2], ranges, containers)
                    r = _binop_range(entry[1], current, value)
                else:  # "range"
                    args = [
                        int_expr_range(a, ranges, containers) for a in entry[1]
                    ]
                    if any(a is None for a in args):
                        r = None
                    elif len(args) == 1:
                        hi = args[0][1] - 1
                        # range(n) with n <= 0 yields nothing: no binding.
                        r = None if hi < 0 else (0, hi)
                    else:
                        # Values of range(a, b[, step]) always lie within the
                        # hull of the two endpoint intervals.
                        r = interval_join(args[0], args[1])
                if r is not None:
                    parts.append(r)
            if not parts:
                continue
            new = parts[0]
            for p in parts[1:]:
                new = interval_join(new, p)
            old = ranges[name]
            if old is not None:
                new = interval_join(new, old)
                if new != old and sweeps > 4:
                    # Widening: a bound still moving after 4 sweeps goes to
                    # infinity, which guarantees termination.
                    new = (
                        NEG_INF if new[0] < old[0] else new[0],
                        POS_INF if new[1] > old[1] else new[1],
                    )
            if new != old:
                ranges[name] = new
                changed = True
        if not changed:
            return ranges
