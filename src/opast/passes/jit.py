"""Opt-in numba JIT injection (``--jit``).

One-shot pass executed once after the optimization fixpoint (never inside
the iteration loop).  Module-level functions that were not inlined are
decorated with :func:`opast.jitsupport.maybe_njit` when they are

* **hot** by a static heuristic: contain nested loops, or a loop whose trip
  count is statically ≥ 10_000 (``for ... in range(<consts>)``, or
  ``while name < <const>``) -- compilation costs ~0.1-0.5s per function, so
  cold functions are not worth it;
* **plausibly numba-compatible** by a strict whitelist: plain positional
  parameters (numeric constant defaults only), numeric constants, int/float
  arithmetic, tuples (packing, unpacking, multiple returns, constant-index
  reads -- numba's static_getitem types those even on heterogeneous
  tuples), ``if``/``while``/``for``-over-``range``/``return``, calls
  limited to ``range/abs/min/max/round/divmod/int/float/bool/len``,
  ``math.*`` and other candidate functions (see below), names limited to
  parameters, locals, and proven single-binding numeric/tuple module
  constants (numba freezes globals at compile time; the single binding
  makes the frozen value exact -- see :func:`frozen_const_globals`).  No
  strings, lists/dicts/sets (reflected containers copy per call and are
  deprecated), attributes beyond ``math``, variable-index or store
  subscripts, closures, try/with/yield/async;
* bound exactly once at module level, never ``global``-declared, and the
  module contains no dynamic constructs.

Lazy candidates (runtime-decided hotness)
-----------------------------------------

Whitelisted functions that are *not* statically hot but contain a loop
whose trip count only runtime knows (``for i in range(n)``,
``while i < n``) get :func:`opast.jitsupport.maybe_njit_lazy` instead:
plain Python plus observation until a runtime trigger (bound-argument
size, single-call time, or call volume) proves the function hot, then the
usual compile + verify + guarded-dispatch machinery takes over.  Loop
bounds that name a positional parameter are passed to the decorator as
argument indices, enabling the immediate size trigger.  A lazy candidate
with peer calls compiles from a ``_opast_jitsrc`` copy whose calls target
raw-dispatcher aliases; the aliases are *backfilled at trigger time*
(already-compiled peers resolve via ``compiled_of``, not-yet-compiled
ones -- lazy wrappers and plain undecorated candidates alike -- are
compiled on the spot; any peer that cannot compile makes the caller
permanently Python).

Argument-mutating candidates (a subscript store on an annotated array
parameter, or such a parameter passed on to a peer) compile with
``verify=False``: the first-call comparison re-runs the function on the
same arguments, which is only sound for argument-pure code -- verifying
an in-place partition would re-partition around a different pivot and
hand the caller stale boundaries.  They rely on the whitelist and
numba's typing alone.

njit inter-calls
----------------

Candidates may call each other: candidacy is a fixpoint (calling anything
outside the shrinking candidate set disqualifies) with call cycles dropped
(numba's recursion support is unreliable).  Hot candidates seed the
selection and pull in their callees.  A caller cannot be compiled from its
original body -- at runtime the callee's global name holds the fallback
*wrapper*, which numba cannot type -- so callers get a duplicated
``_opast_jitsrc_F`` def whose peer calls are rewritten to ``_opast_njit_G``
raw-dispatcher aliases (``jitsupport.compiled_of``); the original def stays
as the Python fallback, whose calls hit the wrappers and remain guarded.
A callee that fails to compile leaves its alias None, and every compiled
caller then degrades to its own Python fallback at first call.

The whitelist only predicts *compilation success*; correctness under all
argument types is guaranteed by the runtime dispatcher's permanent Python
fallback (see :mod:`opast.jitsupport`).  The int64-wraparound caveat is
inherent to opting in and documented in the README.

Loop outlining
--------------

Hot numeric code rarely arrives pre-packaged as a compatible function, so a
second stage *outlines* hot whitelisted loops -- at module top level and
inside functions that were not selected above -- into fresh module-level
functions ``_opast_jit_loop_N`` decorated the same way, replacing the loop
with ``out1, out2 = _opast_jit_loop_N(in1, in2)``.

* **Inputs** are the loop's names that are *definitely bound* before the
  loop (same straight-line dominance scan as LICM) plus names never bound in
  the enclosing scope (outer/builtin reads, captured once at the call --
  nothing can rebind them mid-loop because the whitelist forbids calls to
  user code).  A read of a name only *conditionally* bound in the scope is
  rejected: making it a function local would turn a working read into
  ``UnboundLocalError``.
* **Outputs** are the loop's stores that are observable afterwards (any use
  elsewhere in the scope; at module level every global is an observable
  surface, so all of them).  An output must be an input (so the zero-
  iteration call returns the pre-loop value unchanged) or provably stored on
  every run: the ``for`` target of a constant ``range`` with >=1 trips, or
  an unconditional per-iteration assignment when the trip count is >=1 and
  the loop has no ``break``/``continue``.
* Loop-local temporaries need a store-before-load order proof (per
  iteration, ``for`` targets count as stored at body entry).
* Rejected outright: loops containing ``return``, loops with ``orelse``,
  loops under ``try``/``with`` (an exception escaping mid-loop would leave
  the write-back unexecuted, so a handler could observe pre-loop values --
  outside ``try``/``with`` the scope's locals are unobservable after an
  escaping exception, frame introspection being dynamic-gated), and loops
  touching names ``global``/``nonlocal``-declared in the scope.
* The assembled function must pass the same ``_is_hot`` and
  ``_numba_compatible`` gates as whole functions; the runtime dispatcher's
  permanent Python fallback again covers every mistyped call.

Remaining opt-in caveats (beyond int64 wraparound): the write-back happens
once after the loop instead of per iteration, and an exception raised
mid-loop propagates with the enclosing scope's variables still holding their
pre-loop values.  Both are unobservable to single-threaded code that cannot
run during the loop (no user calls inside) and outside ``try``/``with``.
"""

from __future__ import annotations

import ast
import copy
from collections import Counter

from ..analysis import all_bound_names, binding_names
from ..safety import iter_region, tree_has_dynamic
from .licm import definite_bindings, unbound_risk_names

_HELPER_NAME = "_opast_jit"
_OUTLINE_PREFIX = "_opast_jit_loop"
_HOT_TRIP = 10_000

_ALLOWED_CALLS = frozenset(
    {"range", "abs", "min", "max", "round", "divmod", "int", "float", "bool",
     "len"}
)

#: numpy functions numba's nopython mode supports well; attribute access on
#: the proven numpy root is limited to these (Load), calls included.
_NP_FUNCS = frozenset({
    "abs", "arange", "array", "asarray", "ceil", "copy", "cos", "dot",
    "empty", "empty_like", "exp", "floor", "full", "full_like", "log",
    "maximum", "mean", "minimum", "ones", "ones_like", "prod", "sin",
    "sqrt", "sum", "tan", "zeros", "zeros_like",
})

#: The subset that provably yields an ndarray -- locals bound once from one
#: of these calls may be subscript-stored (``buf[i] = x``).
_NP_CTORS = frozenset({
    "arange", "array", "asarray", "copy", "empty", "empty_like", "full",
    "full_like", "ones", "ones_like", "zeros", "zeros_like",
})

#: Array attributes numba types natively (reads only).
_ARRAY_ATTRS = frozenset({"size", "shape", "ndim"})

#: Builtin exceptions a candidate may ``raise`` with constant arguments
#: (numba lowers those; dynamic messages like f-strings do not type).
_EXC_NAMES = frozenset({
    "ArithmeticError", "AssertionError", "Exception", "IndexError",
    "KeyError", "LookupError", "NotImplementedError", "OverflowError",
    "RuntimeError", "TypeError", "ValueError", "ZeroDivisionError",
})

_ALLOWED_STMTS = (
    ast.Return,
    ast.Assign,
    ast.AugAssign,
    ast.If,
    ast.While,
    ast.For,
    ast.Break,
    ast.Continue,
    ast.Pass,
    ast.Raise,  # constant-argument builtin exceptions only (checked below)
    ast.Expr,  # only a DynArray growth call (checked below); a bare call to
    # anything else would be an effect the whitelist does not model
)

_ALLOWED_EXPRS = (
    ast.BoolOp,
    ast.BinOp,
    ast.UnaryOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.Tuple,
    ast.Name,
    ast.Constant,
    ast.Attribute,
    ast.Subscript,
    ast.Slice,  # basic array slicing; gated with subscripts below
)


def _const_int_index(node: ast.expr) -> bool:
    """A literal (possibly negated) int index -- numba's static_getitem
    handles it on heterogeneous tuples, where a runtime index would not
    type."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    return isinstance(node, ast.Constant) and type(node.value) is int


def frozen_const_globals(tree: ast.Module) -> frozenset[str]:
    """Module-level names safe to leave as global reads in a candidate:
    bound exactly once at module top level to a numeric/bool literal (or a
    tuple of those), and never ``global``-declared.  numba freezes globals
    to their compile-time values; the single proven binding makes that
    value the only one the name can ever hold, so freezing is exact.
    """
    counts: Counter[str] = Counter()
    for n in iter_region(tree):
        counts.update(binding_names(n))
    declared: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Global):
            declared.update(n.names)
    consts: set[str] = set()
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and counts.get(stmt.targets[0].id, 0) == 1
            and stmt.targets[0].id not in declared
            and _frozen_value(stmt.value)
        ):
            consts.add(stmt.targets[0].id)
    return frozenset(consts)


def _frozen_value(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return type(node.value) in (int, float, bool, complex)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        return _frozen_value(node.operand)
    if isinstance(node, ast.Tuple) and isinstance(node.ctx, ast.Load):
        return all(_frozen_value(e) for e in node.elts)
    return False

_ALLOWED_CONST_TYPES = (int, float, bool, complex, type(None))

_REJECTED_OPS = (ast.MatMult, ast.In, ast.NotIn, ast.Is, ast.IsNot)

_LOOPS = (ast.While, ast.For)


def _is_range_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "range"
        and not node.keywords
        and 1 <= len(node.args) <= 3
    )


def _local_names(func: ast.FunctionDef) -> set[str]:
    names = set()
    for n in ast.walk(func):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)
    return names


class _ArrayAnnCtx:
    """What an ndarray *annotation* looks like in this module: ``np.ndarray``
    on the proven numpy root, ``npt.NDArray[...]`` on a single-binding
    ``import numpy.typing as npt``, names imported directly from
    numpy/numpy.typing, and single-binding module-level aliases thereof
    (``NumericArray = npt.NDArray[np.number]``, ``TypeAlias`` included).
    Only consulted under the aggressive ``annotations`` option -- trusting
    the annotation to reflect the runtime type is exactly that bet."""

    def __init__(self, tree: ast.Module, np_root: str | None,
                 module_counts) -> None:
        self.np_root = np_root
        self.npt_root = None
        self.direct: set[str] = set()
        for s in tree.body:
            if isinstance(s, ast.Import):
                for a in s.names:
                    if (
                        a.name == "numpy.typing"
                        and a.asname
                        and module_counts.get(a.asname, 0) == 1
                    ):
                        self.npt_root = a.asname
            elif isinstance(s, ast.ImportFrom) and s.module in (
                "numpy", "numpy.typing"
            ):
                for a in s.names:
                    if a.name in ("NDArray", "ndarray"):
                        bound = a.asname or a.name
                        if module_counts.get(bound, 0) == 1:
                            self.direct.add(bound)
        self.aliases: set[str] = set()
        changed = True
        while changed:  # aliases may reference earlier aliases
            changed = False
            for s in tree.body:
                target = value = None
                if (
                    isinstance(s, ast.Assign)
                    and len(s.targets) == 1
                    and isinstance(s.targets[0], ast.Name)
                ):
                    target, value = s.targets[0].id, s.value
                elif isinstance(s, ast.AnnAssign) and isinstance(
                    s.target, ast.Name
                ) and s.value is not None:
                    target, value = s.target.id, s.value
                if (
                    target is not None
                    and target not in self.aliases
                    and module_counts.get(target, 0) == 1
                    and self.matches(value)
                ):
                    self.aliases.add(target)
                    changed = True

    def matches(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Subscript):
            return self.matches(node.value)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == self.np_root and node.attr == "ndarray":
                return True
            return node.value.id == self.npt_root and node.attr == "NDArray"
        if isinstance(node, ast.Name):
            return node.id in self.direct or node.id in self.aliases
        return False


def _annotated_array_params(
    func: ast.FunctionDef, ctx: "_ArrayAnnCtx | None"
) -> frozenset[str]:
    """Parameters annotated as ndarrays and never re-bound inside the
    function (a re-bound name could hold anything by store time)."""
    if ctx is None:
        return frozenset()
    rebound = {
        n.id
        for n in ast.walk(func)
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))
    }
    out: set[str] = set()
    for a in [*func.args.posonlyargs, *func.args.args]:
        if (
            a.annotation is not None
            and a.arg not in rebound
            and ctx.matches(a.annotation)
        ):
            out.add(a.arg)
    return frozenset(out)


#: Growth methods only the dynamic form may use.
_DYN_METHODS = frozenset({"append", "pop"})


def dynarray_root(tree: ast.Module, module_counts) -> str | None:
    """The single-binding name ``opast.DynArray`` is imported under, if any."""
    for s in tree.body:
        if isinstance(s, ast.ImportFrom) and s.module in (
            "opast", "opast.containers"
        ):
            for a in s.names:
                if a.name == "DynArray":
                    bound = a.asname or "DynArray"
                    if module_counts.get(bound, 0) == 1:
                        return bound
    return None


def _dyn_ctor_kind(node: ast.AST, dyn_name: str) -> str | None:
    """``"dynamic"`` for ``DynArray()``, ``"fixed"`` for
    ``DynArray.zeros(...)``/``.full(...)``, else None."""
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if isinstance(func, ast.Name) and func.id == dyn_name:
        if not node.args and not node.keywords:
            return "dynamic"
        return None
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == dyn_name
        and func.attr in ("zeros", "full")
    ):
        return "fixed"
    return None


def dyn_locals(func: ast.FunctionDef, dyn_name: str):
    """``(kinds, ok)``: locals bound exactly once to a ``DynArray`` factory,
    mapped to ``"dynamic"``/``"fixed"``.

    *ok* is False when the function touches ``DynArray`` in any way this
    pass does not model -- an unrecognised factory call, a container that
    escapes (returned, passed on, aliased, stored into something else), or
    a growth method on a fixed-capacity buffer.  A rejected function is not
    a jit candidate at all: its compiled copy would have to keep calling
    ``DynArray``, which numba cannot type anyway.
    """
    counts: Counter[str] = Counter()
    for n in ast.walk(func):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            counts[n.id] += 1
        elif isinstance(n, ast.arg):
            counts[n.arg] += 1

    kinds: dict[str, str] = {}
    ctor_ids: set[int] = set()
    for n in ast.walk(func):
        if not (
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
        ):
            continue
        kind = _dyn_ctor_kind(n.value, dyn_name)
        if kind is None:
            continue
        name = n.targets[0].id
        if counts[name] != 1:
            return {}, False  # rebound: the later value is unknown
        kinds[name] = kind
        ctor_ids.add(id(n.value))

    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(func):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent

    for n in ast.walk(func):
        if not (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)):
            continue
        if n.id == dyn_name:
            # The factory name itself may only appear inside a recognised
            # constructor call bound to a trusted local.
            parent = parents.get(id(n))
            call = parent if isinstance(parent, ast.Call) else parents.get(
                id(parent)
            )
            if id(call) not in ctor_ids:
                return {}, False
            continue
        kind = kinds.get(n.id)
        if kind is None:
            continue
        parent = parents.get(id(n))
        if isinstance(parent, ast.Attribute):
            grand = parents.get(id(parent))
            if not (
                parent.attr in _DYN_METHODS
                and kind == "dynamic"
                and isinstance(grand, ast.Call)
                and grand.func is parent
                and not grand.keywords
            ):
                return {}, False
        elif isinstance(parent, ast.Subscript):
            if parent.value is not n:
                return {}, False
        elif isinstance(parent, ast.Call):
            if not (
                isinstance(parent.func, ast.Name)
                and parent.func.id == "len"
                and list(parent.args) == [n]
                and not parent.keywords
            ):
                return {}, False  # passed on: it would escape the function
        elif isinstance(parent, (ast.For, ast.AsyncFor)):
            if parent.iter is not n:
                return {}, False
        elif isinstance(parent, (ast.If, ast.While, ast.IfExp)):
            if parent.test is not n:
                return {}, False
        elif isinstance(parent, ast.BoolOp):
            pass  # ``while stack and ...``: a truth test, rewritten below
        elif isinstance(parent, ast.UnaryOp):
            if not isinstance(parent.op, ast.Not):
                return {}, False
        else:
            return {}, False  # returned, aliased, compared, ...
    return kinds, True


class _DynArrayRewriter(ast.NodeTransformer):
    """Replaces ``DynArray`` factories with the representation numba is
    fastest with -- applied to the *compiled copy* only, so the Python
    fallback keeps running the original list."""

    def __init__(self, dyn_name: str, np_root: str | None,
                 kinds: dict[str, str]) -> None:
        self.dyn_name = dyn_name
        self.np_root = np_root
        self.kinds = kinds

    def _truth(self, node: ast.expr) -> ast.expr:
        """``stack`` -> ``len(stack) > 0`` in test position: numba types the
        explicit comparison on both representations, while container
        truthiness is not something to bet a whole function's compilation
        on.  The Python fallback keeps the idiomatic form."""
        if isinstance(node, ast.Name) and node.id in self.kinds:
            return ast.copy_location(
                ast.Compare(
                    left=ast.Call(
                        func=ast.Name(id="len", ctx=ast.Load()),
                        args=[node],
                        keywords=[],
                    ),
                    ops=[ast.Gt()],
                    comparators=[ast.Constant(0)],
                ),
                node,
            )
        return node

    def _visit_test(self, node):
        node = self.generic_visit(node)
        node.test = self._truth(node.test)
        return node

    visit_While = _visit_test
    visit_If = _visit_test
    visit_IfExp = _visit_test

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        node = self.generic_visit(node)
        node.values = [self._truth(v) for v in node.values]
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Not):
            node.operand = self._truth(node.operand)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        kind = _dyn_ctor_kind(node, self.dyn_name)
        if kind is None:
            return self.generic_visit(node)
        if kind == "dynamic":
            # numba types a locally built list with no reflection.
            return ast.copy_location(ast.List(elts=[], ctx=ast.Load()), node)
        node = self.generic_visit(node)
        attr = node.func.attr
        size = node.args[0]
        if attr == "full":
            fill = node.args[1] if len(node.args) > 1 else next(
                (k.value for k in node.keywords if k.arg == "value"), None
            )
            if fill is None:
                return node
            if self.np_root is not None:
                return ast.copy_location(
                    _np_call(self.np_root, "full", [size, fill]), node
                )
            return ast.copy_location(_filled_list(fill, size), node)
        dtype = node.args[1] if len(node.args) > 1 else next(
            (k.value for k in node.keywords if k.arg == "dtype"), None
        )
        is_int = isinstance(dtype, ast.Name) and dtype.id == "int"
        if self.np_root is not None:
            args = [size]
            if is_int:
                args.append(
                    ast.Attribute(
                        value=ast.Name(id=self.np_root, ctx=ast.Load()),
                        attr="int64",
                        ctx=ast.Load(),
                    )
                )
            return ast.copy_location(_np_call(self.np_root, "zeros", args), node)
        zero = ast.Constant(0 if is_int else 0.0)
        return ast.copy_location(_filled_list(zero, size), node)


def _np_call(root: str, attr: str, args: list) -> ast.Call:
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id=root, ctx=ast.Load()), attr=attr, ctx=ast.Load()
        ),
        args=args,
        keywords=[],
    )


def _filled_list(value: ast.expr, size: ast.expr) -> ast.ListComp:
    """``[value for _opast_i in range(size)]`` -- the numpy-free fallback
    representation for a fixed-capacity buffer (numba builds a typed list
    from a comprehension; list multiplication it does not support)."""
    return ast.ListComp(
        elt=value,
        generators=[
            ast.comprehension(
                target=ast.Name(id="_opast_fill_i", ctx=ast.Store()),
                iter=ast.Call(
                    func=ast.Name(id="range", ctx=ast.Load()),
                    args=[size],
                    keywords=[],
                ),
                ifs=[],
                is_async=0,
            )
        ],
    )


def _mutates_annotated_params(
    func: ast.FunctionDef, ctx: "_ArrayAnnCtx | None",
    peers: frozenset[str] = frozenset(),
) -> bool:
    """True when *func* visibly mutates caller-owned arrays: a subscript
    store on an annotated array parameter, or such a parameter passed on to
    a peer (which may mutate it in turn).  First-call verification re-runs
    the function on the same arguments and is only sound for argument-pure
    functions, so these candidates compile unverified."""
    arrays = _annotated_array_params(func, ctx)
    if not arrays:
        return False
    for n in ast.walk(func):
        if (
            isinstance(n, ast.Subscript)
            and isinstance(n.ctx, (ast.Store, ast.Del))
            and isinstance(n.value, ast.Name)
            and n.value.id in arrays
        ):
            return True
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id in peers
            and any(
                isinstance(a, ast.Name) and a.id in arrays for a in n.args
            )
        ):
            return True
    return False


def _np_locals(func: ast.FunctionDef, np_root: str) -> frozenset[str]:
    """Locals bound exactly once, from a ``np.<ctor>(...)`` call: provably
    ndarrays, so subscript stores on them are in-bounds-checked array
    writes (an IndexError is identical in both paths)."""
    counts: Counter[str] = Counter()
    for n in ast.walk(func):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            counts[n.id] += 1
        elif isinstance(n, ast.arg):
            counts[n.arg] += 1
    out: set[str] = set()
    for n in ast.walk(func):
        if (
            isinstance(n, ast.Assign)
            and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name)
            and counts[n.targets[0].id] == 1
            and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute)
            and isinstance(n.value.func.value, ast.Name)
            and n.value.func.value.id == np_root
            and n.value.func.attr in _NP_CTORS
        ):
            out.add(n.targets[0].id)
    return frozenset(out)


def _numba_compatible(
    func: ast.FunctionDef,
    math_ok: bool,
    peers: frozenset[str] = frozenset(),
    consts: frozenset[str] = frozenset(),
    np_root: str | None = None,
    ann_ctx: "_ArrayAnnCtx | None" = None,
    dyn_name: str | None = None,
) -> bool:
    """*peers* are other jit candidates that may be called (call position
    only -- a bare load of a peer would hand numba the fallback wrapper,
    which it cannot type).  *consts* are proven single-binding module
    constants (see :func:`frozen_const_globals`) numba may freeze.
    *np_root* is the proven single-binding numpy import alias, unlocking
    ``np.<whitelisted>`` calls, variable-index subscript reads, and
    subscript stores on locals provably bound to ``np`` constructors."""
    args = func.args
    if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs:
        return False
    for default in args.defaults:
        if not (
            isinstance(default, ast.Constant)
            and type(default.value) in (int, float, bool)
        ):
            return False

    own_names = {a.arg for a in args.args} | _local_names(func)
    allowed_names = own_names | _ALLOWED_CALLS | consts
    if math_ok:
        allowed_names.add("math")
    np_arrays: frozenset[str] = frozenset()
    if np_root is not None and np_root not in own_names:
        allowed_names.add(np_root)
        np_arrays = _np_locals(func, np_root)
    else:
        np_root = None  # a local shadowing the alias disables numpy trust
    # Aggressive ``annotations``: NDArray-annotated, never-rebound
    # parameters count as proven arrays (stores, .size/.shape, slices).
    np_arrays = np_arrays | _annotated_array_params(func, ann_ctx)
    # ``DynArray`` locals: the marker's contract (see opast.containers) is
    # what makes the representation swap in the compiled copy legitimate.
    dyn_kinds: dict[str, str] = {}
    if dyn_name is not None and dyn_name not in own_names:
        dyn_kinds, dyn_ok = dyn_locals(func, dyn_name)
        if not dyn_ok:
            return False
        if dyn_kinds:
            allowed_names.add(dyn_name)

    body = list(func.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]  # docstring is fine for numba

    stack: list[ast.AST] = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.stmt):
            if not isinstance(node, _ALLOWED_STMTS):
                return False
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, (ast.Name, ast.Tuple)):
                        continue
                    if (
                        isinstance(t, ast.Subscript)
                        and isinstance(t.value, ast.Name)
                        and (t.value.id in np_arrays or t.value.id in dyn_kinds)
                    ):
                        continue
                    return False
            elif isinstance(node, ast.AugAssign):
                if not (
                    isinstance(node.target, ast.Name)
                    or (
                        isinstance(node.target, ast.Subscript)
                        and isinstance(node.target.value, ast.Name)
                        and (
                            node.target.value.id in np_arrays
                            or node.target.value.id in dyn_kinds
                        )
                    )
                ):
                    return False
            elif isinstance(node, ast.For):
                if node.orelse or not isinstance(node.target, ast.Name):
                    return False
                if not _is_range_call(node.iter) and not (
                    isinstance(node.iter, ast.Name)
                    and node.iter.id in dyn_kinds
                ):
                    return False
            elif isinstance(node, ast.While):
                if node.orelse:
                    return False
            elif isinstance(node, ast.Expr):
                call = node.value
                if not (
                    isinstance(call, ast.Call)
                    and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name)
                    and call.func.value.id in dyn_kinds
                    and call.func.attr in _DYN_METHODS
                ):
                    return False
            elif isinstance(node, ast.Raise):
                # Constant-argument builtin exceptions lower fine in numba;
                # anything dynamic (f-strings, custom classes) does not.
                exc = node.exc
                if node.cause is not None or exc is None:
                    return False
                if isinstance(exc, ast.Name):
                    if not (exc.id in _EXC_NAMES and exc.id not in own_names):
                        return False
                elif (
                    isinstance(exc, ast.Call)
                    and isinstance(exc.func, ast.Name)
                    and exc.func.id in _EXC_NAMES
                    and exc.func.id not in own_names
                    and not exc.keywords
                    and all(
                        isinstance(a, ast.Constant)
                        and type(a.value) in (str, int, float, bool)
                        for a in exc.args
                    )
                ):
                    pass
                else:
                    return False
                continue  # validated whole; the str constants must not
                # reach the generic Constant check
        elif isinstance(node, ast.expr):
            if not isinstance(node, _ALLOWED_EXPRS):
                return False
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Del):
                    return False
                if isinstance(node.ctx, ast.Load) and node.id not in allowed_names:
                    return False
            elif isinstance(node, ast.Constant):
                if type(node.value) not in _ALLOWED_CONST_TYPES:
                    return False
            elif isinstance(node, ast.Attribute):
                math_attr = (
                    math_ok
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "math"
                    and isinstance(node.ctx, ast.Load)
                )
                np_attr = (
                    np_root is not None
                    and isinstance(node.value, ast.Name)
                    and node.value.id == np_root
                    and node.attr in _NP_FUNCS
                    and isinstance(node.ctx, ast.Load)
                )
                array_attr = (
                    isinstance(node.value, ast.Name)
                    and node.value.id in np_arrays
                    and node.attr in _ARRAY_ATTRS
                    and isinstance(node.ctx, ast.Load)
                )
                dyn_attr = (
                    isinstance(node.value, ast.Name)
                    and (
                        node.value.id in dyn_kinds
                        and node.attr in _DYN_METHODS
                        or node.value.id == dyn_name
                        and node.attr in ("zeros", "full")
                    )
                    and isinstance(node.ctx, ast.Load)
                )
                if not (math_attr or np_attr or array_attr or dyn_attr):
                    return False
            elif isinstance(node, ast.Subscript):
                # Reads: constant index always (tuple element access via
                # static_getitem); any index once a proven numpy import is
                # in scope (arrays and homogeneous tuples type it; anything
                # else fails to compile and falls back).  Stores: proven
                # ndarray locals only.
                if isinstance(node.ctx, ast.Load):
                    if not (
                        np_root is not None
                        or np_arrays
                        or dyn_kinds
                        or _const_int_index(node.slice)
                    ):
                        return False
                elif not (
                    isinstance(node.value, ast.Name)
                    and (node.value.id in np_arrays or node.value.id in dyn_kinds)
                ):
                    return False
            elif isinstance(node, ast.Call):
                fn = node.func
                name_ok = isinstance(fn, ast.Name) and fn.id in _ALLOWED_CALLS
                # A local/param shadowing a peer makes every reference local
                # (whole-function scoping), so it is not a peer call -- and
                # must not be rewritten to the raw dispatcher.
                peer_call = (
                    isinstance(fn, ast.Name)
                    and fn.id in peers
                    and fn.id not in own_names
                )
                math_call = (
                    math_ok
                    and isinstance(fn, ast.Attribute)
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id == "math"
                )
                np_call = (
                    np_root is not None
                    and isinstance(fn, ast.Attribute)
                    and isinstance(fn.value, ast.Name)
                    and fn.value.id == np_root
                    and fn.attr in _NP_FUNCS
                )
                # ``x.append(v)``/``x.pop()`` on a growable DynArray local,
                # and the factory calls themselves (rewritten in the copy).
                dyn_call = (
                    isinstance(fn, ast.Attribute)
                    and isinstance(fn.value, ast.Name)
                    and (
                        fn.value.id in dyn_kinds
                        and fn.attr in _DYN_METHODS
                        or fn.value.id == dyn_name
                        and fn.attr in ("zeros", "full")
                    )
                ) or (
                    isinstance(fn, ast.Name)
                    and dyn_name is not None
                    and fn.id == dyn_name
                )
                if not (name_ok or math_call or np_call or peer_call or dyn_call):
                    return False
                if any(isinstance(a, ast.Starred) for a in node.args):
                    return False
                # ``DynArray.zeros(n, dtype=int)`` is the one keyword form
                # in the whitelist; it disappears in the compiled copy.
                if node.keywords and not dyn_call:
                    return False
                if peer_call:
                    # walk the arguments but not the callee Name itself
                    stack.extend(node.args)
                    continue
        elif isinstance(node, _REJECTED_OPS):
            return False
        stack.extend(ast.iter_child_nodes(node))
    return True


def _static_trip_count(loop: ast.AST) -> int | None:
    if isinstance(loop, ast.For) and _is_range_call(loop.iter):
        values = []
        for a in loop.iter.args:
            if isinstance(a, ast.Constant) and type(a.value) is int:
                values.append(a.value)
            else:
                return None
        try:
            return len(range(*values))
        except (TypeError, ValueError):
            return None
    if isinstance(loop, ast.While):
        test = loop.test
        if (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], (ast.Lt, ast.LtE))
            and isinstance(test.left, ast.Name)
            and isinstance(test.comparators[0], ast.Constant)
            and type(test.comparators[0].value) is int
        ):
            return test.comparators[0].value  # heuristic: counting from ~0
    return None


def _is_hot(func: ast.FunctionDef) -> bool:
    for node in ast.walk(func):
        if not isinstance(node, _LOOPS):
            continue
        if any(
            isinstance(inner, _LOOPS) and inner is not node
            for inner in ast.walk(node)
        ):
            return True  # nested loops
        trip = _static_trip_count(node)
        if trip is not None and trip >= _HOT_TRIP:
            return True
    return False


def _lazy_bound_args(func: ast.FunctionDef) -> tuple[bool, tuple[int, ...]]:
    """``(latent, bound_arg_indices)`` for the lazy-jit path.

    *latent* is True when the function has at least one loop whose trip
    count is statically unknown -- hotness can only be decided at runtime.
    (Loops with a *known* small trip count contribute nothing: those are
    provably cold, and known-hot ones were already taken by :func:`_is_hot`.)
    The indices name positional parameters that appear as a ``range()``
    argument or as the ``while name < bound`` comparator of such a loop, so
    the runtime wrapper can trigger compilation from the argument value
    alone; rebinding of the parameter inside the body only skews the
    heuristic, never correctness."""
    params = {a.arg: i for i, a in enumerate(func.args.args)}
    latent = False
    indices: set[int] = set()
    for node in ast.walk(func):
        if not isinstance(node, _LOOPS):
            continue
        if _static_trip_count(node) is not None:
            continue
        latent = True
        if isinstance(node, ast.For) and _is_range_call(node.iter):
            for a in node.iter.args:
                if isinstance(a, ast.Name) and a.id in params:
                    indices.add(params[a.id])
        elif isinstance(node, ast.While):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and len(test.ops) == 1
                and isinstance(test.ops[0], (ast.Lt, ast.LtE))
                and isinstance(test.comparators[0], ast.Name)
                and test.comparators[0].id in params
            ):
                indices.add(params[test.comparators[0].id])
    return latent, tuple(sorted(indices))


# -- njit inter-calls --------------------------------------------------------


def _peer_calls(func: ast.FunctionDef, peers: frozenset[str]) -> set[str]:
    """Peer functions *func* calls (shadowed names were already rejected by
    the compatibility check, so a plain name match is exact here)."""
    return {
        n.func.id
        for n in ast.walk(func)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id in peers
    }


def _acyclic_names(calls: dict[str, set[str]]) -> set[str]:
    """Names whose peer-call graph is cycle-free (resolvable callee-first).
    Self- and mutual recursion are dropped: numba's recursion support is
    unreliable, and a compile-time cycle would otherwise only surface as a
    runtime fallback of something selected as hot."""
    resolved: set[str] = set()
    changed = True
    while changed:
        changed = False
        for name, callees in calls.items():
            if name not in resolved and callees <= resolved:
                resolved.add(name)
                changed = True
    return resolved


class _PeerCallRewriter(ast.NodeTransformer):
    """Redirects peer calls to the raw njit dispatchers (``_opast_njit_*``)
    inside the *compiled* copy of a caller.  The original def keeps calling
    the wrappers, so the Python fallback path stays fully guarded."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self.mapping = mapping

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in self.mapping:
            node.func = ast.copy_location(
                ast.Name(id=self.mapping[node.func.id], ctx=ast.Load()),
                node.func,
            )
        return node


# -- loop outlining ----------------------------------------------------------


def _scope_params(scope: ast.AST) -> set[str]:
    if isinstance(scope, ast.Module):
        return set()
    a = scope.args
    extra = [x for x in (a.vararg, a.kwarg) if x is not None]
    return {x.arg for x in [*a.posonlyargs, *a.args, *a.kwonlyargs, *extra]}


def _use_counts(root: ast.AST) -> Counter:
    """Reads of a name's current value: loads, dels and augmented-assignment
    targets (``x += 1`` reads ``x`` even though its ctx is Store)."""
    counts: Counter = Counter()
    for n in ast.walk(root):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Load, ast.Del)):
            counts[n.id] += 1
        elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name):
            counts[n.target.id] += 1
    return counts


def _bind_counts(root: ast.AST) -> Counter:
    """Bindings anywhere in *root*'s subtree, nested scopes included."""
    counts: Counter = Counter()
    for n in ast.walk(root):
        counts.update(binding_names(n))
        if isinstance(n, ast.arg):
            counts[n.arg] += 1
    return counts


def _scope_bind_counts(scope: ast.AST) -> Counter:
    """Bindings of *scope*'s own variables: its region (nested scopes' locals
    are different variables) plus every ``global``/``nonlocal`` declaration in
    the subtree (a nested function may rebind this scope's name when called)."""
    counts: Counter = Counter()
    for n in iter_region(scope):
        counts.update(binding_names(n))
    for p in _scope_params(scope):
        counts[p] += 1
    for n in ast.walk(scope):
        if isinstance(n, (ast.Global, ast.Nonlocal)):
            counts.update(n.names)
    return counts


def _expr_uses(node: ast.AST) -> set[str]:
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _check_block_order(stmts, pending: set[str]) -> set[str] | None:
    """Forward scan proving every read of a name in *pending* is preceded, in
    guaranteed per-iteration order, by a store.  Returns the names still not
    definitely stored after the block, or None when a read may come first.
    Sound under ``break``/``continue``: a jump only *skips* later statements,
    and the store-before-load requirement is checked in body order, which any
    later iteration re-enters from the top."""
    pending = set(pending)
    for s in stmts:
        if isinstance(s, ast.Assign):
            if _expr_uses(s.value) & pending:
                return None
            for t in s.targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                        pending.discard(n.id)
        elif isinstance(s, ast.AugAssign):
            if isinstance(s.target, ast.Name) and s.target.id in pending:
                return None
            if (_expr_uses(s.value) | _expr_uses(s.target)) & pending:
                return None
        elif isinstance(s, ast.If):
            if _expr_uses(s.test) & pending:
                return None
            a = _check_block_order(s.body, pending)
            b = _check_block_order(s.orelse, pending)
            if a is None or b is None:
                return None
            pending = a | b  # stored only if stored on *both* branches
        elif isinstance(s, ast.While):
            if _expr_uses(s.test) & pending:
                return None
            if _check_block_order(s.body, pending) is None:
                return None
            # zero iterations possible: no removals survive the loop
        elif isinstance(s, ast.For):
            if _expr_uses(s.iter) & pending:
                return None
            inner = set(pending)
            if isinstance(s.target, ast.Name):
                inner.discard(s.target.id)  # stored at body entry
            if _check_block_order(s.body, inner) is None:
                return None
        elif isinstance(s, (ast.Break, ast.Continue, ast.Pass)):
            pass
        else:  # anything else fails the numba whitelist anyway
            return None
    return pending


def _definite_iter_stores(stmts) -> set[str]:
    """Names stored on every path through one body execution."""
    got: set[str] = set()
    for s in stmts:
        got |= definite_bindings(s)
        if isinstance(s, ast.If):
            got |= _definite_iter_stores(s.body) & _definite_iter_stores(s.orelse)
    return got


class _LoopOutliner:
    """Extracts hot whitelisted loops into fresh module-level functions
    decorated with ``maybe_njit`` (see module docstring)."""

    def __init__(self, owner, tree: ast.Module, math_ok: bool,
                 helper: str, taken: set[str],
                 np_root: str | None = None) -> None:
        self.owner = owner
        self.tree = tree
        self.math_ok = math_ok
        self.np_root = np_root
        self.helper = helper
        self.taken = taken
        self.tree_binds = _bind_counts(tree)
        self.defs: list[ast.FunctionDef] = []
        self._counter = 0

    def outline(self, skip_ids: set[int]) -> list[ast.FunctionDef]:
        scopes: list[ast.AST] = [self.tree]
        for node in ast.walk(self.tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and id(node) not in skip_ids
            ):
                scopes.append(node)
        for scope in scopes:
            self._do_scope(scope)
        return self.defs

    def _do_scope(self, scope: ast.AST) -> None:
        self._is_module = isinstance(scope, ast.Module)
        self._scope_binds = _scope_bind_counts(scope)
        self._scope_uses = _use_counts(scope)
        self._region_globals = {
            name
            for node in iter_region(scope)
            if isinstance(node, (ast.Global, ast.Nonlocal))
            for name in node.names
        }
        scope.body = self._walk_block(scope.body, _scope_params(scope))

    def _walk_block(self, stmts, bound: set[str]) -> list:
        bound = set(bound)
        out = []
        for stmt in stmts:
            if isinstance(stmt, (ast.For, ast.While)):
                replacement = self._try_outline(stmt, bound)
                if replacement is not None:
                    out.append(replacement)
                    bound |= definite_bindings(replacement)
                    continue
                risk = unbound_risk_names(stmt)
                inner = bound - risk
                body_bound = inner
                if (
                    isinstance(stmt, ast.For)
                    and isinstance(stmt.target, ast.Name)
                    and stmt.target.id not in risk
                ):
                    body_bound = inner | {stmt.target.id}
                stmt.body = self._walk_block(stmt.body, body_bound)
                stmt.orelse = self._walk_block(stmt.orelse, inner)
            elif isinstance(stmt, ast.If):
                stmt.body = self._walk_block(stmt.body, bound)
                stmt.orelse = self._walk_block(stmt.orelse, bound)
            # Try/With/Match subtrees are never descended into: an exception
            # (or a suppressing context manager) could observe the enclosing
            # scope mid-flight, which outlining's single write-back breaks.
            out.append(stmt)
            bound |= definite_bindings(stmt)
            bound -= unbound_risk_names(stmt)
        return out

    def _try_outline(self, loop, bound: set[str]):
        if loop.orelse or not _is_hot(loop):
            return None
        if any(isinstance(n, ast.Return) for n in ast.walk(loop)):
            return None

        loop_binds = _bind_counts(loop)
        loop_uses = _use_counts(loop)
        names = set(loop_binds) | set(loop_uses)
        if names & self._region_globals:
            return None

        # Whitelisted builtins (and a vetted ``math``) stay global references
        # inside the outlined function; shadowed ones fall through to normal
        # classification, where the runtime fallback keeps them correct.
        special: set[str] = set()
        for n in names:
            if n == "math":
                if self.math_ok and self.tree_binds["math"] == 1:
                    special.add(n)
            elif n == self.np_root:
                if self.tree_binds[n] == 1:
                    special.add(n)
            elif n in _ALLOWED_CALLS and self.tree_binds[n] == 0:
                special.add(n)

        provable = self._provable_stores(loop)
        params: list[str] = []
        outputs: list[str] = []
        needs_proof: set[str] = set()
        for n in sorted(names - special):
            stored = loop_binds[n] > 0
            loaded = loop_uses[n] > 0
            outside_bind = self._scope_binds[n] > loop_binds[n]
            live = self._is_module or self._scope_uses[n] > loop_uses[n]
            if n in bound:
                params.append(n)
                if stored and live:
                    outputs.append(n)
            elif stored:
                if loaded and outside_bind:
                    return None  # may read a conditionally pre-bound value
                if loaded:
                    needs_proof.add(n)
                if live:
                    if n not in provable:
                        return None  # write-back could be of an unbound name
                    outputs.append(n)
            elif outside_bind:
                return None  # conditionally bound local: UnboundLocal hazard
            else:
                params.append(n)  # outer-scope / builtin / NameError read
        if needs_proof and not self._order_ok(loop, needs_proof):
            return None

        fn = ast.parse("def _f():\n    pass").body[0]
        fn.name = self._fresh_name()
        fn.args.args = [ast.arg(arg=p) for p in params]
        fn.body = [loop]
        if outputs:
            if len(outputs) == 1:
                value = ast.Name(id=outputs[0], ctx=ast.Load())
            else:
                value = ast.Tuple(
                    elts=[ast.Name(id=o, ctx=ast.Load()) for o in outputs],
                    ctx=ast.Load(),
                )
            fn.body.append(ast.Return(value=value))
        fn.decorator_list = [
            ast.Attribute(
                value=ast.Name(id=self.helper, ctx=ast.Load()),
                attr="maybe_njit",
                ctx=ast.Load(),
            )
        ]
        ast.copy_location(fn, loop)
        if not _numba_compatible(fn, self.math_ok, np_root=self.np_root):
            return None

        call = ast.Call(
            func=ast.Name(id=fn.name, ctx=ast.Load()),
            args=[ast.Name(id=p, ctx=ast.Load()) for p in params],
            keywords=[],
        )
        if not outputs:
            replacement = ast.Expr(value=call)
        else:
            if len(outputs) == 1:
                target = ast.Name(id=outputs[0], ctx=ast.Store())
            else:
                target = ast.Tuple(
                    elts=[ast.Name(id=o, ctx=ast.Store()) for o in outputs],
                    ctx=ast.Store(),
                )
            replacement = ast.Assign(targets=[target], value=call)
        ast.copy_location(replacement, loop)
        self.defs.append(fn)
        self.owner.changes += 1
        return replacement

    @staticmethod
    def _provable_stores(loop) -> frozenset[str] | set[str]:
        """Names definitely stored by the loop on every run.  Only a ``for``
        over a constant ``range`` with >=1 trips proves anything (the While
        trip heuristic guesses the start value, so it proves nothing)."""
        if not (isinstance(loop, ast.For) and _static_trip_count(loop)):
            return frozenset()
        got: set[str] = set()
        if isinstance(loop.target, ast.Name):
            got.add(loop.target.id)  # stored before body, break-proof
        if not any(
            isinstance(n, (ast.Break, ast.Continue)) for n in ast.walk(loop)
        ):
            got |= _definite_iter_stores(loop.body)
        return got

    @staticmethod
    def _order_ok(loop, pending: set[str]) -> bool:
        if isinstance(loop, ast.While):
            if _expr_uses(loop.test) & pending:
                return False
            return _check_block_order(loop.body, pending) is not None
        if _expr_uses(loop.iter) & pending:
            return False
        inner = set(pending)
        if isinstance(loop.target, ast.Name):
            inner.discard(loop.target.id)
        return _check_block_order(loop.body, inner) is not None

    def _fresh_name(self) -> str:
        while True:
            name = f"{_OUTLINE_PREFIX}_{self._counter}"
            self._counter += 1
            if name not in self.taken:
                self.taken.add(name)
                return name


class JitInjection:
    """One-shot pass; not part of the fixpoint iteration."""

    name = "jit"

    def __init__(self) -> None:
        self.changes = 0
        self.skipped_scopes = 0

    def run(self, tree: ast.Module) -> ast.Module:
        if tree_has_dynamic(tree):
            self.skipped_scopes += 1
            return tree

        module_counts: Counter[str] = Counter()
        for n in iter_region(tree):
            module_counts.update(binding_names(n))
        global_declared: set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Global):
                global_declared.update(n.names)
        math_ok = module_counts.get("math", 0) == 1 and any(
            isinstance(s, ast.Import)
            and any(a.name == "math" and a.asname is None for a in s.names)
            for s in tree.body
        )
        consts = frozen_const_globals(tree)
        np_root = None
        for s in tree.body:
            if isinstance(s, ast.Import):
                for a in s.names:
                    if a.name == "numpy":
                        root = a.asname or "numpy"
                        if module_counts.get(root, 0) == 1:
                            np_root = root

        ann_ctx = (
            _ArrayAnnCtx(tree, np_root, module_counts)
            if "annotations" in self.aggressive
            else None
        )
        dyn_name = dynarray_root(tree, module_counts)

        helper = _HELPER_NAME
        taken = all_bound_names(tree) | {
            n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
        }
        while helper in taken:
            helper += "_"

        # Candidate set is a fixpoint: peers may call each other, a function
        # calling anything outside the shrinking candidate set is itself
        # incompatible, and call cycles are dropped entirely.
        candidates: dict[str, ast.FunctionDef] = {}
        for stmt in tree.body:
            if not isinstance(stmt, ast.FunctionDef) or stmt.decorator_list:
                continue
            if module_counts.get(stmt.name, 0) != 1 or stmt.name in global_declared:
                continue
            if stmt.name.startswith("_opast_vec"):
                # Vectorizer helpers: the numpy version takes the numpy
                # module as a parameter (untypeable), and decorating the
                # Python fallback would stack dispatchers.
                continue
            candidates[stmt.name] = stmt
        calls: dict[str, set[str]] = {}
        while True:
            peer_names = frozenset(candidates)
            compatible = {
                name: fn
                for name, fn in candidates.items()
                if _numba_compatible(
                    fn, math_ok, peers=peer_names, consts=consts,
                    np_root=np_root, ann_ctx=ann_ctx, dyn_name=dyn_name,
                )
            }
            calls = {
                name: _peer_calls(fn, frozenset(compatible))
                for name, fn in compatible.items()
            }
            keep = _acyclic_names(calls)
            if keep == set(candidates):
                break
            candidates = {name: compatible[name] for name in keep}

        # Hot functions seed the selection; their whitelisted callees are
        # pulled in too -- the caller's nopython build needs them compiled.
        selected_names: set[str] = set()
        work = [name for name, fn in candidates.items() if _is_hot(fn)]
        while work:
            name = work.pop()
            if name in selected_names:
                continue
            selected_names.add(name)
            work.extend(calls[name])
        selected = [candidates[name] for name in selected_names]

        # Latent-hot leftovers: whitelisted, with a loop whose trip count
        # only runtime knows.  They observe first and compile on a runtime
        # trigger.  Peer calls are allowed: the caller compiles from a
        # jitsrc copy whose aliases are backfilled at trigger time (see
        # jitsupport.maybe_njit_lazy).
        lazy: dict[str, tuple[int, ...]] = {}
        for name, fn in candidates.items():
            if name in selected_names:
                continue
            latent, bounds = _lazy_bound_args(fn)
            if latent:
                lazy[name] = bounds

        outliner = _LoopOutliner(self, tree, math_ok, helper, taken,
                                 np_root=np_root)
        outlined = outliner.outline({id(f) for f in selected})
        if not selected and not outlined and not lazy:
            return tree

        called_by_peer: set[str] = set()
        for name in selected_names:
            called_by_peer |= calls[name]
        for name in lazy:
            called_by_peer |= calls[name]
        njit_names: dict[str, str] = {}
        for name in sorted(called_by_peer):
            alias = f"_opast_njit_{name}"
            while alias in taken:
                alias += "_"
            taken.add(alias)
            njit_names[name] = alias

        def _helper_call(
            attr: str, *call_args: ast.expr, keywords: tuple = ()
        ) -> ast.Call:
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=helper, ctx=ast.Load()),
                    attr=attr,
                    ctx=ast.Load(),
                ),
                args=list(call_args),
                keywords=list(keywords),
            )

        def _no_verify_kw() -> ast.keyword:
            return ast.keyword(arg="verify", value=ast.Constant(False))

        def _mutating(name: str) -> bool:
            return _mutates_annotated_params(
                candidates[name], ann_ctx, peers=frozenset(calls.get(name, ()))
            )

        # A function holding DynArray locals must be compiled from a copy:
        # the representation swap belongs to the compiled body only, so the
        # untouched original stays available as the Python fallback.
        dyn_users = {
            name
            for name in candidates
            if dyn_name is not None and dyn_locals(candidates[name], dyn_name)[0]
        }

        def _make_src(func: ast.FunctionDef):
            src = copy.deepcopy(func)
            src_name = f"_opast_jitsrc_{func.name}"
            while src_name in taken:
                src_name += "_"
            taken.add(src_name)
            src.name = src_name
            if calls[func.name]:
                _PeerCallRewriter(
                    {g: njit_names[g] for g in calls[func.name]}
                ).visit(src)
            if func.name in dyn_users:
                _DynArrayRewriter(
                    dyn_name, np_root, dyn_locals(func, dyn_name)[0]
                ).visit(src)
                ast.fix_missing_locations(src)
            return src, src_name

        new_body: list[ast.stmt] = []
        for stmt in tree.body:
            new_body.append(stmt)
            if isinstance(stmt, ast.FunctionDef) and stmt.name in lazy:
                keywords = []
                if calls[stmt.name] or stmt.name in dyn_users:
                    src, src_name = _make_src(stmt)
                    # The copy must be defined before the decorator call
                    # that references it evaluates.
                    new_body.insert(len(new_body) - 1, src)
                    keywords = [
                        ast.keyword(
                            arg="source",
                            value=ast.Name(id=src_name, ctx=ast.Load()),
                        ),
                        ast.keyword(
                            arg="peers",
                            value=ast.Tuple(
                                elts=[
                                    ast.Tuple(
                                        elts=[
                                            ast.Constant(g),
                                            ast.Constant(njit_names[g]),
                                        ],
                                        ctx=ast.Load(),
                                    )
                                    for g in sorted(calls[stmt.name])
                                ],
                                ctx=ast.Load(),
                            ),
                        ),
                    ]
                if _mutating(stmt.name):
                    keywords.append(_no_verify_kw())
                decorator = ast.Call(
                    func=ast.Attribute(
                        value=ast.Name(id=helper, ctx=ast.Load()),
                        attr="maybe_njit_lazy",
                        ctx=ast.Load(),
                    ),
                    args=[
                        ast.Tuple(
                            elts=[
                                ast.Constant(i) for i in lazy[stmt.name]
                            ],
                            ctx=ast.Load(),
                        )
                    ],
                    keywords=keywords,
                )
                stmt.decorator_list = [ast.copy_location(decorator, stmt)]
                self.changes += 1
                continue
            if not (
                isinstance(stmt, ast.FunctionDef) and stmt.name in selected_names
            ):
                continue
            func = stmt
            if calls[func.name] or func.name in dyn_users:
                # The compiled copy calls the raw dispatchers and holds the
                # DynArray representation swap; the original def stays
                # untouched as the Python fallback (its calls hit the
                # wrappers, which guard their own fallback).
                src, src_name = _make_src(func)
                new_body.append(src)
                new_body.append(
                    ast.Assign(
                        targets=[ast.Name(id=func.name, ctx=ast.Store())],
                        value=_helper_call(
                            "dispatch",
                            _helper_call(
                                "compile_only",
                                ast.Name(id=src_name, ctx=ast.Load()),
                            ),
                            ast.Name(id=func.name, ctx=ast.Load()),
                            keywords=(
                                (_no_verify_kw(),)
                                if _mutating(func.name)
                                else ()
                            ),
                        ),
                    )
                )
            else:
                decorator = ast.Attribute(
                    value=ast.Name(id=helper, ctx=ast.Load()),
                    attr="maybe_njit",
                    ctx=ast.Load(),
                )
                if _mutating(func.name):
                    decorator = ast.Call(
                        func=decorator, args=[], keywords=[_no_verify_kw()]
                    )
                func.decorator_list = [ast.copy_location(decorator, func)]
            if func.name in njit_names:
                # Raw dispatcher alias for compiled callers (None when the
                # callee could not compile -- the caller then falls back).
                new_body.append(
                    ast.Assign(
                        targets=[
                            ast.Name(id=njit_names[func.name], ctx=ast.Store())
                        ],
                        value=_helper_call(
                            "compiled_of", ast.Name(id=func.name, ctx=ast.Load())
                        ),
                    )
                )
            self.changes += 1
        tree.body = new_body

        insert_at = 0
        for i, stmt in enumerate(tree.body):
            if (
                i == 0
                and isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            ):
                insert_at = i + 1
                continue
            if isinstance(stmt, ast.ImportFrom) and stmt.module == "__future__":
                insert_at = i + 1
                continue
            break
        helper_import = ast.Import(
            names=[ast.alias(name="opast.jitsupport", asname=helper)]
        )
        # Outlined defs go right after the helper import: they execute (and
        # evaluate their decorator) before any call site further down.
        tree.body[insert_at:insert_at] = [helper_import, *outlined]
        return tree
