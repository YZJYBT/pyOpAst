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
argument indices, enabling the immediate size trigger.  Lazy candidates
must be standalone -- functions participating in peer calls are excluded,
because the raw-dispatcher aliases (``_opast_njit_G``) resolve at
decoration time, which a lazily-compiled callee cannot honour.

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


def _numba_compatible(
    func: ast.FunctionDef,
    math_ok: bool,
    peers: frozenset[str] = frozenset(),
    consts: frozenset[str] = frozenset(),
) -> bool:
    """*peers* are other jit candidates that may be called (call position
    only -- a bare load of a peer would hand numba the fallback wrapper,
    which it cannot type).  *consts* are proven single-binding module
    constants (see :func:`frozen_const_globals`) numba may freeze."""
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
                    if not isinstance(t, (ast.Name, ast.Tuple)):
                        return False
            elif isinstance(node, ast.AugAssign):
                if not isinstance(node.target, ast.Name):
                    return False
            elif isinstance(node, ast.For):
                if node.orelse or not isinstance(node.target, ast.Name):
                    return False
                if not _is_range_call(node.iter):
                    return False
            elif isinstance(node, ast.While):
                if node.orelse:
                    return False
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
                if not (
                    math_ok
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "math"
                    and isinstance(node.ctx, ast.Load)
                ):
                    return False
            elif isinstance(node, ast.Subscript):
                # Constant-index reads only: tuple element access.  The
                # runtime fallback covers a non-indexable value; a variable
                # index on a heterogeneous tuple would not type, so it is
                # not predicted to compile.
                if not (
                    isinstance(node.ctx, ast.Load)
                    and _const_int_index(node.slice)
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
                if not (name_ok or math_call or peer_call):
                    return False
                if node.keywords or any(
                    isinstance(a, ast.Starred) for a in node.args
                ):
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
                 helper: str, taken: set[str]) -> None:
        self.owner = owner
        self.tree = tree
        self.math_ok = math_ok
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
        if not _numba_compatible(fn, self.math_ok):
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
            candidates[stmt.name] = stmt
        calls: dict[str, set[str]] = {}
        while True:
            peer_names = frozenset(candidates)
            compatible = {
                name: fn
                for name, fn in candidates.items()
                if _numba_compatible(fn, math_ok, peers=peer_names, consts=consts)
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

        # Latent-hot leftovers: whitelisted, standalone (no peer calls --
        # peer aliases resolve at decoration time, incompatible with an
        # unknown compile time), with a loop whose trip count only runtime
        # knows.  They observe first and compile on a runtime trigger.
        lazy: dict[str, tuple[int, ...]] = {}
        for name, fn in candidates.items():
            if name in selected_names or calls[name]:
                continue
            latent, bounds = _lazy_bound_args(fn)
            if latent:
                lazy[name] = bounds

        outliner = _LoopOutliner(self, tree, math_ok, helper, taken)
        outlined = outliner.outline({id(f) for f in selected})
        if not selected and not outlined and not lazy:
            return tree

        called_by_peer: set[str] = set()
        for name in selected_names:
            called_by_peer |= calls[name]
        njit_names: dict[str, str] = {}
        for name in sorted(called_by_peer):
            alias = f"_opast_njit_{name}"
            while alias in taken:
                alias += "_"
            taken.add(alias)
            njit_names[name] = alias

        def _helper_call(attr: str, *call_args: ast.expr) -> ast.Call:
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=helper, ctx=ast.Load()),
                    attr=attr,
                    ctx=ast.Load(),
                ),
                args=list(call_args),
                keywords=[],
            )

        new_body: list[ast.stmt] = []
        for stmt in tree.body:
            new_body.append(stmt)
            if isinstance(stmt, ast.FunctionDef) and stmt.name in lazy:
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
                    keywords=[],
                )
                stmt.decorator_list = [ast.copy_location(decorator, stmt)]
                self.changes += 1
                continue
            if not (
                isinstance(stmt, ast.FunctionDef) and stmt.name in selected_names
            ):
                continue
            func = stmt
            if calls[func.name]:
                # The compiled copy calls the raw dispatchers; the original
                # def stays untouched as the Python fallback (its calls hit
                # the wrappers, which guard their own fallback).
                src = copy.deepcopy(func)
                src_name = f"_opast_jitsrc_{func.name}"
                while src_name in taken:
                    src_name += "_"
                taken.add(src_name)
                src.name = src_name
                _PeerCallRewriter(
                    {g: njit_names[g] for g in calls[func.name]}
                ).visit(src)
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
                        ),
                    )
                )
            else:
                decorator = ast.Attribute(
                    value=ast.Name(id=helper, ctx=ast.Load()),
                    attr="maybe_njit",
                    ctx=ast.Load(),
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
