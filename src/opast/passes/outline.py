"""Module-level loop outlining: give top-level hot loops fast locals.

Code at module level compiles to ``LOAD_NAME``/``STORE_NAME`` -- a chain of
dictionary lookups -- while the same code inside a function uses
``LOAD_FAST``/``STORE_FAST`` array slots.  Measured on CPython 3.14 a plain
accumulator loop runs about **twice** as fast once moved into a function,
which is why "put it in a function" is folklore advice.  This pass does it
mechanically::

    total = 0                       def _opast_outline_0(total):
    for i in range(1_000_000):          for i in range(1_000_000):
        total = total + i * 2      ->        total = total + i * 2
                                        return (total, i)
                                    total = 0
                                    (total, i) = _opast_outline_0(total)

Only *module-level* loops are eligible; inside a function the names are
already fast.  The rewrite is the same shape as the ``--jit`` loop
outliner, but without numba's whitelist, so the conditions that the
whitelist implied must be established directly:

**Inputs** are the names the loop reads that are *definitely bound* before
it (straight-line dominance scan).  A name never bound anywhere in the
module -- a builtin, say -- stays a global read inside the new function.  A
name that is only *conditionally* bound before the loop, or bound only
after it, rejects the outline: turning it into a parameter would change a
``NameError`` into passing an undefined value.

**Outputs** are the stores the module can still observe.  Because the
write-back happens once at the end instead of per iteration, an output is
only safe when nothing can read it mid-loop, and when a zero-iteration run
still leaves the right value:

* the name must not be read by any nested scope -- a function called from
  the loop body would otherwise see the stale pre-loop value;
* it must be an input as well (so zero iterations returns what was already
  there), or provably stored on every run: the target of a constant
  ``range()`` with at least one trip, in a loop with no ``break``.

Rejected outright: loops under ``try``/``with`` (an exception escaping
mid-loop would let a handler observe pre-loop values, since the write-back
never runs), loops containing ``def``/``class`` (they would close over the
new function's locals instead of the module), ``import`` statements,
``global``/``nonlocal`` declarations, ``yield``/``await``, ``del`` of a
tracked name, and any module with dynamic constructs (``globals()`` would
see through the whole thing).

Remaining opt-in cost, identical to the jit outliner's: the module globals
are written back once after the loop rather than per iteration, and an
exception escaping the loop leaves them at their pre-loop values.  Neither
is observable to single-threaded code that cannot run during the loop.

The aggressive ``module-locals`` option additionally treats a module-level
name that *nothing in the module reads* as private rather than API, which
removes the write-back requirement for loop counters and temporaries and
so makes ``for i in range(n)`` with a variable bound eligible.
"""

from __future__ import annotations

import ast

from ..analysis import all_bound_names, binding_names, for_range_binding
from ..safety import SCOPE_NODES, iter_region, tree_has_dynamic
from .base import ScopedTransformer
from .licm import definite_bindings, subtree_bindings, unbound_risk_names

_PREFIX = "_opast_outline"

#: A loop is worth a function call when it is nested, iterates an unknown
#: number of times, or has a large constant trip count.
_HOT_TRIPS = 1_000
_GENEROUS_HOT_TRIPS = 50

_FORBIDDEN = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Lambda,
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
    ast.Return,
)


def _static_trips(loop: ast.AST):
    """Trip count of a constant ``range()`` for loop, else None."""
    info = for_range_binding(loop)
    if info is None:
        return None
    args = info[1]
    if not all(
        isinstance(a, ast.Constant) and type(a.value) is int for a in args
    ):
        return None
    try:
        return len(range(*[a.value for a in args]))
    except (TypeError, ValueError):
        return None


def _reads_and_stores(node: ast.AST):
    reads: set[str] = set()
    stores: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            if isinstance(n.ctx, ast.Load):
                reads.add(n.id)
            else:
                stores.add(n.id)
    return reads, stores


def _loads(node: ast.AST) -> set[str]:
    return {
        n.id
        for n in ast.walk(node)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }


def _plain_targets(stmt: ast.stmt) -> set[str]:
    """Names a *direct* statement of the body assigns outright."""
    if isinstance(stmt, ast.Assign):
        names = set()
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)) and all(
                isinstance(e, ast.Name) for e in target.elts
            ):
                names.update(e.id for e in target.elts)
        return names
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return {stmt.target.id} if stmt.value is not None else set()
    return set()


def _iteration_analysis(loop):
    """``(needs_input, stored_every_iteration)`` for one iteration.

    A name is *loop-local* when the iteration writes it before reading it,
    so it needs no value from the module; anything read while still unwritten
    must be handed in as a parameter.  Statements nested in conditionals
    contribute their reads but never count as writes -- the conservative
    direction on both questions.
    """
    written: set[str] = set()
    needs_input: set[str] = set()
    unconditional: set[str] = set()

    def observe(node) -> None:
        needs_input.update(_loads(node) - written)

    if isinstance(loop, ast.For):
        observe(loop.iter)
        for n in ast.walk(loop.target):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                written.add(n.id)
                unconditional.add(n.id)
    else:
        observe(loop.test)

    for stmt in loop.body:
        if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            # ``x += e`` reads x first.
            if stmt.target.id not in written:
                needs_input.add(stmt.target.id)
            observe(stmt.value)
            written.add(stmt.target.id)
            unconditional.add(stmt.target.id)
            continue
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            observe(stmt.value if stmt.value is not None else ast.Pass())
            targets = _plain_targets(stmt)
            # A subscript/attribute target reads its base.
            for t in (
                stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            ):
                if not isinstance(t, ast.Name):
                    observe(t)
            written |= targets
            unconditional |= targets
            continue
        observe(stmt)
        # Stores inside compound statements are not guaranteed to happen.
    return needs_input, unconditional


def _nested_scope_reads(tree: ast.Module) -> set[str]:
    """Every name loaded inside any nested scope of the module.  A name in
    here can be observed mid-loop by a function the loop body calls, so it
    can never be turned into a function local."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, SCOPE_NODES):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    names.add(sub.id)
                elif isinstance(sub, (ast.Global, ast.Nonlocal)):
                    names.update(sub.names)
    return names


class ModuleLoopOutlining(ScopedTransformer):
    name = "outline"

    def run(self, tree: ast.Module) -> ast.Module:
        # Opt-in only: the write-back happens after the loop, so an escaping
        # exception leaves the module's globals at their pre-loop values --
        # observable to whoever wrapped the module's execution in a ``try``
        # (``exec``, or ``import`` inside a handler).
        if "loop-state" not in self.aggressive:
            return tree
        if tree_has_dynamic(tree):
            self.skipped_scopes += 1
            return tree
        self._module_bound = {
            name for n in iter_region(tree) for name in binding_names(n)
        }
        self._scope_reads = _nested_scope_reads(tree)
        self._taken = all_bound_names(tree) | {
            n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
        }
        self._counter = 0
        self._module_locals = "module-locals" in self.aggressive
        self._hot_trips = (
            _GENEROUS_HOT_TRIPS if "budgets" in self.aggressive else _HOT_TRIPS
        )
        tree.body = self._process(tree.body)
        return tree

    def _fresh_name(self) -> str:
        while True:
            name = f"{_PREFIX}_{self._counter}"
            self._counter += 1
            if name not in self._taken:
                self._taken.add(name)
                return name

    # -- top-level walk -----------------------------------------------------
    def _process(self, stmts: list) -> list:
        out: list[ast.stmt] = []
        bound: set[str] = set()
        for index, stmt in enumerate(stmts):
            if isinstance(stmt, (ast.For, ast.While)):
                built = self._try_outline(stmt, bound, stmts[index + 1:])
                if built is not None:
                    func, call = built
                    out.append(func)
                    out.append(call)
                    self.changes += 1
                    bound |= definite_bindings(call)
                    continue
            out.append(stmt)
            bound |= definite_bindings(stmt)
            bound -= unbound_risk_names(stmt)
        return out

    # -- one loop -----------------------------------------------------------
    def _try_outline(self, loop, bound: set[str], following: list):
        if not self._is_hot(loop):
            return None
        for node in ast.walk(loop):
            if isinstance(node, _FORBIDDEN):
                return None
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Del):
                return None
        _, stores = _reads_and_stores(loop)
        needs_input, unconditional = _iteration_analysis(loop)

        # Inputs: values the iteration reads before writing.  Those must be
        # definitely bound before the loop; a name the module never binds
        # (a builtin) stays a global read inside the new function, and
        # anything else -- conditionally bound, or bound only later -- is
        # unsafe to capture, so the outline is declined.
        inputs = []
        for name in sorted(needs_input):
            if name in bound:
                inputs.append(name)
            elif name in self._module_bound:
                return None

        has_break = any(
            isinstance(n, (ast.Break, ast.Continue)) for n in ast.walk(loop)
        )
        provable = self._provably_stored(loop, has_break, unconditional)
        outputs = []
        for name in sorted(stores):
            if self._module_locals and name not in self._scope_reads and (
                not self._read_later(name, following)
            ):
                continue  # private module temporary: no write-back needed
            if name in self._scope_reads:
                return None  # a called function could observe it mid-loop
            if name in inputs or name in provable:
                outputs.append(name)
                continue
            return None
        if not inputs and not outputs:
            return None

        params = ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=name) for name in inputs],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        )
        result = (
            ast.Tuple(
                elts=[ast.Name(id=n, ctx=ast.Load()) for n in outputs],
                ctx=ast.Load(),
            )
            if len(outputs) != 1
            else ast.Name(id=outputs[0], ctx=ast.Load())
        )
        name = self._fresh_name()
        func = ast.FunctionDef(
            name=name,
            args=params,
            body=[loop, ast.Return(value=result)] if outputs else [loop],
            decorator_list=[],
            returns=None,
            type_params=[],
        )
        call = ast.Call(
            func=ast.Name(id=name, ctx=ast.Load()),
            args=[ast.Name(id=n, ctx=ast.Load()) for n in inputs],
            keywords=[],
        )
        if outputs:
            target = (
                ast.Tuple(
                    elts=[ast.Name(id=n, ctx=ast.Store()) for n in outputs],
                    ctx=ast.Store(),
                )
                if len(outputs) != 1
                else ast.Name(id=outputs[0], ctx=ast.Store())
            )
            stmt = ast.Assign(targets=[target], value=call)
        else:
            stmt = ast.Expr(value=call)
        return (
            ast.copy_location(func, loop),
            ast.copy_location(stmt, loop),
        )

    def _is_hot(self, loop) -> bool:
        if any(
            isinstance(n, (ast.For, ast.While)) and n is not loop
            for n in ast.walk(loop)
        ):
            return True  # nested loops
        trips = _static_trips(loop)
        if trips is None:
            return True  # unknown count: assume it is worth a call
        return trips >= self._hot_trips

    @staticmethod
    def _provably_stored(loop, has_break: bool, unconditional: set[str]) -> set[str]:
        """Names certainly stored by the time the loop finishes: everything
        the iteration writes unconditionally, provided the loop provably runs
        at least once and no ``break``/``continue`` can skip a write."""
        if has_break:
            return set()
        trips = _static_trips(loop)
        if trips is None or trips < 1:
            return set()
        return set(unconditional)

    @staticmethod
    def _read_later(name: str, following: list) -> bool:
        for stmt in following:
            for n in ast.walk(stmt):
                if isinstance(n, ast.Name) and n.id == name:
                    return True
        return False
