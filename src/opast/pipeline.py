"""Optimisation pipeline: runs all passes to a fixpoint."""

from __future__ import annotations

import ast
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

from .passes import (
    AlgebraicSimplification,
    AttributeHoisting,
    CommonSubexpressionElimination,
    ComprehensionToMap,
    ConditionNarrowing,
    ConstantFolding,
    ConstantPropagation,
    DeadCodeElimination,
    DeDynamize,
    FunctionInlining,
    GlobalLocalization,
    LoopFolding,
    LoopInvariantMotion,
    LoopToComprehension,
    ModuleLoopOutlining,
    RangeToIteration,
    ScalarReplacement,
    SlotsInjection,
    TailRecursion,
    UnusedElimination,
    VectorizeLoops,
)

#: Pass order within one iteration.  De-dynamization runs first: expanding
#: pointless dynamic code (eval('...')/exec('...')/globals()[...] with
#: constant strings) un-taints the module and unlocks every other pass.
#: Folding feeds propagation (``x = 2+3`` folds before ``x`` propagates),
#: propagation feeds loop-fold (``range(n)`` becomes ``range(1000)``) and
#: algebra/dead-code (constant tests), unused-elimination runs after dead
#: code freed the last uses, and inlining runs late so the next iteration
#: can fold the inlined expressions and drop helpers that became unused.
#: Loop-to-comp runs after inlining (an inlined helper can land in the
#: element expression) and right before comp-to-map, which picks the new
#: comprehension up in the same iteration.
PASS_CLASSES = (
    DeDynamize,
    ConstantFolding,
    ConstantPropagation,
    AlgebraicSimplification,
    LoopFolding,
    ConditionNarrowing,  # decides interval-provable tests; dead-code reaps
    DeadCodeElimination,
    # After dead code (tail positions are clearer once unreachable returns
    # are gone) and before the loop passes, which then optimise the loop it
    # produces in the same iteration.
    TailRecursion,
    # Before LICM/CSE: those hoist repeated ``len(x)`` into a temp, which
    # would destroy the ``range(len(x))`` shape this pass matches (algebra
    # stays earlier so index loops get their strength rewrites first).
    RangeToIteration,
    LoopInvariantMotion,
    AttributeHoisting,  # aggressive "attrs": after LICM, same block walk
    CommonSubexpressionElimination,
    UnusedElimination,
    FunctionInlining,
    # After inlining: an inlined constructor call or tuple return lands as
    # a plain local aggregate this pass can then dissolve; the scalars it
    # leaves behind feed const-prop / CSE / unused next iteration.
    ScalarReplacement,

    # Loop fission (passes/fission.py) sat here, before loop-to-comp, and
    # is complete but deliberately unregistered: measured 0.88-1.02x on
    # 3.14 -- the second traversal costs more than LIST_APPEND buys.  It
    # waits for a numpy vectorization consumer.
    LoopToComprehension,  # picks up range-to-iter output in the same pass
    ComprehensionToMap,
    # Late: every other pass gets to work on the loop while it is still a
    # plain module-level statement, and the function it lands in is not a
    # candidate for the function-scope passes until the next iteration.
    ModuleLoopOutlining,
    GlobalLocalization,  # last: inline/comp-to-map get first claim on names
)

#: Public names accepted by the ``disable`` parameter (plus ``"jit"`` for the
#: one-shot jit pass).  ``slotify`` is a one-shot pass too: it runs once on
#: the pristine module *before* the fixpoint loop, because its rejections
#: rest on module-wide evidence (a ``setattr`` reference, a class alias)
#: that later passes may legitimately erase.
PASS_NAMES = tuple(cls.name for cls in PASS_CLASSES) + (
    SlotsInjection.name,
    VectorizeLoops.name,
)

DEFAULT_MAX_ITERATIONS = 8

#: Opt-in *aggressive* options.  Everything outside this set is backed by a
#: static proof; each name here instead rests on an assumption the user
#: accepts by enabling it (reported by ``--report``).
AGGRESSIVE_NAMES = (
    "annotations",
    "attrs",
    "budgets",
    "fastmath",
    "jit",
    "loop-state",
    "module-locals",
    "numpy",
    "opt-imports",
    "pure-calls",
    "slots",
    "tail-calls",
    "unbounded-recursion",
)

#: One line per aggressive option, printed by ``--report`` so the user can
#: see exactly which bets are in play.
AGGRESSIVE_ASSUMPTIONS = {
    "annotations": (
        "parameter/return annotations reflect runtime types "
        "(and no bool is passed where int is annotated)"
    ),
    "attrs": (
        "attributes read inside loops are stable: no property/__getattr__ "
        "side effects, nothing rebinds them mid-loop, and a zero-iteration "
        "loop may now perform the lookup once"
    ),
    # Not a semantic bet -- a resource one, kept here so one flag covers
    # every "you asked for it" knob.
    "budgets": (
        "longer optimization time and bigger emitted literals/code are "
        "acceptable (no semantic change)"
    ),
    "fastmath": (
        "float rewrites need not be bit-exact: -0.0 may become 0.0, "
        "overflow may yield inf instead of OverflowError, and reciprocal "
        "multiplication may differ in the last ulp"
    ),
    "jit": "numba int64 arithmetic does not wrap for these values",
    "loop-state": (
        "when an exception escapes a module-level loop, the module's globals "
        "may hold their pre-loop values instead of partial results"
    ),
    "module-locals": (
        "module-level names that nothing in the module reads are private "
        "temporaries, not part of the module's API (refines loop-state)"
    ),
    "numpy": (
        "numeric loops over proven sequences may run as numpy vector "
        "operations: intermediates are fixed-width int64/float64 (large "
        "ints wrap instead of growing), a float reduction is reassociated "
        "into one partial sum, and numpy must be importable for the fast "
        "path (everything stays plain Python otherwise)"
    ),
    "opt-imports": "optimized imported modules are not monkeypatched",
    "pure-calls": (
        "module-level functions with no visibly effectful body are pure and "
        "never rebound: their calls may be deduplicated, hoisted, or "
        "dropped, so they may run at different times or not at all "
        "(exceptions and non-termination inside them may move or vanish), "
        "same-argument calls may share one result object, and argument "
        "objects are not mutated across the optimized region"
    ),
    "slots": (
        "classes gain __slots__: instances never receive attributes beyond "
        "those the class's own methods assign, and nothing relies on "
        "instance __dict__ (vars(obj), protocol-0/1 pickle)"
    ),
    "tail-calls": (
        "RecursionError from eliminated tail recursion is approximated by a "
        "depth counter: it fires at a somewhat different depth and follows "
        "sys.getrecursionlimit() rather than the interpreter's own stack"
    ),
    "unbounded-recursion": (
        "eliminated tail recursion may run past the recursion limit instead "
        "of raising RecursionError (drops the depth counter; refines "
        "tail-calls)"
    ),
}

#: Which passes act on each aggressive option -- used both to hand the
#: option down and to decide whether it is worth reporting (an option whose
#: every consumer was ``--disable``d is not in play).
_AGGRESSIVE_CONSUMERS = {
    "attrs": (AttributeHoisting,),
    "annotations": (
        AlgebraicSimplification,
        ConditionNarrowing,
        LoopInvariantMotion,
        CommonSubexpressionElimination,
    ),
    "fastmath": (AlgebraicSimplification,),
    "budgets": (
        ConstantFolding,
        ConstantPropagation,
        AlgebraicSimplification,
        LoopFolding,
        FunctionInlining,
        ScalarReplacement,
        ModuleLoopOutlining,
    ),
    "module-locals": (ModuleLoopOutlining,),
    "loop-state": (ModuleLoopOutlining, LoopToComprehension),
    "pure-calls": (
        LoopInvariantMotion,
        CommonSubexpressionElimination,
        UnusedElimination,
    ),
    "tail-calls": (TailRecursion,),
    "unbounded-recursion": (TailRecursion,),
}

#: Options that merely *refine* another one and do nothing on their own.
#: Enabling such an option alone is not an error -- it simply has no effect,
#: and must therefore not be reported as a bet in play.  It deliberately
#: does not imply its prerequisite: that would silently sign the user up for
#: an assumption they did not ask for.
_AGGRESSIVE_REQUIRES = {
    "module-locals": "loop-state",
    "unbounded-recursion": "tail-calls",
}


def rewrite_aggressive_argv(argv: list[str], value_options) -> list[str]:
    """Pre-parse fix for the bare ``--aggressive`` / ``-O3`` flag.

    argparse's ``nargs='?'`` swallows the next non-dash token as the
    option's value, so ``opast -O3 script.py`` would eat the script path.
    The bare flag is therefore rewritten to ``--aggressive=all`` unless the
    following token is a *valid* option list, which keeps the space-
    separated value form working.  A script literally named like an option
    list can always be passed as ``./annotations`` or with ``=all``.

    Rewriting stops at the end of *our own* options: at a literal ``--``
    and at the first positional (the script path / first workload), since
    everything beyond belongs to the user's program and must reach its
    ``sys.argv`` byte-for-byte.  Recognising where a positional starts
    needs to know which options take a space-separated value, hence
    *value_options* (e.g. ``-o``/``--disable`` for the main CLI); ``=``
    forms are self-contained and need no entry.
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == "--":
            out.extend(argv[i:])
            break
        if token in ("--aggressive", "-O3"):
            nxt = argv[i + 1] if i + 1 < len(argv) else None
            if nxt is not None and not nxt.startswith("-"):
                try:
                    normalize_aggressive(nxt)
                except ValueError:
                    pass
                else:
                    out.append(f"--aggressive={nxt}")
                    i += 2
                    continue
            out.append("--aggressive=all")
            i += 1
            continue
        if token.startswith("-"):
            out.append(token)
            if token in value_options and i + 1 < len(argv):
                out.append(argv[i + 1])
                i += 2
                continue
            i += 1
            continue
        # First positional: the program's own arguments start here.
        out.extend(argv[i:])
        break
    return out


def normalize_aggressive(aggressive) -> frozenset[str]:
    """Accept True/None (meaning *all*), an iterable, or a comma-separated
    string; reject unknown names loudly."""
    if aggressive is None or aggressive is False:
        return frozenset()
    if aggressive is True or aggressive == "all":
        return frozenset(AGGRESSIVE_NAMES)
    if isinstance(aggressive, str):
        names = [part.strip() for part in aggressive.split(",") if part.strip()]
    else:
        names = list(aggressive)
    unknown = sorted(set(names) - set(AGGRESSIVE_NAMES))
    if unknown:
        raise ValueError(
            f"unknown aggressive option(s): {', '.join(unknown)} "
            f"(valid: {', '.join(AGGRESSIVE_NAMES)})"
        )
    return frozenset(names)


def _normalize_disable(disable) -> frozenset[str]:
    """Accept an iterable of pass names or one comma-separated string;
    reject unknown names loudly instead of silently ignoring a typo."""
    if not disable:
        return frozenset()
    if isinstance(disable, str):
        names = [part.strip() for part in disable.split(",") if part.strip()]
    else:
        names = list(disable)
    valid = set(PASS_NAMES) | {"jit"}
    unknown = sorted(set(names) - valid)
    if unknown:
        raise ValueError(
            f"unknown pass name(s): {', '.join(unknown)} "
            f"(valid: {', '.join((*PASS_NAMES, 'jit'))})"
        )
    return frozenset(names)


@dataclass
class PassStats:
    changes: int = 0
    skipped_scopes: int = 0


@dataclass
class OptimizationReport:
    per_pass: dict[str, PassStats] = field(default_factory=dict)
    iterations: int = 0
    aggressive: frozenset = field(default_factory=frozenset)

    def record(self, pass_) -> None:
        stats = self.per_pass.setdefault(pass_.name, PassStats())
        stats.changes += pass_.changes
        stats.skipped_scopes += pass_.skipped_scopes

    @property
    def total_changes(self) -> int:
        return sum(s.changes for s in self.per_pass.values())

    def summary(self) -> str:
        lines = []
        if self.aggressive:
            enabled = sorted(self.aggressive)
            lines.append(f"aggressive: {', '.join(enabled)}")
            for name in enabled:
                lines.append(f"  assuming: {AGGRESSIVE_ASSUMPTIONS[name]}")
        lines.append(
            f"opast: {self.total_changes} change(s) in {self.iterations} iteration(s)"
        )
        for name, stats in self.per_pass.items():
            line = f"  {name}: {stats.changes} change(s)"
            if stats.skipped_scopes:
                line += f", {stats.skipped_scopes} dynamic scope(s) skipped"
            lines.append(line)
        return "\n".join(lines)


@dataclass
class OptimizationResult:
    tree: ast.Module
    report: OptimizationReport
    filename: str = "<opast>"

    @property
    def source(self) -> str:
        return ast.unparse(self.tree)


def optimize_ast(
    tree: ast.Module,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    jit: bool = False,
    disable=(),
    aggressive=(),
) -> tuple[ast.Module, OptimizationReport]:
    """Optimise *tree* in place-ish (the returned tree should be used).

    *disable* skips passes by name -- an iterable or a comma-separated
    string of :data:`PASS_NAMES` entries (``"jit"`` is also accepted and
    overrides the *jit* flag).  Unknown names raise :class:`ValueError`.

    *aggressive* opts into assumption-backed optimisation: ``True`` for all
    of :data:`AGGRESSIVE_NAMES`, or a subset (iterable or comma-separated
    string).  ``"jit"`` there is equivalent to the *jit* flag; *disable*
    still wins over both.
    """
    disabled = _normalize_disable(disable)
    aggressive = normalize_aggressive(aggressive)
    jit = (jit or "jit" in aggressive) and "jit" not in disabled
    slots = "slots" in aggressive and SlotsInjection.name not in disabled
    vectorize = "numpy" in aggressive and VectorizeLoops.name not in disabled
    active = [cls for cls in PASS_CLASSES if cls.name not in disabled]
    # An assumption is only worth reporting when something actually acts on
    # it: ``--disable`` can switch off the jit pass or an option's every
    # consumer, and ``opt-imports`` is the caller's business, not the
    # pipeline's (the CLI adds that line when it installs the hook).
    in_play = {
        name
        for name, consumers in _AGGRESSIVE_CONSUMERS.items()
        if name in aggressive and any(cls in consumers for cls in active)
    }
    # A refining option is only in play once the option it refines is.
    in_play -= {
        name
        for name, required in _AGGRESSIVE_REQUIRES.items()
        if required not in in_play
    }
    if jit:
        in_play.add("jit")
    if slots:
        in_play.add("slots")
    if vectorize:
        in_play.add("numpy")
    report = OptimizationReport(aggressive=frozenset(in_play))
    if slots:
        # One-shot, before the fixpoint: the pass's rejections rest on
        # module-wide evidence (a setattr reference, a class alias, a
        # visible extra attribute) that later passes may legitimately
        # erase -- deciding on the pristine module keeps a rejection a
        # rejection.  Downstream passes tolerate the injected line.
        slots_pass = SlotsInjection()
        slots_pass.aggressive = frozenset({"slots"})
        tree = slots_pass.run(tree)
        report.record(slots_pass)
        ast.fix_missing_locations(tree)
    # Decided once, pre-loop, on the pristine module -- same reasoning as
    # slots: purity rejections rest on module-wide evidence (a rebinding, an
    # attribute store) that later passes may legitimately erase.
    pure_set = frozenset()
    if "pure-calls" in in_play:
        from .purity import trusted_pure_functions

        pure_set = trusted_pure_functions(tree)
    for iteration in range(max_iterations):
        iteration_changes = 0
        for pass_class in active:
            pass_ = pass_class()
            if pass_class in (GlobalLocalization, UnusedElimination):
                # Under --jit, keep the names numba's whitelist keys on:
                # localize must not alias them, and unused must not drop a
                # numpy import the jit pass is about to start reading.
                pass_.jit_mode = jit
            # Each pass picks the options it understands out of the set.
            pass_.aggressive = frozenset(
                name
                for name, consumers in _AGGRESSIVE_CONSUMERS.items()
                if name in aggressive and pass_class in consumers
            )
            if pure_set and pass_class in _AGGRESSIVE_CONSUMERS["pure-calls"]:
                pass_.pure_calls = pure_set
            tree = pass_.run(tree)
            report.record(pass_)
            iteration_changes += pass_.changes
        ast.fix_missing_locations(tree)
        report.iterations = iteration + 1
        if not iteration_changes:
            break
    if vectorize:
        # One-shot, after the fixpoint (the loop shapes it matches are the
        # fixpoint's *output* -- loop-to-comp comprehensions) and before the
        # jit pass (which skips the generated ``_opast_vec`` helpers).
        vec_pass = VectorizeLoops()
        tree = vec_pass.run(tree)
        report.record(vec_pass)
        ast.fix_missing_locations(tree)
    if jit:
        # One-shot, after the fixpoint: decorates hot numeric functions with
        # opast.jitsupport.maybe_njit (opt-in, see README caveats).
        from .passes.jit import JitInjection

        jit_pass = JitInjection()
        # The jit whitelist consults the ``annotations`` bet (NDArray-
        # annotated parameters count as proven arrays).
        jit_pass.aggressive = aggressive & frozenset({"annotations"})
        tree = jit_pass.run(tree)
        report.record(jit_pass)
        ast.fix_missing_locations(tree)
    return tree, report


def optimize_source(
    source: str,
    filename: str = "<opast>",
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    jit: bool = False,
    disable=(),
    aggressive=(),
) -> OptimizationResult:
    tree = ast.parse(source, filename=filename)
    tree, report = optimize_ast(
        tree,
        max_iterations=max_iterations,
        jit=jit,
        disable=disable,
        aggressive=aggressive,
    )
    return OptimizationResult(tree=tree, report=report, filename=filename)


def optimize_file(
    path: str | Path,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    jit: bool = False,
    disable=(),
    aggressive=(),
) -> OptimizationResult:
    path = Path(path)
    with tokenize.open(path) as fh:  # honours PEP 263 encoding cookies
        source = fh.read()
    return optimize_source(
        source,
        filename=str(path),
        max_iterations=max_iterations,
        jit=jit,
        disable=disable,
        aggressive=aggressive,
    )
