"""Optimisation pipeline: runs all passes to a fixpoint."""

from __future__ import annotations

import ast
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

from .passes import (
    AlgebraicSimplification,
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
    RangeToIteration,
    UnusedElimination,
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
    # Before LICM/CSE: those hoist repeated ``len(x)`` into a temp, which
    # would destroy the ``range(len(x))`` shape this pass matches (algebra
    # stays earlier so index loops get their strength rewrites first).
    RangeToIteration,
    LoopInvariantMotion,
    CommonSubexpressionElimination,
    UnusedElimination,
    FunctionInlining,
    LoopToComprehension,  # picks up range-to-iter output in the same pass
    ComprehensionToMap,
    GlobalLocalization,  # last: inline/comp-to-map get first claim on names
)

#: Public names accepted by the ``disable`` parameter (plus ``"jit"`` for the
#: one-shot jit pass).
PASS_NAMES = tuple(cls.name for cls in PASS_CLASSES)

DEFAULT_MAX_ITERATIONS = 8


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

    def record(self, pass_) -> None:
        stats = self.per_pass.setdefault(pass_.name, PassStats())
        stats.changes += pass_.changes
        stats.skipped_scopes += pass_.skipped_scopes

    @property
    def total_changes(self) -> int:
        return sum(s.changes for s in self.per_pass.values())

    def summary(self) -> str:
        lines = [
            f"opast: {self.total_changes} change(s) in {self.iterations} iteration(s)"
        ]
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
) -> tuple[ast.Module, OptimizationReport]:
    """Optimise *tree* in place-ish (the returned tree should be used).

    *disable* skips passes by name -- an iterable or a comma-separated
    string of :data:`PASS_NAMES` entries (``"jit"`` is also accepted and
    overrides the *jit* flag).  Unknown names raise :class:`ValueError`.
    """
    disabled = _normalize_disable(disable)
    active = [cls for cls in PASS_CLASSES if cls.name not in disabled]
    report = OptimizationReport()
    for iteration in range(max_iterations):
        iteration_changes = 0
        for pass_class in active:
            pass_ = pass_class()
            if pass_class is GlobalLocalization:
                # Under --jit, keep numba-whitelist builtins as globals so
                # jit candidates still pass numba's typing.
                pass_.jit_mode = jit and "jit" not in disabled
            tree = pass_.run(tree)
            report.record(pass_)
            iteration_changes += pass_.changes
        ast.fix_missing_locations(tree)
        report.iterations = iteration + 1
        if not iteration_changes:
            break
    if jit and "jit" not in disabled:
        # One-shot, after the fixpoint: decorates hot numeric functions with
        # opast.jitsupport.maybe_njit (opt-in, see README caveats).
        from .passes.jit import JitInjection

        jit_pass = JitInjection()
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
) -> OptimizationResult:
    tree = ast.parse(source, filename=filename)
    tree, report = optimize_ast(
        tree, max_iterations=max_iterations, jit=jit, disable=disable
    )
    return OptimizationResult(tree=tree, report=report, filename=filename)


def optimize_file(
    path: str | Path,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    jit: bool = False,
    disable=(),
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
    )
