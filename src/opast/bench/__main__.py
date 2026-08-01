"""Benchmark harness: ``python -m opast.bench [--repeat N] [workload ...]``.

Every workload is measured in two modes -- **default** (proof-backed passes
only) and **aggressive** (``-O3``: every assumption-backed option, jit
included) -- each against the original source in the same interpreter
(fresh globals, GC disabled during timing).  Both variants of both modes
must produce the same module-level ``RESULT``.  ``--mode`` restricts the
run to a single mode, which also switches to a more detailed per-mode
table (optimization cost, change counts).
"""

from __future__ import annotations

import argparse
import gc
import math
import statistics
import sys
import time
from pathlib import Path

from ..pipeline import optimize_source
from ..runner import _materialize_source

WORKLOAD_DIR = Path(__file__).resolve().parent / "workloads"

_MISSING = object()

#: Mode name -> optimize_source keyword arguments.  Aggressive turns on
#: every assumption-backed option; ``jit`` rides along both through the
#: option set and explicitly (the flag also selects source
#: materialization, which numba's inspect-based source retrieval needs).
MODES = {
    "default": {"jit": False, "aggressive": ()},
    "aggressive": {"jit": True, "aggressive": True},
}


def discover(selection: list[str]) -> list[Path]:
    if not selection:
        return sorted(
            p for p in WORKLOAD_DIR.glob("*.py") if p.name != "__init__.py"
        )
    paths = []
    for name in selection:
        candidate = Path(name)
        if candidate.is_file():
            paths.append(candidate)
            continue
        builtin = WORKLOAD_DIR / f"{Path(name).stem}.py"
        if builtin.is_file():
            paths.append(builtin)
            continue
        raise SystemExit(f"opast.bench: unknown workload {name!r}")
    return paths


def run_once(code):
    """One timed execution of *code* with fresh globals (GC disabled)."""
    globs = {"__name__": "__bench__"}
    gc.collect()
    was_enabled = gc.isenabled()
    gc.disable()
    start = time.perf_counter()
    exec(code, globs)
    elapsed = time.perf_counter() - start
    if was_enabled:
        gc.enable()
    return elapsed, globs.get("RESULT", _MISSING)


def bench_one(path: Path, repeat: int, jit: bool = False, aggressive=()) -> dict:
    source = path.read_text(encoding="utf-8")
    filename = str(path)

    code_original = compile(source, filename, "exec")
    start = time.perf_counter()
    optimized = optimize_source(
        source, filename=filename, jit=jit, aggressive=aggressive
    )
    optimize_cost = time.perf_counter() - start
    # With --jit numba retrieves function sources via inspect (co_filename),
    # so compile against a materialized copy -- same as runner.execute().
    compile_target = _materialize_source(optimized) if jit else filename
    code_optimized = compile(optimized.tree, compile_target, "exec")

    # Warmup runs double as the correctness check.
    _, orig_result = run_once(code_original)
    _, opt_result = run_once(code_optimized)

    # Interleave the variants so environment drift (thermal, background
    # load) is shared instead of biasing whichever variant runs last.
    orig_times: list[float] = []
    opt_times: list[float] = []
    for _ in range(repeat):
        orig_times.append(run_once(code_original)[0])
        opt_times.append(run_once(code_optimized)[0])

    orig_best, opt_best = min(orig_times), min(opt_times)
    return {
        "name": path.stem,
        "orig_best": orig_best,
        "orig_mean": statistics.mean(orig_times),
        "opt_best": opt_best,
        "opt_mean": statistics.mean(opt_times),
        "speedup": orig_best / opt_best if opt_best > 0 else float("inf"),
        "optimize_cost": optimize_cost,
        "changes": optimized.report.total_changes,
        "results_match": orig_result == opt_result,
    }


def _single_header() -> str:
    return (f"{'workload':<12} {'original':>10} {'optimized':>10} "
            f"{'speedup':>8} {'opt-cost':>9} {'changes':>8}  result")


def _single_row(row: dict, mode: str) -> str:
    r = row[mode]
    return (
        f"{row['name']:<12} "
        f"{r['orig_best'] * 1000:>8.1f}ms "
        f"{r['opt_best'] * 1000:>8.1f}ms "
        f"{r['speedup']:>7.2f}x "
        f"{r['optimize_cost'] * 1000:>7.1f}ms "
        f"{r['changes']:>8} "
        f" {'OK' if r['results_match'] else 'MISMATCH!'}"
    )


def _both_header() -> str:
    return (f"{'workload':<12} {'original':>10} {'default':>10} "
            f"{'speedup':>8} {'aggressive':>11} {'speedup':>8} "
            f"{'changes':>8}  result")


def _both_row(row: dict) -> str:
    d, a = row["default"], row["aggressive"]
    ok = d["results_match"] and a["results_match"]
    return (
        f"{row['name']:<12} "
        f"{d['orig_best'] * 1000:>8.1f}ms "
        f"{d['opt_best'] * 1000:>8.1f}ms "
        f"{d['speedup']:>7.2f}x "
        f"{a['opt_best'] * 1000:>9.1f}ms "
        f"{a['speedup']:>7.2f}x "
        f"{str(d['changes']) + '/' + str(a['changes']):>8} "
        f" {'OK' if ok else 'MISMATCH!'}"
    )


def _geomean(rows: list[dict], mode: str) -> float:
    return math.exp(
        statistics.mean(math.log(row[mode]["speedup"]) for row in rows)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m opast.bench",
        description="Compare opast-optimized execution against plain CPython "
                    "in the default (proof-backed) and aggressive (-O3, jit "
                    "included) modes.",
    )
    parser.add_argument("workloads", nargs="*",
                        help="workload names or .py paths (default: all built-in)")
    parser.add_argument("-r", "--repeat", type=int, default=3,
                        help="timed runs per variant (default: %(default)s)")
    parser.add_argument("--mode", choices=("both", "default", "aggressive"),
                        default="both",
                        help="which tier(s) to measure (default: %(default)s; "
                             "aggressive enables every -O3 option, jit "
                             "included -- numba recommended, the warmup run "
                             "absorbs import + compile cost)")
    parser.add_argument("--list", action="store_true",
                        help="list built-in workloads and exit")
    argv = sys.argv[1:] if argv is None else list(argv)
    ns = parser.parse_args(argv)

    if ns.list:
        for path in discover([]):
            first_line = path.read_text(encoding="utf-8").lstrip().splitlines()[0]
            summary = first_line.strip("\"' ")
            print(f"{path.stem:<12} {summary}")
        return 0

    modes = ["default", "aggressive"] if ns.mode == "both" else [ns.mode]
    paths = discover(ns.workloads)
    print(f"Python {sys.version.split()[0]} | {len(paths)} workload(s) | "
          f"best of {ns.repeat} run(s) per variant | "
          f"mode: {' + '.join(modes)}\n")

    both = len(modes) == 2
    header = _both_header() if both else _single_header()
    print(header)
    print("-" * len(header))

    rows = []
    all_match = True
    for path in paths:
        row = {"name": path.stem}
        for mode in modes:
            row[mode] = bench_one(path, ns.repeat, **MODES[mode])
            all_match &= row[mode]["results_match"]
        rows.append(row)
        print(_both_row(row) if both else _single_row(row, modes[0]))

    if len(rows) > 1:
        print("-" * len(header))
        if both:
            print(f"geomean: default {_geomean(rows, 'default'):.2f}x | "
                  f"aggressive {_geomean(rows, 'aggressive'):.2f}x")
        else:
            print(f"geomean: {_geomean(rows, modes[0]):.2f}x")

    if not all_match:
        print("\nopast.bench: RESULT mismatch detected -- optimizer bug!",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
