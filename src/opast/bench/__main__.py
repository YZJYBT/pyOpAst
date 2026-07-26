"""Benchmark harness: ``python -m opast.bench [--repeat N] [workload ...]``.

For every workload the harness compiles the original source and the
opast-optimized AST, executes both in this interpreter ``--repeat`` times
each (fresh globals, GC disabled during timing), verifies that both variants
produce the same module-level ``RESULT``, and reports best-of-N wall times,
the speedup, the one-off optimization cost and the number of AST changes.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m opast.bench",
        description="Compare opast-optimized execution against plain CPython.",
    )
    parser.add_argument("workloads", nargs="*",
                        help="workload names or .py paths (default: all built-in)")
    parser.add_argument("-r", "--repeat", type=int, default=3,
                        help="timed runs per variant (default: %(default)s)")
    parser.add_argument("--jit", action="store_true",
                        help="run the jit pass too (requires numba; the "
                             "warmup run absorbs import + compile cost)")
    parser.add_argument("--aggressive", nargs="?", const=True, default=None,
                        metavar="OPTIONS",
                        help="enable assumption-backed optimization (bare "
                             "flag = all; see the main CLI)")
    parser.add_argument("--list", action="store_true",
                        help="list built-in workloads and exit")
    ns = parser.parse_args(argv)

    if ns.list:
        for path in discover([]):
            first_line = path.read_text(encoding="utf-8").lstrip().splitlines()[0]
            summary = first_line.strip("\"' ")
            print(f"{path.stem:<12} {summary}")
        return 0

    paths = discover(ns.workloads)
    print(f"Python {sys.version.split()[0]} | {len(paths)} workload(s) | "
          f"best of {ns.repeat} run(s) per variant"
          f"{' | jit' if ns.jit else ''}"
          f"{' | aggressive' if ns.aggressive else ''}\n")

    header = (f"{'workload':<12} {'original':>10} {'optimized':>10} "
              f"{'speedup':>8} {'opt-cost':>9} {'changes':>8}  result")
    print(header)
    print("-" * len(header))

    rows = []
    all_match = True
    for path in paths:
        row = bench_one(
            path, ns.repeat, jit=ns.jit, aggressive=ns.aggressive or ()
        )
        rows.append(row)
        all_match &= row["results_match"]
        print(
            f"{row['name']:<12} "
            f"{row['orig_best'] * 1000:>8.1f}ms "
            f"{row['opt_best'] * 1000:>8.1f}ms "
            f"{row['speedup']:>7.2f}x "
            f"{row['optimize_cost'] * 1000:>7.1f}ms "
            f"{row['changes']:>8} "
            f" {'OK' if row['results_match'] else 'MISMATCH!'}"
        )

    if len(rows) > 1:
        geo = math.exp(statistics.mean(math.log(r["speedup"]) for r in rows))
        print("-" * len(header))
        print(f"{'geomean':<12} {'':>10} {'':>10} {geo:>7.2f}x")

    if not all_match:
        print("\nopast.bench: RESULT mismatch detected -- optimizer bug!",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
