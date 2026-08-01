"""Batch build mode: optimize a whole source tree ahead of packaging.

::

    python -m opast.build src/ -o build_opt/
    pyinstaller build_opt/main.py ...

Every ``*.py`` under the source directory is optimized and written to the
mirrored path under the output directory; every other file is copied
verbatim, so the output tree is a drop-in replacement for the input.  A
file the optimizer cannot process falls back to a verbatim copy with a
warning -- the build never breaks, the fallback simply is not optimized
(``--strict`` turns any fallback into a non-zero exit instead).

Directory builds treat every file as a **library module**: its whole top
level is public API (importers consume exactly the functions and imports
that look unused from inside), so unused-elimination is forced off -- the
same contract as the import hook.  Mark entry scripts with ``--entry
main.py`` to restore full script-mode cleanup for them; a single-file
source is built with script semantics like the main CLI.

The default tier emits plain Python with **no runtime dependency on
opast**, which is what makes the output suitable for freezing (PyInstaller
and friends).  ``--jit`` output, by contrast, decorates hot functions with
``opast.jitsupport.maybe_njit`` -- bundle opast and numba if you freeze it,
or ``--aggressive --disable jit`` for the assumption-backed tier without
the runtime dependency.

Same caveats as the single-file CLI ``-o``: comments and formatting are
not preserved (the output is unparsed from the AST; a leading ``#!``
shebang line is kept), and line numbers shift.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

from .pipeline import (
    AGGRESSIVE_ASSUMPTIONS,
    AGGRESSIVE_NAMES,
    DEFAULT_MAX_ITERATIONS,
    PASS_NAMES,
    _normalize_disable,
    normalize_aggressive,
    optimize_file,
    rewrite_aggressive_argv,
)

#: Options taking a space-separated value (for rewrite_aggressive_argv).
_VALUE_OPTIONS = frozenset(
    {"-o", "--output", "--max-iterations", "--disable", "--exclude", "--entry"}
)

#: Directory / file names never entered or copied.
DEFAULT_EXCLUDES = frozenset({
    "__pycache__", ".git", ".hg", ".svn", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "node_modules",
})


class _Stats:
    def __init__(self) -> None:
        self.optimized = 0
        self.changed_files = 0
        self.total_changes = 0
        self.copied = 0
        self.fallbacks: list[Path] = []
        self.per_pass: Counter = Counter()
        self.in_play: set[str] = set()


def _shebang(path: Path) -> str:
    with open(path, "rb") as fh:
        first = fh.readline()
    if first.startswith(b"\xef\xbb\xbf"):  # UTF-8 BOM
        first = first[3:]
    if first.startswith(b"#!"):
        return first.decode("utf-8", "replace").rstrip() + "\n"
    return ""


def _optimize_one(
    src: Path, dst: Path, rel, ns, aggressive, stats, entry: bool
) -> None:
    # A library module's entire top level is its public surface (same
    # contract as the import hook): unused-elimination is forced off for
    # everything except files marked --entry (and single-file sources), and
    # so is the aggressive ``module-locals`` option -- "module-level names
    # nothing in the module reads" are exactly what importers read.
    disable = set(_normalize_disable(ns.disable))
    if not entry:
        disable.add("unused")
        aggressive = set(aggressive) - {"module-locals"}
    try:
        result = optimize_file(
            src,
            max_iterations=ns.max_iterations,
            jit=ns.jit,
            disable=disable,
            aggressive=aggressive,
        )
        text = _shebang(src) + result.source + "\n"
    except Exception as exc:
        # Never break the build: ship the file unoptimized instead.
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        stats.fallbacks.append(src)
        print(
            f"opast.build: {rel}: {type(exc).__name__}: {exc} "
            f"-- copied unchanged",
            file=sys.stderr,
        )
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    stats.optimized += 1
    changes = result.report.total_changes
    if changes:
        stats.changed_files += 1
        stats.total_changes += changes
        if not ns.quiet:
            print(f"{rel}: {changes} change(s)")
    for name, pass_stats in result.report.per_pass.items():
        stats.per_pass[name] += pass_stats.changes
    stats.in_play |= set(result.report.aggressive)


def _copy_one(src: Path, dst: Path, stats) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    stats.copied += 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m opast.build",
        description=(
            "Optimize a whole source tree into an output directory "
            "(non-.py files copied verbatim) -- e.g. as a PyInstaller "
            "pre-processing step. Files the optimizer cannot process are "
            "copied unchanged with a warning."
        ),
    )
    parser.add_argument("source", help="source directory (or single file)")
    parser.add_argument("-o", "--output", required=True, metavar="PATH",
                        help="output directory (or file, for a single-file "
                             "source); must be outside the source tree")
    parser.add_argument("--max-iterations", type=int,
                        default=DEFAULT_MAX_ITERATIONS, metavar="N",
                        help="max pipeline iterations (default: %(default)s)")
    parser.add_argument("--jit", action="store_true",
                        help="decorate hot numeric functions with numba.njit; "
                             "NOTE: the output then imports opast.jitsupport "
                             "at runtime -- bundle opast and numba when "
                             "freezing")
    parser.add_argument("--aggressive", "-O3", nargs="?", const=True,
                        default=None, metavar="OPTIONS",
                        help="enable assumption-backed optimization; bare "
                             "flag turns on all of: "
                             + ", ".join(AGGRESSIVE_NAMES)
                             + " (jit included -- see --jit's note; use "
                               "--disable jit to stay dependency-free)")
    parser.add_argument("--disable", metavar="PASSES", default="",
                        help="comma-separated pass names to skip: "
                             + ", ".join((*PASS_NAMES, "jit")))
    parser.add_argument("--exclude", metavar="NAME", action="append",
                        default=[],
                        help="directory/file name to skip (repeatable; "
                             "added to: " + ", ".join(sorted(DEFAULT_EXCLUDES))
                             + ")")
    parser.add_argument("--entry", metavar="REL", action="append", default=[],
                        help="relative path of an entry script (repeatable). "
                             "Directory builds treat every file as a library "
                             "module -- its whole top level is public API, so "
                             "unused-elimination stays off; --entry restores "
                             "full script-mode cleanup for the named files")
    parser.add_argument("--report", action="store_true",
                        help="aggregated per-pass statistics to stderr")
    parser.add_argument("--quiet", action="store_true",
                        help="print only the final summary")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero if any file fell back to a "
                             "verbatim copy")
    argv = sys.argv[1:] if argv is None else list(argv)
    ns = parser.parse_args(rewrite_aggressive_argv(argv, _VALUE_OPTIONS))

    try:
        aggressive = normalize_aggressive(ns.aggressive)
        disabled = _normalize_disable(ns.disable)  # also validates upfront
    except ValueError as exc:
        parser.error(str(exc))
    # ``--disable jit`` wins: no numba warning for a pass that will not run.
    ns.jit = (ns.jit or "jit" in aggressive) and "jit" not in disabled

    if ns.jit:
        from . import jitsupport

        if not jitsupport.numba_available():
            print(
                "opast.build: numba is not available on this interpreter; "
                "--jit decorations will run as plain Python.",
                file=sys.stderr,
            )

    src_root = Path(ns.source).resolve()
    out_root = Path(ns.output).resolve()
    if not src_root.exists():
        parser.error(f"source not found: {ns.source}")

    stats = _Stats()
    excludes = DEFAULT_EXCLUDES | set(ns.exclude)

    entries = {os.path.normcase(os.path.normpath(e)) for e in ns.entry}

    if src_root.is_file():
        dst = out_root / src_root.name if out_root.is_dir() else out_root
        if dst.resolve() == src_root:
            parser.error("output must differ from the source file")
        if src_root.suffix == ".py":
            # A lone file is built with script semantics, like the main CLI.
            _optimize_one(
                src_root, dst, src_root.name, ns, aggressive, stats,
                entry=True,
            )
        else:
            _copy_one(src_root, dst, stats)
    else:
        if out_root == src_root or out_root.is_relative_to(src_root):
            parser.error(
                "output directory must be outside the source directory"
            )
        for dirpath, dirnames, filenames in os.walk(src_root):
            dirnames[:] = sorted(d for d in dirnames if d not in excludes)
            for fname in sorted(filenames):
                if fname in excludes:
                    continue
                src = Path(dirpath) / fname
                rel = src.relative_to(src_root)
                dst = out_root / rel
                if fname.endswith(".py"):
                    is_entry = (
                        os.path.normcase(os.path.normpath(str(rel)))
                        in entries
                    )
                    _optimize_one(
                        src, dst, rel, ns, aggressive, stats, entry=is_entry
                    )
                else:
                    _copy_one(src, dst, stats)

    summary = (
        f"opast.build: {stats.optimized} file(s) optimized "
        f"({stats.changed_files} with changes, "
        f"{stats.total_changes} change(s) total), "
        f"{stats.copied} file(s) copied"
    )
    if stats.fallbacks:
        summary += f", {len(stats.fallbacks)} fallback(s)"
    print(summary)

    if ns.report:
        lines = []
        for name in sorted(stats.in_play):
            lines.append(f"aggressive: {name}")
            lines.append(f"  assuming: {AGGRESSIVE_ASSUMPTIONS[name]}")
        for name, changes in stats.per_pass.items():
            if changes:
                lines.append(f"  {name}: {changes} change(s)")
        if lines:
            print("\n".join(lines), file=sys.stderr)

    return 1 if (ns.strict and stats.fallbacks) else 0


if __name__ == "__main__":
    sys.exit(main())
