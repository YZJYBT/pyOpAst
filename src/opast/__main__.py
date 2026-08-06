"""Command line interface: ``python -m opast [options] script.py [args...]``
or ``python -m opast [options] -c "code" [args...]``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .pipeline import (
    AGGRESSIVE_NAMES,
    DEFAULT_MAX_ITERATIONS,
    PASS_NAMES,
    THIRD_PARTY_OPTIONS,
    _normalize_disable,
    normalize_aggressive,
    optimize_file,
    optimize_source,
    rewrite_aggressive_argv,
)
from .runner import echo, execute


#: Our options that take a space-separated value -- rewrite_aggressive_argv
#: needs these to locate the first positional (see its docstring).
_VALUE_OPTIONS = frozenset(
    {"-o", "--output", "--max-iterations", "--disable",
     "--opt-imports-under", "-c", "--code"}
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="opast",
        description=(
            "Optimize a Python script with AST passes (constant folding, "
            "algebraic simplification, dead code elimination, simple function "
            "inlining) and run it with the current interpreter. Scopes that "
            "use dynamic constructs (eval/exec/globals/locals/vars/compile/"
            "__import__/import *) are left untouched."
        ),
    )
    parser.add_argument("--show", action="store_true",
                        help="print the optimized source to stdout")
    parser.add_argument("-o", "--output", metavar="FILE",
                        help="write the optimized source to FILE")
    parser.add_argument("--no-run", action="store_true",
                        help="only optimize, do not execute the script")
    parser.add_argument("--report", action="store_true",
                        help="print per-pass statistics to stderr")
    parser.add_argument("--max-iterations", type=int,
                        default=DEFAULT_MAX_ITERATIONS, metavar="N",
                        help="max pipeline iterations (default: %(default)s)")
    parser.add_argument("--jit", action="store_true",
                        help="decorate hot numeric functions with numba.njit "
                             "(optional dependency 'opast[jit]'; any numba "
                             "failure falls back to plain Python at runtime; "
                             "see README for the int64 caveat)")
    parser.add_argument("--aggressive", "-O3", nargs="?", const=True,
                        default=None, metavar="OPTIONS",
                        help="enable assumption-backed optimization; bare "
                             "flag turns on all of: "
                             + ", ".join(AGGRESSIVE_NAMES)
                             + ". Pass a comma-separated subset to pick. "
                               "Unlike the default passes these are not "
                               "proof-backed -- --report lists what each "
                               "one assumes")
    parser.add_argument("-O2", dest="aggressive", action="store_const",
                        const="stdlib",
                        help="like -O3 minus the options that need a "
                             "third-party package ("
                             + ", ".join(sorted(THIRD_PARTY_OPTIONS))
                             + "), so the optimized code runs on a bare "
                               "interpreter; same as --aggressive=stdlib")
    parser.add_argument("--disable", metavar="PASSES", default="",
                        help="comma-separated pass names to skip: "
                             + ", ".join((*PASS_NAMES, "jit")))
    parser.add_argument("--opt-imports", action="store_true",
                        help="also optimize imported pure-Python modules "
                             "under the script's directory (opt-in: modules "
                             "whose globals are monkeypatched from outside "
                             "must not be optimized; see README)")
    parser.add_argument("--opt-imports-under", metavar="DIR", action="append",
                        default=[],
                        help="optimize imports under DIR (repeatable; "
                             "implies the hook without adding the script's "
                             "directory unless --opt-imports is also given)")
    parser.add_argument("-c", "--code", metavar="CODE",
                        help="optimize and run CODE directly (like python -c; "
                             "remaining arguments go to sys.argv)")
    parser.add_argument("script", nargs="?",
                        help="path to the Python script (or, with -c, the "
                             "first script argument)")
    parser.add_argument("args", nargs=argparse.REMAINDER,
                        help="arguments passed to the script")
    argv = sys.argv[1:] if argv is None else list(argv)
    ns = parser.parse_args(rewrite_aggressive_argv(argv, _VALUE_OPTIONS))

    if ns.code is None and ns.script is None:
        parser.error("a script path is required (or use -c CODE)")

    try:
        aggressive = normalize_aggressive(ns.aggressive)
        disabled = _normalize_disable(ns.disable)
    except ValueError as exc:
        parser.error(str(exc))
    # The umbrella flag also turns on the two options that live outside the
    # pass pipeline; the dedicated flags keep working on their own.
    # ``--disable jit`` wins over both -- otherwise the CLI would warn
    # about numba (and materialize source for it) for a pass that the
    # pipeline is not going to run.
    ns.jit = (ns.jit or "jit" in aggressive) and "jit" not in disabled
    ns.opt_imports = ns.opt_imports or "opt-imports" in aggressive
    # The import hook is a run-time affair; its assumption only applies when
    # the hook is actually installed (the pipeline reports the rest).
    hook_planned = not ns.no_run and (ns.opt_imports or ns.opt_imports_under)

    try:
        if ns.code is not None:
            result = optimize_source(
                ns.code, filename="<string>",
                max_iterations=ns.max_iterations, jit=ns.jit,
                disable=ns.disable, aggressive=aggressive,
            )
        else:
            result = optimize_file(
                ns.script, max_iterations=ns.max_iterations, jit=ns.jit,
                disable=ns.disable, aggressive=aggressive,
            )
    except ValueError as exc:  # unknown --disable pass name
        parser.error(str(exc))

    if ns.jit:
        from . import jitsupport

        if not jitsupport.numba_available():
            print(
                "opast: numba is not available on this interpreter; "
                "--jit decorations will run as plain Python.",
                file=sys.stderr,
            )
    if ns.show:
        echo(result.source)
    if ns.output:
        Path(ns.output).write_text(result.source + "\n", encoding="utf-8")
    if ns.report:
        if hook_planned and "opt-imports" in aggressive:
            result.report.aggressive |= {"opt-imports"}
        echo(result.report.summary(), file=sys.stderr)
    if not ns.no_run:
        finder = None
        if ns.opt_imports or ns.opt_imports_under:
            from . import importhook

            roots = list(ns.opt_imports_under)
            if ns.opt_imports:
                if ns.code is not None:
                    roots.append(os.getcwd())
                else:
                    roots.append(str(Path(ns.script).resolve().parent))
            finder = importhook.install(
                roots,
                max_iterations=ns.max_iterations,
                jit=ns.jit,
                disable=ns.disable,
                report=ns.report,
                aggressive=aggressive,
            )
        try:
            if ns.code is not None:
                # With -c the positional slot, if used, is really the first
                # script argument -- mimic ``python -c`` (sys.argv[0] == "-c").
                args = ([ns.script] if ns.script is not None else []) + list(ns.args)
                execute(result, tuple(args), write_source=ns.jit, argv0="-c")
            else:
                execute(result, tuple(ns.args), write_source=ns.jit)
        finally:
            if finder is not None:
                importhook.uninstall(finder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
