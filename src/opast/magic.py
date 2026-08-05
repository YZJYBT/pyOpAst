"""IPython/Jupyter integration: the ``%%opast`` cell magic.

Usage::

    %load_ext opast          # or: %load_ext opast.magic

    %%opast [--show] [--report] [--no-run] [--jit] [--aggressive[=OPTIONS]]
            [--disable PASSES] [--max-iterations N]
    total = 0
    for i in range(50_000):
        total += i * 2
    total

``--aggressive`` mirrors the CLI's ``-O3``: bare it enables every option in
:data:`opast.pipeline.AGGRESSIVE_NAMES`, or pass a comma-separated subset
(``--aggressive=annotations,numpy``).  ``--report`` then lists the
assumptions actually in play.

The cell is optimized with the regular pipeline and executed in the
interactive namespace, so assignments persist across cells and a trailing
expression is displayed as usual (its value also lands in ``_``).

Caveats specific to notebooks:

* Analyses are **cell-scoped**: the optimizer sees one cell as one "module".
  Names defined in *other* cells are unknown, so passes whose safety depends
  on module-wide guarantees (inlining, de-dynamization, comprehension-to-map
  and the builtin-shadowing checks) judge only the current cell.  If an
  earlier cell rebinds a builtin (``range = ...``), disable the affected
  passes or don't use the magic on cells relying on it.
* The same cut works the other way for evidence *this* cell lacks: purity
  trust (``pure-calls``), ``slots`` and the jit whitelist's ``math``/numpy
  roots and NDArray annotations are all established from what the cell
  itself contains.  Keep the imports, type aliases and helper definitions a
  hot function depends on in the *same* cell, or those options simply find
  nothing to act on.
* The cell must be pure Python -- other magics or ``!`` shell escapes inside
  the cell can't be parsed and raise ``SyntaxError``.
* ``--jit`` inherits the CLI's opt-in caveats (int64 wraparound, see README)
  and writes the optimized cell to a temp file so numba can read sources.

This module is only imported when the extension loads, keeping IPython an
optional dependency.
"""

from __future__ import annotations

import ast
import sys

from IPython.core import magic_arguments
from IPython.core.error import UsageError
from IPython.core.magic import Magics, cell_magic, magics_class

from .pipeline import (
    AGGRESSIVE_NAMES,
    DEFAULT_MAX_ITERATIONS,
    PASS_NAMES,
    OptimizationResult,
    optimize_ast,
)
from .runner import _materialize_source

_CELL_FILENAME = "<opast-cell>"
_SENTINEL = "_opast_cell_result"


@magics_class
class OpastMagics(Magics):
    @magic_arguments.magic_arguments()
    @magic_arguments.argument("--show", action="store_true",
                              help="print the optimized source")
    @magic_arguments.argument("--report", action="store_true",
                              help="print per-pass statistics to stderr")
    @magic_arguments.argument("--no-run", action="store_true",
                              help="only optimize, do not execute the cell")
    @magic_arguments.argument("--jit", action="store_true",
                              help="numba-accelerate hot numeric code "
                                   "(opt-in; see README for the int64 caveat)")
    @magic_arguments.argument("--aggressive", nargs="?", const=True,
                              default=None, metavar="OPTIONS",
                              help="assumption-backed optimization; bare flag "
                                   "enables all of "
                                   + ", ".join(AGGRESSIVE_NAMES))
    @magic_arguments.argument("--disable", default="", metavar="PASSES",
                              help="comma-separated pass names to skip: "
                                   + ", ".join((*PASS_NAMES, "jit")))
    @magic_arguments.argument("--max-iterations", type=int,
                              default=DEFAULT_MAX_ITERATIONS, metavar="N",
                              help="max pipeline iterations")
    @cell_magic
    def opast(self, line: str, cell: str) -> None:
        """Optimize the cell with opast, then run it in the user namespace."""
        args = magic_arguments.parse_argstring(self.opast, line)
        tree = ast.parse(cell, filename=_CELL_FILENAME)

        # A trailing expression is the cell's *output*, but to the pipeline
        # it is dead code (a bare constant after folding gets eliminated,
        # and popping it beforehand would let unused-elimination drop
        # helpers it references).  Rewrite it as an assignment to a fresh
        # sentinel -- module-level assignments are kept by every pass and
        # keep their names "used" -- and unwrap it after optimization.
        sentinel = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            sentinel = _SENTINEL
            taken = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            taken |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
            while sentinel in taken:
                sentinel += "_"
            assign = ast.Assign(
                targets=[ast.Name(id=sentinel, ctx=ast.Store())],
                value=tree.body[-1].value,
            )
            tree.body[-1] = ast.copy_location(assign, tree.body[-1])
            ast.fix_missing_locations(tree)

        try:
            tree, report = optimize_ast(
                tree,
                max_iterations=args.max_iterations,
                jit=args.jit,
                disable=args.disable,
                aggressive=args.aggressive,
            )
        except ValueError as exc:  # unknown --disable pass name
            raise UsageError(str(exc)) from None

        trailing = None
        if sentinel is not None:
            for i in range(len(tree.body) - 1, -1, -1):
                stmt = tree.body[i]
                if (
                    isinstance(stmt, ast.Assign)
                    and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id == sentinel
                ):
                    trailing = ast.Expr(value=stmt.value)
                    ast.copy_location(trailing, stmt)
                    del tree.body[i]
                    break
        result = OptimizationResult(
            tree=tree, report=report, filename=_CELL_FILENAME
        )

        if args.show:
            shown = result.source
            if trailing is not None:
                shown = f"{shown}\n{ast.unparse(trailing)}" if shown else (
                    ast.unparse(trailing)
                )
            print(shown)
        if args.report:
            print(report.summary(), file=sys.stderr)
        if args.no_run:
            return

        # numba reads sources via inspect, which needs a real file.
        filename = _materialize_source(result) if args.jit else _CELL_FILENAME

        # Run the trailing expression in "single" mode inside the shell's
        # display_trap, so IPython's displayhook (Out caching, the ``_``
        # binding in user_ns) fires exactly as for an unmagic'd cell even
        # when the magic is invoked outside run_cell (the trap nests).
        ns = self.shell.user_ns
        if tree.body:
            exec(compile(tree, filename, "exec"), ns)
        if trailing is not None:
            interactive = ast.Interactive(body=[trailing])
            with self.shell.display_trap:
                exec(compile(interactive, filename, "single"), ns)


def load_ipython_extension(ipython) -> None:
    ipython.register_magics(OpastMagics)
