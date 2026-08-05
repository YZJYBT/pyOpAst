"""opast -- AST-based Python source optimizer.

Optimizations: constant folding, algebraic simplification, dead code
elimination and simple function inlining.  Scopes containing dynamic
constructs (eval/exec/globals/locals/vars/compile/__import__/``import *`` ...)
are conservatively skipped so optimization never changes their behaviour.

Typical use::

    from opast import optimize_file, run_path

    result = optimize_file("script.py")
    print(result.source)            # optimized source
    print(result.report.summary())  # what happened

    run_path("script.py", argv=("--flag",))  # optimize then execute

Every entry point accepts ``disable`` -- an iterable or comma-separated
string of pass names (see :data:`PASS_NAMES`, plus ``"jit"``) to skip::

    optimize_file("script.py", disable="inline,licm")
    run_source("print(1+1)", disable=("constant-folding",))

:class:`DynArray` (a plain ``list`` subclass) lets a hot function state
that a numeric container is homogeneous, private and identity-agnostic, so
the ``--jit`` path may compile it as a numba list or a numpy buffer; see
:mod:`opast.containers`.
"""

from .containers import DynArray
from .pipeline import (
    AGGRESSIVE_NAMES,
    DEFAULT_MAX_ITERATIONS,
    PASS_NAMES,
    OptimizationReport,
    OptimizationResult,
    PassStats,
    optimize_ast,
    optimize_file,
    optimize_source,
)
from .runner import execute, run_path, run_source


def load_ipython_extension(ipython) -> None:
    """Enable ``%%opast`` in IPython/Jupyter via ``%load_ext opast``."""
    from .magic import load_ipython_extension as _load

    _load(ipython)


__version__ = "0.5.0"

__all__ = [
    "AGGRESSIVE_NAMES",
    "DynArray",
    "DEFAULT_MAX_ITERATIONS",
    "PASS_NAMES",
    "OptimizationReport",
    "OptimizationResult",
    "PassStats",
    "optimize_ast",
    "optimize_file",
    "optimize_source",
    "execute",
    "run_path",
    "run_source",
    "__version__",
]
