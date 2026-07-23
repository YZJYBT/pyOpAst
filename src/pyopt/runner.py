"""Execute optimised code with the interpreter running pyopt (CPython by
default) -- mimicking ``python script.py`` as closely as practical."""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from pathlib import Path

from .pipeline import (
    DEFAULT_MAX_ITERATIONS,
    OptimizationResult,
    optimize_file,
    optimize_source,
)


def _materialize_source(result: OptimizationResult) -> str:
    """Write the optimized source to a stable temp file and return its path.

    Needed for ``--jit``: numba retrieves function sources via
    ``inspect.getsource``, which reads ``co_filename`` -- after our
    transforms the original file's lines no longer match.  The path is
    content-addressed so numba's on-disk cache survives across runs of an
    unchanged script.
    """
    text = result.source + "\n"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
    stem = Path(result.filename).stem or "module"
    stem = "".join(c if c.isalnum() or c in "-_." else "" for c in stem) or "module"
    cache_dir = Path(tempfile.gettempdir()) / "pyopt-jit"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{stem}-{digest}.py"
    if not path.exists():
        path.write_text(text, encoding="utf-8")
    return str(path)


def execute(
    result: OptimizationResult,
    argv: tuple[str, ...] = (),
    write_source: bool = False,
    argv0: str | None = None,
) -> None:
    """Compile *result* and run it as ``__main__`` in this interpreter.

    ``sys.argv`` and ``sys.path`` are adjusted like a normal script run and
    restored afterwards.  Exceptions (including ``SystemExit``) propagate.
    With *write_source* the code object points at a materialized copy of the
    optimized source (``__file__`` still points at the original script).
    *argv0* overrides ``sys.argv[0]`` (the CLI passes ``"-c"`` to mimic
    ``python -c``).
    """
    filename = result.filename
    is_real_file = filename not in ("<pyopt>", "<string>") and os.path.exists(filename)
    script_path = str(Path(filename).resolve()) if is_real_file else filename

    compile_target = _materialize_source(result) if write_source else script_path
    code = compile(result.tree, compile_target, "exec")
    globs = {
        "__name__": "__main__",
        "__builtins__": __builtins__,
        "__spec__": None,
        "__loader__": None,
        "__package__": None,
        "__cached__": None,
    }
    if is_real_file:
        globs["__file__"] = script_path

    old_argv = sys.argv[:]
    inserted = None
    if is_real_file:
        inserted = str(Path(script_path).parent)
        sys.path.insert(0, inserted)
    if argv0 is None:
        argv0 = filename if not is_real_file else script_path
    sys.argv = [argv0, *argv]
    try:
        exec(code, globs)
    finally:
        sys.argv = old_argv
        if inserted is not None:
            try:
                sys.path.remove(inserted)
            except ValueError:
                pass


def run_path(
    path: str | os.PathLike,
    argv: tuple[str, ...] = (),
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    jit: bool = False,
    disable=(),
) -> OptimizationResult:
    """Optimise the script at *path* and run it. Returns the result.

    *disable* skips passes by name (see :data:`pyopt.PASS_NAMES`)."""
    result = optimize_file(
        path, max_iterations=max_iterations, jit=jit, disable=disable
    )
    execute(result, argv, write_source=jit)
    return result


def run_source(
    source: str,
    argv: tuple[str, ...] = (),
    filename: str = "<pyopt>",
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    jit: bool = False,
    disable=(),
) -> OptimizationResult:
    """Optimise *source* and run it. Returns the result.

    *disable* skips passes by name (see :data:`pyopt.PASS_NAMES`)."""
    result = optimize_source(
        source, filename=filename, max_iterations=max_iterations, jit=jit,
        disable=disable,
    )
    execute(result, argv, write_source=jit)
    return result
