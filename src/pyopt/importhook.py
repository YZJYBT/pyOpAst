"""Import hook: optimize imported pure-Python modules (``--opt-imports``).

A :class:`PyoptFinder` on ``sys.meta_path`` intercepts imports whose source
file lives under one of the configured *roots* (by default the entry
script's directory) and compiles the module from pyopt-optimized source.

Design constraints:

* **Opt-in only.**  Optimizing a *library* module widens the observable
  surface compared to a script: external code can rebind another module's
  globals (``mod.f = patched``, ``unittest.mock.patch``), which silently
  defeats passes that rely on intra-module binding stability (inlining,
  const-prop, comp-to-map).  Do not enable this for code that is
  monkeypatched at runtime.
* **The whole top level is public.**  Functions and imports unused *within*
  a library module are exactly what importers consume (``mod.calc()``,
  re-exports), so unused-elimination is forced off for imported modules
  (entry scripts keep it).
* **Never touch ``__pycache__``.**  The standard pyc cache is keyed only by
  the source file, so optimized bytecode written there would be picked up
  by plain ``python`` runs later.  :meth:`PyoptLoader.get_code` bypasses
  the pyc machinery entirely and uses its own content-addressed cache of
  optimized *source* (``%TMP%/pyopt-imports``), keyed by source text, pyopt
  version and options.  ``PYOPT_NO_IMPORT_CACHE=1`` bypasses that cache.
* **Take over or stand aside.**  ``find_spec`` returns None for anything it
  does not rewrite, so builtin/frozen/extension modules, non-matching
  paths and other meta-path finders behave exactly as without the hook.
* Modules are compiled from the optimized source text, so traceback line
  numbers inside optimized imports refer to the optimized layout (the
  original file path is kept for display; with ``--jit`` the cache file is
  used instead so numba can read matching sources).
* Any failure to optimize falls back to compiling the original source --
  an import must never break because of pyopt.
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from dataclasses import dataclass
from importlib.machinery import PathFinder, SourceFileLoader
from pathlib import Path

from .pipeline import DEFAULT_MAX_ITERATIONS, _normalize_disable, optimize_source

_CACHE_DIR = Path(tempfile.gettempdir()) / "pyopt-imports"


def _norm(path: str | os.PathLike) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


@dataclass(frozen=True)
class _HookOptions:
    max_iterations: int
    jit: bool
    disable: tuple[str, ...]
    report: bool
    no_cache: bool


class PyoptLoader(SourceFileLoader):
    """Compiles a module from optimized source, bypassing pyc caching."""

    def __init__(self, fullname: str, path: str, options: _HookOptions) -> None:
        super().__init__(fullname, path)
        self._options = options

    def get_code(self, fullname: str):
        source = self.get_source(fullname)
        opts = self._options
        key = "\x00".join(
            (
                source,
                _pyopt_version(),
                str(opts.max_iterations),
                str(opts.jit),
                ",".join(opts.disable),
            )
        )
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
        stem = fullname.rpartition(".")[2] or "module"
        cache_path = _CACHE_DIR / f"{stem}-{digest}.py"

        optimized = None
        if not opts.no_cache:
            try:
                optimized = cache_path.read_text(encoding="utf-8")
            except OSError:
                pass
        if optimized is None:
            try:
                result = optimize_source(
                    source,
                    filename=self.path,
                    max_iterations=opts.max_iterations,
                    jit=opts.jit,
                    disable=opts.disable,
                )
            except Exception as exc:  # never break an import
                print(
                    f"pyopt: optimizing import {fullname!r} failed ({exc!r}); "
                    "running it unoptimized",
                    file=sys.stderr,
                )
                return compile(source, self.path, "exec", dont_inherit=True)
            optimized = result.source + "\n"
            if opts.report:
                print(
                    f"pyopt: import {fullname!r}: "
                    f"{result.report.total_changes} change(s) in "
                    f"{result.report.iterations} iteration(s)",
                    file=sys.stderr,
                )
            try:
                _CACHE_DIR.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(optimized, encoding="utf-8")
            except OSError:
                pass  # cache is best-effort
        elif opts.report:
            print(f"pyopt: import {fullname!r}: cached", file=sys.stderr)

        # With --jit numba reads sources via inspect: compile against the
        # cache file, whose lines match exactly.  Otherwise keep the
        # original path for tracebacks (line numbers follow the optimized
        # layout -- see module docstring).
        target = str(cache_path) if opts.jit and cache_path.exists() else self.path
        return compile(optimized, target, "exec", dont_inherit=True)


class PyoptFinder:
    """Meta-path finder wrapping matching modules with :class:`PyoptLoader`."""

    def __init__(self, roots, options: _HookOptions) -> None:
        self._roots = [_norm(r) + os.sep for r in roots]
        self._exclude = _norm(Path(__file__).parent) + os.sep  # pyopt itself
        self._options = options

    def find_spec(self, fullname, path=None, target=None):
        if fullname.partition(".")[0] == "pyopt":
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if (
            spec is None
            or not spec.origin
            or not spec.origin.endswith(".py")
            or not isinstance(spec.loader, SourceFileLoader)
        ):
            return None
        origin = _norm(spec.origin)
        if origin.startswith(self._exclude):
            return None
        if not any(origin.startswith(root) for root in self._roots):
            return None
        spec.loader = PyoptLoader(fullname, spec.origin, self._options)
        return spec


def _pyopt_version() -> str:
    from . import __version__

    return __version__


def install(
    roots,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    jit: bool = False,
    disable=(),
    report: bool = False,
) -> PyoptFinder:
    """Install the import hook for modules under *roots* (directories).
    Returns the finder; pass it to :func:`uninstall` to remove."""
    # A library module's entire top level is its public surface: functions
    # and imports that are unused *within* the module are exactly what
    # importers consume (``mod.calc()``, re-exports).  Unused-elimination is
    # therefore forced off for imported modules (entry scripts keep it).
    disabled = set(_normalize_disable(disable)) | {"unused"}
    options = _HookOptions(
        max_iterations=max_iterations,
        jit=jit,
        disable=tuple(sorted(disabled)),
        report=report,
        no_cache=bool(os.environ.get("PYOPT_NO_IMPORT_CACHE")),
    )
    finder = PyoptFinder(roots, options)
    sys.meta_path.insert(0, finder)
    return finder


def uninstall(finder: PyoptFinder) -> None:
    try:
        sys.meta_path.remove(finder)
    except ValueError:
        pass
