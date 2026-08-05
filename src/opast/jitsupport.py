"""Runtime support for ``--jit``: best-effort numba acceleration.

:func:`maybe_njit` is applied by the jit pass to selected hot numeric
functions.  It degrades gracefully at every level:

* numba missing / incompatible with this interpreter (e.g. a CPython version
  numba does not support yet), or ``OPAST_DISABLE_JIT`` set -> the function
  is returned unchanged;
* decoration succeeds -> calls go through a dispatcher that tries the
  compiled version and, on *any* numba-raised error (typing failure,
  unsupported feature, lowering error...), permanently falls back to the
  original Python function.  ``ImportError`` from inside the compiled call is
  treated the same way: whitelisted functions contain no imports, so it can
  only come from numba's own machinery (e.g. a stale on-disk cache entry).
  Other non-numba exceptions (ZeroDivisionError, ...) propagate unchanged.
* **first-call verification**: the whitelist guarantees the jitted function
  is pure, so the first call runs both the Python and the compiled version
  and compares results.  A divergence -- in practice int64 wraparound, which
  no static filter can rule out -- triggers a permanent Python fallback
  instead of silently wrong answers for the rest of the process.  The
  verification call returns the Python result (exact semantics) and costs
  one extra Python execution; ``OPAST_JIT_NO_VERIFY=1`` opts out.  It is a
  heuristic net, not a proof: later calls with much larger arguments can
  still wrap.

The dispatcher is kept for the lifetime of the program (the module global is
never rebound to the raw numba dispatcher): a later call with types the
compiled version cannot handle must fall back instead of raising
``TypingError`` where plain Python would have worked.

**Lazy compilation** (:func:`maybe_njit_lazy`) serves candidates whose loop
bounds are only known at runtime (``for i in range(n)``): static analysis
cannot decide hotness, so the wrapper starts in *observe* mode -- every call
runs plain Python while accumulating evidence -- and compiles only when one
trigger fires:

* **size**: the jit pass identified which parameter feeds a loop bound; a
  call with that argument >= ``OPAST_JIT_LAZY_BOUND`` (default 10_000, the
  static hotness threshold) compiles immediately;
* **single-call time**: one Python execution took >= 0.1s -- compilation
  pays for itself even if the function is never called again;
* **volume**: >= 10 calls totalling >= 0.3s -- the amortized hot case.

The triggering call already ran the Python version, so its result doubles
as the ``expected`` side of first-call verification: compile, run the
compiled version on the same arguments, compare -- one call pays for
observation, compilation *and* verification together (and the verification
uses real hot arguments, where wraparound is likeliest to show).  After a
successful trigger the wrapper delegates to the same guarded dispatcher as
the eager path; on any failure it permanently falls back to Python.
Observation counts live in the process-wide registry keyed by code
identity, so re-``exec``-ing the same optimized module (bench) keeps
accumulating instead of resetting -- and a module re-exec after a
successful compile adopts the compiled function immediately.

``numba`` itself is imported on first *compilation attempt*, not at
module import: a ``--jit`` run whose lazy candidates never get hot no
longer pays the ~0.6s numba import at all.

Compiled functions and their verification status are memoized process-wide,
keyed by code identity (``co_filename`` is content-addressed for ``--jit``
runs), so re-executing the same optimized module -- the bench harness does
this dozens of times -- compiles and verifies once.  numba's on-disk cache
(``cache=True``) is only requested when the defining module is registered in
``sys.modules``: for scratch globals (bench, plain ``exec``) numba would
record the environment module as ``<dynamic>`` and every later process
loading that entry dies with ``ModuleNotFoundError``.

Known, documented semantic caveat of opting in: numba integers are fixed
64-bit -- Python code whose intermediate values exceed ``int64`` silently
wraps around under numba.  This is why ``--jit`` is opt-in, why the static
filter in :mod:`opast.passes.jit` stays narrow, and what the first-call
verification above is for.
"""

from __future__ import annotations

import functools
import math
import os
import sys
import time

_numba = None
_numba_failed = False


def _ensure_numba():
    """Import numba on first compilation attempt (never at module import:
    lazy candidates that stay cold must not pay the ~0.6s import)."""
    global _numba, _numba_failed
    if _numba is None and not _numba_failed:
        try:
            import numba  # type: ignore[import-not-found]

            _numba = numba
        except Exception:  # ImportError or any version-incompatibility crash
            _numba_failed = True
    return _numba


_numpy = None
_numpy_failed = False


def _ensure_numpy():
    """Import numpy on the first vectorized call (lazy for the same reason
    as numba: cold code paths must not pay the import)."""
    global _numpy, _numpy_failed
    if _numpy is None and not _numpy_failed:
        try:
            import numpy  # type: ignore[import-not-found]

            _numpy = numpy
        except Exception:
            _numpy_failed = True
    return _numpy


def numba_available() -> bool:
    """Cheap installedness probe for CLI warnings, without importing numba
    (the import itself stays deferred to the first compilation attempt).
    An installed-but-import-crashing numba is only discovered at compile
    time; the runtime path degrades to plain Python either way."""
    if _numba is not None:
        return True
    if _numba_failed:
        return False
    import importlib.util

    try:
        return importlib.util.find_spec("numba") is not None
    except Exception:
        return False


_DEBUG = bool(os.environ.get("OPAST_JIT_DEBUG"))
_VERIFY = not os.environ.get("OPAST_JIT_NO_VERIFY")

#: Lazy-compilation trigger thresholds (see module docstring).
_LAZY_BOUND = int(os.environ.get("OPAST_JIT_LAZY_BOUND", "10000"))
_LAZY_SINGLE_S = 0.1
_LAZY_CALLS = 10
_LAZY_TOTAL_S = 0.3

#: Process-wide state per jitted function: key -> [compiled_or_None, verified].
#: ``compiled`` is set to None when the function is poisoned (decoration
#: failure, numba runtime error or a verification mismatch) so later
#: re-executions of the same module skip numba entirely.
_registry: dict[tuple, list] = {}

#: Lazy-mode observation state: key -> [call_count, total_seconds].  Kept
#: separate from ``_registry`` (which only exists once compilation was
#: attempted) and process-wide for the same re-exec reason.
_observations: dict[tuple, list] = {}


def _func_key(func) -> tuple:
    code = func.__code__
    return (code.co_filename, code.co_firstlineno, func.__qualname__)


def _is_numba_error(exc: BaseException) -> bool:
    return type(exc).__module__.split(".")[0] == "numba"


def _is_infra_error(exc: BaseException) -> bool:
    # ImportError/ModuleNotFoundError cannot originate from whitelisted code
    # (it contains no imports) -- only from numba internals such as a cache
    # entry recorded against an unimportable module.
    return _is_numba_error(exc) or isinstance(exc, ImportError)


def _debug(message: str) -> None:
    if _DEBUG:
        print(f"opast-jit: {message}", file=sys.stderr)


def _cacheable(func) -> bool:
    """Only use numba's on-disk cache when the defining module is importable.

    The cache pickles the compile environment by module name; functions
    exec'd in scratch globals get recorded as ``<dynamic>`` and crash every
    later process that tries to load the entry (and a cache keyed to an
    unregistered module could never be reloaded usefully anyway).
    """
    modname = getattr(func, "__module__", None)
    return bool(modname) and modname in sys.modules


def _results_match(expected, got) -> bool:
    """Equality for verification: exact for ints/bools, NaN-aware with a tiny
    relative tolerance for floats (numba's libm may differ from CPython's in
    the last ulp), element-wise for the tuples outlined loops return."""
    if isinstance(expected, tuple) and isinstance(got, tuple) or (
        isinstance(expected, list) and isinstance(got, list)
    ):
        return len(expected) == len(got) and all(
            _results_match(e, g) for e, g in zip(expected, got)
        )
    if _numpy is not None and (
        isinstance(expected, _numpy.ndarray) or isinstance(got, _numpy.ndarray)
    ):
        try:  # exact (NaN-aware for float dtypes); ulp drift falls back
            if _numpy.array_equal(expected, got):
                return True
            return bool(_numpy.array_equal(expected, got, equal_nan=True))
        except Exception:  # equal_nan rejects integer dtypes, shape errors
            return False
    try:
        if isinstance(expected, float) or isinstance(got, float):
            e, g = float(expected), float(got)
            if math.isnan(e) and math.isnan(g):
                return True
            return e == g or abs(e - g) <= 1e-12 * max(abs(e), abs(g), 1.0)
        return bool(expected == got)
    except Exception:
        return False


#: Process-wide state per vectorized site: key -> "numpy" | "python".
_vector_state: dict[tuple, str] = {}


def vector_dispatch(python_func, numpy_func):
    """Dispatcher for a vectorized loop (aggressive ``numpy``).

    *python_func* is the exact original computation; *numpy_func* takes the
    numpy module as its first argument and computes the same result with
    array operations.  The first call runs both and compares
    (:func:`_results_match`); any mismatch, any exception from the numpy
    path, or numpy being unavailable permanently selects the Python path
    for this site.  The numpy path runs under ``errstate(divide/invalid=
    'raise')`` so a zero divisor or NaN-producing step becomes an exception
    -- and therefore a fallback re-running the Python path, which then
    raises (or succeeds) exactly as the original loop would.  Int64
    wraparound raises nothing anywhere; it is the option's stated bet.
    """
    key = _func_key(python_func)

    @functools.wraps(python_func)
    def wrapper(*args):
        state = _vector_state.get(key)
        if state == "python":
            return python_func(*args)
        np_mod = _ensure_numpy()
        if np_mod is None:
            _vector_state[key] = "python"
            return python_func(*args)
        if state == "numpy":
            try:
                with np_mod.errstate(divide="raise", invalid="raise"):
                    return numpy_func(np_mod, *args)
            except Exception:
                _vector_state[key] = "python"
                return python_func(*args)
        # First call: verify against the exact Python result.
        expected = python_func(*args)
        if not _VERIFY:
            _vector_state[key] = "numpy"
            return expected
        try:
            with np_mod.errstate(divide="raise", invalid="raise"):
                got = numpy_func(np_mod, *args)
        except Exception as exc:
            _debug(f"vector path of {python_func.__name__} failed: {exc!r}")
            _vector_state[key] = "python"
            return expected
        if _results_match(expected, got):
            _vector_state[key] = "numpy"
        else:
            _debug(f"vector verification mismatch in {python_func.__name__}")
            _vector_state[key] = "python"
        return expected

    return wrapper


def compile_only(func):
    """``numba.njit`` *func*, returning the raw dispatcher -- or None when
    numba is unavailable/disabled, decoration fails, or an earlier run of the
    same function was poisoned.  The raw dispatcher is what other compiled
    functions must call (a Python-level fallback wrapper cannot be typed by
    numba)."""
    if os.environ.get("OPAST_DISABLE_JIT"):
        return None
    key = _func_key(func)
    entry = _registry.get(key)
    if entry is not None:
        return entry[0]
    numba = _ensure_numba()
    if numba is None:
        return None
    try:
        compiled = numba.njit(cache=_cacheable(func))(func)
    except Exception as exc:
        _debug(f"decoration of {func.__name__} failed: {exc!r}")
        compiled = None
    _registry[key] = [compiled, False]
    return compiled


def dispatch(compiled, python_func):
    """Wrapper calling *compiled* and, on any numba/infra error or a
    first-call verification mismatch, permanently falling back to
    *python_func*.  With *compiled* None the original function is returned
    unchanged."""
    if compiled is None:
        return python_func

    state = _registry.get(_func_key(python_func))
    if state is None or state[0] is not compiled:  # direct/standalone use
        state = [compiled, False]
    use_python = False

    @functools.wraps(python_func)
    def _opast_dispatcher(*args, **kwargs):
        nonlocal use_python
        if use_python or state[0] is None:
            return python_func(*args, **kwargs)
        if _VERIFY and not state[1]:
            expected = python_func(*args, **kwargs)
            try:
                got = compiled(*args, **kwargs)
            except Exception as exc:
                # The Python run already succeeded, so *any* exception from
                # the compiled run inside the verification window -- numba
                # infrastructure or a wraparound-induced arithmetic error --
                # is a divergence: fall back with the correct result in hand
                # instead of surfacing an error Python would not have raised.
                _debug(f"{python_func.__name__} fell back to Python: {exc!r}")
                use_python = True
                state[0] = None
                return expected
            if not _results_match(expected, got):
                _debug(
                    f"{python_func.__name__}: compiled result diverges from "
                    f"Python (int64 wraparound?) -- permanent fallback"
                )
                use_python = True
                state[0] = None
                return expected
            state[1] = True
            return expected
        try:
            return compiled(*args, **kwargs)
        except Exception as exc:
            if _is_infra_error(exc):
                _debug(f"{python_func.__name__} fell back to Python: {exc!r}")
                use_python = True
                return python_func(*args, **kwargs)
            raise

    _opast_dispatcher.opast_compiled = compiled
    return _opast_dispatcher


def compiled_of(wrapper):
    """The raw compiled dispatcher behind a :func:`dispatch` wrapper (None
    when compilation never happened) -- referenced by the rewritten sources
    of jitted callers so nopython code calls nopython code."""
    return getattr(wrapper, "opast_compiled", None)


def maybe_njit(func):
    """Compile *func* with ``numba.njit`` if possible; otherwise (or on any
    later numba failure / verification mismatch) run the original function."""
    return dispatch(compile_only(func), func)


def maybe_njit_lazy(bound_args=()):
    """Decorator factory for latent-hot candidates: observe, then compile
    once a trigger fires (see module docstring).  *bound_args* holds the
    positional indices of parameters known to feed a loop bound; an empty
    tuple leaves only the time/volume triggers."""
    indices = tuple(bound_args)

    def decorate(python_func):
        if os.environ.get("OPAST_DISABLE_JIT"):
            return python_func
        key = _func_key(python_func)
        entry = _registry.get(key)
        if entry is not None:
            # A previous exec of this module already decided: adopt the
            # compiled function (verified state carries over) or stay Python.
            if entry[0] is None:
                return python_func
            return dispatch(entry[0], python_func)
        obs = _observations.setdefault(key, [0, 0.0])
        delegate = None

        def _adopt(args, kwargs, expected, reason):
            """Compile now; the triggering call's Python result serves as
            the verification expectation on the very same arguments."""
            _debug(f"{python_func.__name__}: lazy trigger ({reason})")
            compiled = compile_only(python_func)
            if compiled is None:
                return python_func
            state = _registry[key]
            if _VERIFY and not state[1]:
                try:
                    got = compiled(*args, **kwargs)
                except Exception as exc:
                    # Python already succeeded: any compiled-side exception
                    # is a divergence, same policy as the eager verifier.
                    _debug(
                        f"{python_func.__name__} fell back to Python: {exc!r}"
                    )
                    state[0] = None
                    return python_func
                if not _results_match(expected, got):
                    _debug(
                        f"{python_func.__name__}: compiled result diverges "
                        f"from Python (int64 wraparound?) -- permanent "
                        f"fallback"
                    )
                    state[0] = None
                    return python_func
                state[1] = True
            _opast_lazy_dispatcher.opast_compiled = compiled
            return dispatch(compiled, python_func)

        @functools.wraps(python_func)
        def _opast_lazy_dispatcher(*args, **kwargs):
            nonlocal delegate
            if delegate is not None:
                return delegate(*args, **kwargs)
            if indices and any(
                i < len(args)
                and isinstance(args[i], int)
                and args[i] >= _LAZY_BOUND
                for i in indices
            ):
                expected = python_func(*args, **kwargs)
                delegate = _adopt(
                    args, kwargs, expected,
                    f"bound argument >= {_LAZY_BOUND}",
                )
                return expected
            start = time.perf_counter()
            result = python_func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            obs[0] += 1
            obs[1] += elapsed
            if elapsed >= _LAZY_SINGLE_S:
                delegate = _adopt(
                    args, kwargs, result, f"single call took {elapsed:.3f}s"
                )
            elif obs[0] >= _LAZY_CALLS and obs[1] >= _LAZY_TOTAL_S:
                delegate = _adopt(
                    args, kwargs, result,
                    f"{obs[0]} calls / {obs[1]:.3f}s total",
                )
            return result

        _opast_lazy_dispatcher.opast_lazy = True
        return _opast_lazy_dispatcher

    return decorate
