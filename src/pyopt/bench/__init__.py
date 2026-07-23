"""Benchmarks: pyopt-optimized execution vs plain CPython.

Run with::

    python -m pyopt.bench [--repeat N] [workload ...]

Each workload in :mod:`pyopt.bench.workloads` is executed twice per
measurement -- once compiled from the original source, once compiled from the
pyopt-optimized AST -- in the same interpreter, with GC disabled during
timing.  Workloads set a module-level ``RESULT`` so both variants can be
checked for identical behaviour.
"""
