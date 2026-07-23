"""Benchmarks: opast-optimized execution vs plain CPython.

Run with::

    python -m opast.bench [--repeat N] [workload ...]

Each workload in :mod:`opast.bench.workloads` is executed twice per
measurement -- once compiled from the original source, once compiled from the
opast-optimized AST -- in the same interpreter, with GC disabled during
timing.  Workloads set a module-level ``RESULT`` so both variants can be
checked for identical behaviour.
"""
