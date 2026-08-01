"""Benchmarks: opast-optimized execution vs plain CPython.

Run with::

    python -m opast.bench [--repeat N] [--mode both|default|aggressive] [workload ...]

Each workload in :mod:`opast.bench.workloads` is measured in two modes by
default -- the proof-backed **default** tier and the **aggressive** tier
(``-O3``: every assumption-backed option, jit included) -- each executed in
this interpreter against the original source, with GC disabled during
timing.  Workloads set a module-level ``RESULT`` so every variant can be
checked for identical behaviour.
"""
