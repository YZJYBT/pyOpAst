"""``DynArray``: a marker container the jit path may represent differently.

``DynArray`` is a plain ``list`` subclass.  Outside a jitted function -- and
in any interpreter that never runs opast at all -- it *is* a list, with
list performance and list semantics; importing it costs nothing and adds no
dependency.  Its purpose is to let you state a contract that no static
analysis can establish on its own:

    the elements are homogeneous numbers, nothing depends on the
    container's object identity or concrete type, and it does not leave
    the function that built it.

Inside a function the ``--jit`` pass compiles, that promise lets opast pick
the representation numba is fastest with, in the *compiled copy only* (the
Python fallback keeps running this list, so a compilation failure or a
numba-free interpreter is still exactly correct):

* ``DynArray()`` -- a growable stack/queue -- becomes a plain ``[]``, which
  numba's nopython mode types as a homogeneous list with no reflection or
  per-call copying.  Dynamic-capacity algorithms (DFS, monotonic stacks,
  BFS queues) are the case numpy cannot serve, because arrays cannot grow.
* ``DynArray.zeros(n)`` / ``DynArray.full(n, value)`` -- a fixed-capacity
  buffer -- becomes ``np.zeros(n)`` / ``np.full(n, value)`` when the module
  imports numpy, since indexed numeric work on an array beats a typed list
  (direct memory access instead of per-element boxing).  Without a numpy
  import in the module the buffer compiles as a list instead; import numpy
  if you want the array representation.

``dtype`` is explicit rather than inferred so that both representations
agree element for element: ``dtype=float`` (the default) fills with
``0.0``/``np.float64``, ``dtype=int`` with ``0``/``np.int64``.  Mixing the
two -- storing a float into an ``int`` buffer -- truncates in the compiled
version the same way it would in numpy, so pick the one your algorithm
means.

The optimizer verifies the mechanical half of the contract (bound exactly
once, never returned, never passed on, never aliased, growth methods only
on the growable form) and rejects the whole function if anything else
touches the container.  What it cannot verify -- element homogeneity -- is
your promise, and breaking it makes compilation fail, which falls back to
this list rather than producing a wrong answer.
"""

from __future__ import annotations


class DynArray(list):
    """A list whose representation the jit path may choose (see module doc)."""

    __slots__ = ()

    @classmethod
    def zeros(cls, n: int, dtype: type = float) -> "DynArray":
        """A fixed-capacity buffer of *n* zeros (``0.0`` or ``0``)."""
        return cls([_zero(dtype)] * n)

    @classmethod
    def full(cls, n: int, value) -> "DynArray":
        """A fixed-capacity buffer of *n* copies of *value*."""
        return cls([value] * n)


def _zero(dtype: type):
    if dtype is int:
        return 0
    if dtype is float:
        return 0.0
    raise TypeError("DynArray dtype must be int or float")
