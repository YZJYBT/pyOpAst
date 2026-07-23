"""Loop-invariant int expressions in a hot loop -- exercises LICM.

``factor`` is a provable int but not a constant (multiple bindings), so
constant propagation can't touch it; LICM hoists ``factor * 7``,
``factor << 3`` and ``factor // 5`` out of the hot loop.
"""

def work():
    factor = 0
    k = 0
    while k < 50:
        factor = factor * 3 + k
        k += 1
    n = 250_000
    acc = 0
    i = 0
    while i < n:
        acc = (acc + factor * 7 - (factor << 3) + factor // 5) % 1000003
        i += 1
    return acc

RESULT = work()
