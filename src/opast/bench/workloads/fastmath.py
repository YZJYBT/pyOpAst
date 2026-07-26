"""Float kernel exercising the aggressive ``fastmath`` option: the ``+ 0``
noise disappears, ``** 2`` becomes a multiplication (about 3x cheaper per
operation) and the non-power-of-two division becomes a reciprocal
multiply.  Without the flag only the bit-exact ``/ 4.0 -> * 0.25`` rewrite
fires, so this is a ~1.0x workload by default."""


def work():
    x = 0.0
    total = 0.0
    for i in range(300_000):
        x = x + 1.5
        total = total + (x + 0) / 3.0 + x ** 2 / 4.0
    return total


RESULT = work()
