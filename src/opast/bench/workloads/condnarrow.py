"""Interval-provable guards inside a hot loop vanish (cond-narrow): the
range target is non-negative by construction, so the ``i < 0`` guard and
the re-test of an already-established bound are decided at optimise time
and deleted.  The bound is a parameter, so loop-fold cannot fold the loop
away first."""


def kernel(n):
    total = 0
    for i in range(n):
        if i < 0:
            total -= 999
        step = i % 7
        if step > 6:
            total -= 1
        total += step
    return total


def work():
    return kernel(700_000)


RESULT = work()
