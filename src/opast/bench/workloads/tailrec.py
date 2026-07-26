"""Accumulator-style tail recursion: the self-call becomes a jump, so the
per-call frame setup disappears.  By default a depth counter preserves the
RecursionError behaviour and keeps most of the win; ``--aggressive=
unbounded-recursion`` drops the counter for the rest of it."""


def walk(n, acc):
    if n == 0:
        return acc
    return walk(n - 1, acc + n % 7)


def work():
    total = 0
    for _ in range(2000):
        total += walk(400, 0)
    return total


RESULT = work()
