"""For-range counters exercise the interval/strength work: the loop target is
proven int (range gate), so ``i // 4 -> i >> 2``, ``i % 8 -> i & 7``,
``(i % 64) ** 2 -> (i % 64) * (i % 64)`` and interval-proven ``abs(i) -> i``
all fire; LICM/CSE also see the counter as a proven int now."""


def work():
    total = 0
    for i in range(1_200_000):
        q = i // 4 + i % 8
        total = (total + q * (i % 32) + abs(i) + (i % 64) ** 2) % 1000003
    return total


RESULT = work()
