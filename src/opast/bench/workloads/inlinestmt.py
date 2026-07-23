"""Multi-statement helper in a hot loop -- exercises statement-body inlining
(arbitrary argument expressions become ordered temps, the body assignments
splice in renamed, and the idle def is collected by unused-elimination)."""


def blend(a, b):
    d = a - b
    s = a + b
    return d * d + s * (s - 1)


def work(n):
    total = 0
    k = 0
    while k < n:
        v = blend(k % 97, k % 31)
        total = (total + v) % 1000003
        k += 1
    return total


RESULT = work(500_000)
