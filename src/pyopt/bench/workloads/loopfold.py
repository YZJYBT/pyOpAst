"""Constant-range pure-int loops evaluate at optimise time (loop-fold): the
hot kernel collapses to its final constants, and the follow-up passes
(span propagation, unused, inline) cascade the collapse outward."""


def kernel():
    total = 0
    count = 0
    for i in range(3000):
        q = i // 4 + (i & 7)
        if q % 3:
            total = (total + q * 5) % 100003
        else:
            count += 1
    return total * 10 + count + i


def work():
    acc = 0
    for _ in range(400):
        acc = (acc + kernel()) % 1000000007
    return acc


RESULT = work()
