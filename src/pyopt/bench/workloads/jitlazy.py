"""Variable-bound numeric kernel for the lazy jit path (needs --jit): the
static heuristic cannot see range(n) is hot, so the runtime size trigger
compiles on the first call; without --jit this is a plain ~1x workload."""


def kernel(n):
    total = 0
    for i in range(n):
        total += (i % 17) * (i % 19) + (i & 31)
    return total


def work():
    acc = 0
    for _ in range(30):
        acc = (acc + kernel(60_000)) % 1000000007
    return acc


RESULT = work()
