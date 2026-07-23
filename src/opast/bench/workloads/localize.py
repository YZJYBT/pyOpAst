"""Hot loop reading builtins (abs/min) and a module-level helper -- exercises
the localize pass: every per-iteration LOAD_GLOBAL becomes a LOAD_FAST."""

OFFSET = 7


def scale(v):
    # A loop in the body keeps this out of both inlining paths on purpose.
    acc = 0
    for _ in range(2):
        acc += v + OFFSET
    return acc


def work():
    data = [(i * 13) % 97 - 48 for i in range(1500)]
    total = 0
    rounds = 0
    while rounds < 250:
        i = 0
        while i < 1500:
            total = (total + abs(data[i]) + min(i, 40) + scale(i % 9)) % 1000003
            i += 1
        rounds += 1
    return total


RESULT = work()
