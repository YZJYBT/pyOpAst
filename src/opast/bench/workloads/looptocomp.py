"""Append-accumulation loops become list/set comprehensions (loop-to-comp):
LIST_APPEND bytecode replaces the per-iteration attribute lookup + method
call; the guarded loop maps to a comprehension ``if``."""


def work():
    out = []
    for i in range(400_000):
        if i % 3:
            out.append(i * 2 + 1)
    picked = []
    for j in range(0, 400_000, 4):
        picked.append(j * j % 97)
    return len(out) * 7 + out[-1] + len(picked) + picked[-1]


RESULT = work()
