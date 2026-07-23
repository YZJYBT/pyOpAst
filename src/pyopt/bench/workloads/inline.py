"""Tight loop over trivial helper calls -- exercises function inlining.

Plain CPython pays full call overhead (frame push/pop) for every helper
call; pyopt inlines the bodies at the call sites.
"""

def add(a, b):
    return a + b

def double(x):
    return x + x

def scale_shift(v):
    return v * 3 + 1

N = 1_200_000
total = 0
for i in range(N):
    d = double(i)
    total = add(total, d)
    total = scale_shift(total) % 1000003

RESULT = total
