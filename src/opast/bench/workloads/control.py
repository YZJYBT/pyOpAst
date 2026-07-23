"""Control workload: nothing for the optimizer to change (dynamic values,
floats, no trivial helpers).  Expected speedup ~1.00x -- verifies opast adds
no runtime regression when it finds no opportunities.
"""
import random

rng = random.Random(12345)
values = [rng.random() for _ in range(300_000)]

acc = 0.0
for v in values:
    acc += v * v - v / 2

RESULT = round(acc + max(values), 6)
