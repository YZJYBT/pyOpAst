"""Constant-string eval()/getattr() in a hot module-level loop -- exercises
de-dynamization.  Plain CPython re-compiles the eval string on every call;
opast replaces the calls with the static expressions.
"""
import math

acc = 0.0
for i in range(50_000):
    acc += eval("i * i + i")
    acc += getattr(math, "sqrt")(i)

RESULT = round(acc, 3)
