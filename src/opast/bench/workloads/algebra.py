"""Provable-int locals with identity-op noise -- exercises algebraic
simplification (x*1, +0, <<0, //1, |0 ... removed only because the pass can
prove the operands are plain ints).  CPython's own compiler never does this.
"""

def work(n):
    acc = 0
    i = 0
    while i < n:
        acc = (acc + i * 1 + 0 - 0) % 1000003
        acc = (acc << 0) | 0
        acc = acc // 1 + (0 + i) - (i - 0) + i
        i += 1
    return acc

RESULT = work(1_000_000)
