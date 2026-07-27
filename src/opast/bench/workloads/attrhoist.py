"""Loop-invariant attribute chains (aggressive ``attrs``): dotted module
calls and attribute value reads hoist to pre-loop locals; instance method
calls are deliberately left alone (bound-method caching measured *slower*
than 3.14's specialized call path).  ~1.0x without the option."""

import math


class Cfg:
    def __init__(self):
        self.scale = 3
        self.offset = 1.5


def work():
    cfg = Cfg()
    total = 0.0
    for i in range(250_000):
        total += math.floor(i / 7) + cfg.scale * cfg.offset
    return total


RESULT = work()
