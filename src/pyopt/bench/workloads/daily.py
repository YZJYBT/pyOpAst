"""Everyday-style daily order-settlement report -- one realistic script that
touches every practical pass at once: config via eval/getattr/setattr
(de-dynamize), constant folding, const-prop feeding dead branches, a tiny tax
helper (inline), invariant fees (licm), a shared subexpression (cse), a fresh
price table (len-cache), builtin comprehensions (comp-to-map), int identity
noise (algebra) and leftover imports/locals/debug code (unused + dead-code).
"""

import json
import os                                # never used -> unused-import removal

DEBUG = False
SECONDS_PER_DAY = 24 * 60 * 60           # constant folding
CACHE_TTL = eval("30 * 60")              # de-dynamize -> folding
MAX_ORDERS = 250_000

assert MAX_ORDERS > 0                    # const-prop -> fold -> dead-code
if DEBUG:                                # const-prop -> dead branch
    print("debug mode on")


class Ledger:
    def __init__(self):
        self.total = 0
        self.orders = 0


def tax(cents):
    """7% sales tax in integer cents."""
    return cents * 70 // 1000


def settle(n):
    verbose = False                                # const-prop in function scope
    prices = [999, 1499, 249, 4999, 79, 1899, 350]  # fresh list -> len() cache
    base_fee = 25
    rate = 3
    total = 0
    k = 0
    while k < n:
        if verbose:                                # -> dead branch
            print("order", k)
        handling = base_fee * rate * 1 + 8 + 0     # algebra, then licm hoists it
        p = prices[k % len(prices)]                # len(prices) hoisted too
        discount = (k * rate + base_fee) % 11      # \ cse merges the shared
        loyalty = (k * rate + base_fee) % 7        # /  k * rate + base_fee
        t = tax(p)                                 # inline -> p * 70 // 1000
        subtotal = p + t + handling - discount - loyalty
        total = (total + subtotal) % 1000003
        snapshot = total * 2                       # unused local -> removed
        k += 1
    return total
    print("settled")                               # dead code after return


def daily_report(seed):
    deltas = [(i * seed) % 400 - 200 for i in range(25_000)]
    swing = sum(abs(d) for d in deltas)            # -> sum(map(abs, deltas))
    active = sum(d for d in deltas if bool(d))     # -> sum(filter(bool, deltas))
    sampled = len([float(d) for d in deltas])      # -> len(list(map(float, ...)))
    return swing + active + sampled


def checksum(seed):
    # Hot numeric kernel: whitelist-compatible, JIT candidate under --jit.
    acc = seed
    for i in range(150_000):
        acc = (acc * 31 + i) % 1000003
    return acc


ledger = Ledger()
ledger.total = settle(MAX_ORDERS)
setattr(ledger, "orders", MAX_ORDERS)    # de-dynamize -> ledger.orders = ...
grand = getattr(ledger, "total")         # de-dynamize -> ledger.total

RESULT = (grand + daily_report(13) + checksum(grand)) % 1000003
SUMMARY = json.dumps({"orders": MAX_ORDERS, "checksum": RESULT})
