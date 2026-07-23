"""len() of a fresh, never-escaping list in hot loops -- exercises the
fresh-container analysis: LICM hoists ``len(data)`` out of both loop levels
and CSE merges the repeated ``len(data)`` statements."""

def work():
    data = [(i * 7 + 3) % 23 for i in range(3000)]
    lo = len(data) // 4
    hi = len(data) - lo
    total = 0
    rounds = 0
    while rounds < 250:
        i = lo
        while i < hi:
            total = (total + data[i] * (len(data) + 1)) % 1000003
            i += 1
        rounds += 1
    return total

RESULT = work()
