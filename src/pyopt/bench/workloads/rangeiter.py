"""Index loops over a fresh list become direct iteration / enumerate
(range-to-iter): the per-iteration BINARY_SUBSCR index lookups disappear;
the index-only loop drops its counter entirely."""


def work():
    data = [k * 3 % 251 for k in range(300_000)]
    total = 0
    for i in range(len(data)):
        total += data[i]
    weighted = 0
    for j in range(len(data)):
        weighted += (j & 7) * data[j]
    return total * 3 + weighted


RESULT = work()
