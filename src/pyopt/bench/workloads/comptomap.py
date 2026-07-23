"""Generator/list comprehensions over builtin calls -> map/filter."""

def work():
    data = [(i * 37) % 1000 - 500 for i in range(40_000)]
    total = 0
    rounds = 0
    while rounds < 40:
        total += sum(abs(x) for x in data)          # -> sum(map(abs, data))
        total += sum(x for x in data if bool(x))    # -> sum(filter(bool, data))
        total += len([float(x) for x in data])      # -> len(list(map(float, data)))
        rounds += 1
    return total

RESULT = work()
