"""Combined pipeline effects: inline -> fold cascade (area(3,5) becomes the
literal 15 over two iterations), dead branches, assert removal and int
identities together in one hot loop.
"""

def area(w, h):
    return w * h

def perimeter(w, h):
    return 2 * w + 2 * h

def main(n):
    total = 0
    k = 0
    while k < n:
        if False:
            total += 10 ** 100
        assert True
        a = area(3, 5)
        p = perimeter(4, 6)
        total = (total + a + p + k * 1) % 999983
        k += 1
    return total

RESULT = main(700_000)
