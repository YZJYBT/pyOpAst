"""Type-annotated numeric kernels: meaningful only with
``--aggressive`` (or ``--aggressive=annotations``), where the parameter
annotations let LICM/CSE hoist the loop-invariant float work and let
strength reduction fire on an int parameter.  Without the flag this is a
~1.00x workload, since a parameter is never provably typed."""


def integrate(steps: int, dt: float) -> float:
    x = 0.0
    v = 1.5
    g = 9.81
    for i in range(steps):
        a = g * dt * dt
        v = v + g * dt
        x = x + v * dt + a
    return x


def bucket(n: int) -> int:
    if n <= 0:
        return 0
    total = 0
    for i in range(n):
        total += (i % 8) + (n // 4)
    return total


def work():
    return integrate(400_000, 0.001) + float(bucket(400_000))


RESULT = work()
