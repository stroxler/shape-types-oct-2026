from shape_extensions import Int, IntTuple, IntVar


def arithmetic[N: IntVar, M: IntVar](n: Int[N], m: Int[M]) -> Int[2 * N + M]:
    out = n + n + m
    return out


def tuple_and_arithmetic[N: IntVar, M: IntVar](
    n: Int[N], m: Int[M]
) -> IntTuple[N * M, M]:
    out = (n * m, m)
    return out


s = arithmetic(3, 10)
t = tuple_and_arithmetic(3, 10)
