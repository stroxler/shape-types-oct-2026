# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from shape_extensions import Int, IntTuple, IntVar


def arithmetic[N: IntVar, M: IntVar](n: Int[N], m: Int[M]) -> Int[2 * N + M]:
    return n + n + m


def tuple_and_arithmetic[N: IntVar, M: IntVar](
    n: Int[N], m: Int[M]
) -> IntTuple[N * M, M]:
    x = 7 * n + m
    return (n * m, m)


s = arithmetic(3, 10)
t = tuple_and_arithmetic(3, 10)
