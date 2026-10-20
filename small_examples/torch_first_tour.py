"""Torch-first tour of Pyrefly shape types.

A condensed pass over the non-neural-network material from
`building_blocks.py`, `simple_numpy.py`, and `simple_jax_demo.py`, rewritten
with torch throughout: shape-level arithmetic, constructors, reductions,
broadcasting, symbolic shape functions, OLS, and a small statistics vignette
(Gaussian-mixture EM). It closes with the same EM step in numpy and JAX
to show the other backends are covered.
"""

from __future__ import annotations

import torch
from shape_extensions import Int, IntTuple, IntVar
from torch import Tensor

# Describing axis sizes: Int, IntVar, and symbolic arithmetic


def manipulate_sizes[A: IntVar, B: IntVar](
    a: Int[A], b: Int[B]
) -> Int[2 * A + B]:
    return a + a + b

s = manipulate_sizes(3, 10)


# Describing shapes: IntTuple

def construct_shape[N: IntVar, M: IntVar](
    n: Int[N], m: Int[M]
) -> IntTuple[2 * N + 5, M]:
    return (manipulate_sizes(n, 5), m)


# Basics of handling tensors

class ArrayOrTensor[Shape: IntTuple]:
    shape: Shape
    # all the various methods and such, with shapes!
    ...

def examples[N: IntVar](n: Int[N]):
    x = torch.ones(n, 2, 1)
    y = x.sum(dim=1)
    z = y - torch.randn(4)


# Symbolic shape functions.


def matmul[N: IntVar, M: IntVar, K: IntVar](
    x: Tensor[[M, N]], y: Tensor[[N, K]]
) -> Tensor[[M, K]]:
    return x @ y

out = matmul(torch.ones(3, 4), torch.ones(4, 3))

def drop_last_row[N: IntVar, Rest: IntTuple](
    x: Tensor[[N, *Rest]],
) -> Tensor[[N - 1, *Rest]]:
    return x[:-1]


d0 = drop_last_row(torch.randn(4))
d1 = drop_last_row(torch.randn(4, 5, 6))