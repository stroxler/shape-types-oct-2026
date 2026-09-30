import numpy as np
from shape_extensions import IntTuple, IntVar

# Constructors and inlay hints

x: np.ndarray[[2, 2]] = np.array([[2, 3], [4, 5]])

z = np.ones((1, 2, 3))


# Reductions

sum0 = np.random.randn(2, 3).sum(axis=0)


# Broadcasting

b1 = np.random.randn(3, 1) - np.random.randn(4)


# Symbolic manipulation of shapes

def matmul[N: IntVar, M: IntVar, K: IntVar](
    x: np.ndarray[[M, N]], y: np.ndarray[[N, K]]
) -> np.ndarray[[M, K]]:
    return x @ y


product = matmul(np.random.randn(3, 4), np.random.randn(4, 2))


def drop_last_row[N: IntVar, Rest: IntTuple](
    x: np.ndarray[[N, *Rest]],
) -> np.ndarray[[N - 1, *Rest]]:
    return x[:-1]


d0 = drop_last_row(np.random.randn(4))
d1 = drop_last_row(np.random.randn(4, 5))


# Example: Ordinary Least Squares


def ordinary_least_squares[N: IntVar, P: IntVar](
    x: np.ndarray[[N, P]],
    y: np.ndarray[[N, 1]],
) -> np.ndarray[[P, 1]]:
    return np.linalg.solve(x.T @ x, x.T @ y)


def run_ols():
    x = np.random.randn(5, 3)
    y = np.random.randn(5, 1)
    beta = ordinary_least_squares(x, y)
    return beta
