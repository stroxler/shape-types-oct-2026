"""Torch-first tour of Pyrefly shape types.

A condensed pass over the non-neural-network material from
`building_blocks.py`, `simple_numpy.py`, and `simple_jax_demo.py`, rewritten
with torch throughout: shape-level arithmetic, constructors, reductions,
broadcasting, symbolic shape functions, OLS, and a small statistics vignette
(Gaussian-mixture EM). It closes with the same EM step in numpy and JAX
to show the other backends are covered.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
from jax import Array
from shape_extensions import IntVar
from torch import Tensor

# Ordinary least squares via the normal equations.


def torch_ols[N: IntVar, P: IntVar](
    x: Tensor[[N, P]], y: Tensor[[N]]
) -> Tensor[[P]]:
    return torch.linalg.solve(x.T @ x, x.T @ y)


def run_torch_ols():
    x = torch.randn(50, 4)
    y = torch.randn(50)
    return torch_ols(x, y)


# Statistics vignette: one EM step for a Gaussian mixture with shared variance.


def gmm_em_step[N: IntVar, K: IntVar, P: IntVar](
    x: Tensor[[N, P]],
    means: Tensor[[K, P]],
    sigma2: Tensor[[]],
) -> tuple[Tensor[[K, P]], Tensor[[N, K]], Tensor[[]]]:
    # E-step: responsibilities from squared distances.
    deltas = x.unsqueeze(-2) - means.unsqueeze(-3)
    squared = (deltas**2).sum(dim=-1)
    resp = F.softmax(-0.5 * squared / sigma2, dim=-1)
    # M-step: closed-form means and variance.
    counts = resp.sum(dim=0)
    new_means = (resp.T @ x) / counts.unsqueeze(-1)
    new_sigma2 = (resp * squared).sum() / (x.shape[0] * x.shape[1])
    return new_means, resp, new_sigma2


def run_gmm(n_iters: int = 10):
    x = torch.randn(50, 4)
    means = torch.ones(3, x.shape[1])
    sigma2 = torch.ones(())
    resp = torch.ones(x.shape[0], 3)
    for _ in range(n_iters):
        means, resp, sigma2 = gmm_em_step(x, means, sigma2)
    return means, resp, sigma2


# EM algorithm in numpy


def numpy_gmm_em_step[N: IntVar, K: IntVar, P: IntVar](
    x: np.ndarray[[N, P]],
    means: np.ndarray[[K, P]],
    sigma2: np.ndarray[[]],
) -> tuple[np.ndarray[[K, P]], np.ndarray[[N, K]], np.ndarray[[]]]:
    # E-step: responsibilities from squared distances.
    deltas = np.expand_dims(x, axis=-2) - np.expand_dims(means, axis=-3)
    squared = (deltas**2).sum(axis=-1)
    shifted = -0.5 * squared / sigma2
    exp_shifted = np.exp(shifted - shifted.max(axis=-1, keepdims=True))
    resp = exp_shifted / exp_shifted.sum(axis=-1, keepdims=True)
    # M-step: closed-form means and variance.
    counts = resp.sum(axis=0)
    new_means = (resp.T @ x) / np.expand_dims(counts, axis=-1)
    new_sigma2 = (resp * squared).sum() / (x.shape[0] * x.shape[1])
    return new_means, resp, new_sigma2


# EM algorithm in jax

def jax_gmm_em_step[N: IntVar, K: IntVar, P: IntVar](
    x: Array[[N, P]],
    means: Array[[K, P]],
    sigma2: Array[[]],
) -> tuple[Array[[K, P]], Array[[N, K]], Array[[]]]:
    # E-step: responsibilities from squared distances.
    deltas = jnp.expand_dims(x, axis=-2) - jnp.expand_dims(means, axis=-3)
    squared = (deltas**2).sum(axis=-1)
    resp = jax.nn.softmax(-0.5 * squared / sigma2, axis=-1)
    # M-step: closed-form means and variance.
    counts = jnp.sum(resp, axis=0)
    new_means = jnp.matmul(resp.T, x) / jnp.expand_dims(counts, axis=-1)
    new_sigma2 = jnp.sum(resp * squared) / (x.shape[0] * x.shape[1])
    return new_means, resp, new_sigma2


# Driver for the numpy and jax gmm functions


def run_numpy_and_jax_gmm(n_iters: int = 10):
    x_np = np.random.randn(50, 4)
    means_np = np.ones((3, x_np.shape[1]))
    sigma2_np = np.ones(())
    resp_np = np.ones((x_np.shape[0], 3))
    for _ in range(n_iters):
        means_np, resp_np, sigma2_np = numpy_gmm_em_step(
            x_np, means_np, sigma2_np
        )
    x_jax = jnp.asarray(x_np)
    means_jax = jnp.ones((3, x_jax.shape[1]))
    sigma2_jax = jnp.ones(())
    resp_jax = jnp.ones((x_jax.shape[0], 3))
    for _ in range(n_iters):
        means_jax, resp_jax, sigma2_jax = jax_gmm_em_step(
            x_jax, means_jax, sigma2_jax
        )
    return (means_np, resp_np, sigma2_np), (means_jax, resp_jax, sigma2_jax)
