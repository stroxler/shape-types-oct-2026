import jax
import jax.numpy as jnp
from jax import Array
from shape_extensions import Int, IntVar

# A basic unsupervised ML / statistics demo: mixture-of-gaussian EM


def gaussian_mixture_em[N: IntVar, K: IntVar, P: IntVar](
    x: Array[[N, P]], k: Int[K], n_iters: int
) -> tuple[Array[[K, P]], Array[[N, K]], Array[[]]]:
    p = x.shape[1]
    means = jnp.ones((k, p))
    sigma2 = jnp.ones(())
    resp = jnp.ones((x.shape[0], k))  # overwritten by the first E-step
    for _ in range(n_iters):
        resp = expectation(x, means, sigma2)
        means, sigma2 = maximization(x, resp, means)
    return means, resp, sigma2


def expectation[N: IntVar, K: IntVar, P: IntVar](
    x: Array[[N, P]], means: Array[[K, P]], sigma2: Array[[]]
) -> Array[[N, K]]:
    deltas = jnp.expand_dims(x, axis=-2) - jnp.expand_dims(means, axis=-3)
    squared = jnp.sum(deltas**2, axis=-1)
    return jax.nn.softmax(-0.5 * squared / sigma2, axis=-1)


def maximization[N: IntVar, K: IntVar, P: IntVar](
    x: Array[[N, P]], resp: Array[[N, K]], means: Array[[K, P]]
) -> tuple[Array[[K, P]], Array[[]]]:
    deltas = jnp.expand_dims(x, axis=-2) - jnp.expand_dims(means, axis=-3)
    squared = jnp.sum(deltas**2, axis=-1)
    counts = jnp.sum(resp, axis=0)
    means = jnp.matmul(resp.T, x) / jnp.expand_dims(counts, axis=-1)
    sigma2 = jnp.sum(resp * squared) / (x.shape[0] * x.shape[1])
    return means, sigma2


def run_gmm():
    x = jnp.ones((50, 30))
    means, resp, sigma2 = gaussian_mixture_em(x, 4, 10)
    return means, resp, sigma2


# Single-head self-attention as a plain function: jax has no neural-net
# building blocks, so the Q/K/V projections are explicit parameters.


def simple_self_attention[B: IntVar, T: IntVar, C: IntVar, D: IntVar](
    x: Array[[B, T, C]],
    w_q: Array[[C, D]],
    w_k: Array[[C, D]],
    w_v: Array[[C, D]],
) -> Array[[B, T, D]]:
    q = jnp.matmul(x, w_q)
    k = jnp.matmul(x, w_k)
    v = jnp.matmul(x, w_v)
    scores = jnp.matmul(q, k.swapaxes(-2, -1))
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(weights, v)


def run():
    x = jnp.ones((8, 32, 64))
    w = jnp.ones((64, 16))
    out = simple_self_attention(x, w, w, w)
    return out
