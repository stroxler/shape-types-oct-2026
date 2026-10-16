"""Triton vector-add with shape-typed pointers (IDE demo).

Simplified from the stroxler/pyrefly-shapes-for-kernels v1 triton prototype.

This is a demonstration of what Pyrefly support for triton might look like
in the future, although some features we would need to bake into Pyrefly
(like the relationship between kernel and host signature) don't yet exist.

Static type checking only: we don't install triton in this demo because
macOS wheels don't exist, so none of this will run as-is.
"""

from typing import TYPE_CHECKING, assert_type

import torch
import triton.language as tl
from shape_extensions import Int, IntVar

import typed_triton
from typed_triton import constexpr, host_tensor, tlt
from typed_triton.launch_layout import row_output
from typed_triton.torch_views import as_host_tensor, checked_matrix, checked_vector

Rows = IntVar("Rows")
Cols = IntVar("Cols")
Stride = IntVar("Stride")
Block = IntVar("Block")


@typed_triton.jit
def _layer_norm_fwd_fused(
    X: tlt.InPointer[[Rows, Cols], [Stride, 1]],  # pointer to the input
    Y: tlt.OutPointer[[Rows, Cols], [Stride, 1]],  # pointer to the output
    W: tlt.InPointer[[Cols], [1]],  # pointer to the weights
    B: tlt.InPointer[[Cols], [1]],  # pointer to the biases
    Mean: tlt.OutPointer[[Rows], [1]],  # pointer to the mean
    Rstd: tlt.OutPointer[[Rows], [1]],  # pointer to the 1/std
    stride: Int[Stride],  # how much to increase the pointer when moving by 1 row
    N: Int[Cols],  # number of columns in X
    eps: float,  # epsilon to avoid division by zero
    BLOCK_SIZE: constexpr[Int[Block]],
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    # Upstream rebinds Y and X with +=; fresh names expose their 1D row types.
    # A column tile can only be added after this row selection (see probes).
    y_row = Y + row * stride
    x_row = X + row * stride
    # Compute mean, sweeping the row one BLOCK_SIZE tile at a time.
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(x_row + cols, mask=cols < N, other=0.0).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance over the same masked tiles.
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_row + cols, mask=cols < N, other=0.0).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.0)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(y_row + cols, y, mask=mask)


def layer_norm[Rows: IntVar, Cols: IntVar, Stride: IntVar](
    x: host_tensor.Tensor[[Rows, Cols], [Stride, 1]],
    weight: host_tensor.Tensor[[Cols], [1]],
    bias: host_tensor.Tensor[[Cols], [1]],
    *,
    block_size: int = 32,
    eps: float = 1e-5,
) -> tuple[torch.Tensor[[Rows, Cols]], torch.Tensor[[Rows]], torch.Tensor[[Rows]]]:
    """Preserve the input row stride in a checked output and launch one row per PID."""
    x_ptr, stride, rows, cols = checked_matrix(
        x, tlt.InPointer[[Rows, Cols], [Stride, 1]]
    )
    w_ptr, w_cols = checked_vector(weight, tlt.InPointer[[Cols], [1]])
    b_ptr, b_cols = checked_vector(bias, tlt.InPointer[[Cols], [1]])
    if (
        (w_cols, b_cols) != (cols, cols)
        or x.device != weight.device
        or x.device != bias.device
    ):
        raise ValueError("Input, weight, and bias shapes and devices must match")
    if (
        not x.dtype.is_floating_point
        or weight.dtype != x.dtype
        or bias.dtype != x.dtype
    ):
        raise ValueError("Input, weight, and bias dtypes must match and be floating")
    if rows == 0 or cols == 0:
        raise ValueError("Layer norm requires nonempty dimensions")
    if type(block_size) is not int or block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    y = torch.empty_strided((rows, cols), (stride, 1), dtype=x.dtype, device=x.device)
    mean = torch.empty((rows,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((rows,), dtype=torch.float32, device=x.device)
    y_view = as_host_tensor(y, host_tensor.Tensor[[Rows, Cols], [Stride, 1]])
    y_ptr, _, _, _ = checked_matrix(y_view, tlt.OutPointer[[Rows, Cols], [Stride, 1]])
    mean_ptr, _ = checked_vector(as_host_tensor(mean), tlt.OutPointer[[Rows], [1]])
    rstd_ptr, _ = checked_vector(as_host_tensor(rstd), tlt.OutPointer[[Rows], [1]])
    layout = row_output(
        y_view,
        column_parameter="N",
        stride_parameter="stride",
        block_parameter="BLOCK_SIZE",
        block_width=block_size,
    )
    layout.launch(
        _layer_norm_fwd_fused,
        x_ptr,
        y_ptr,
        w_ptr,
        b_ptr,
        mean_ptr,
        rstd_ptr,
        stride,
        cols,
        eps,
        block_size,
    )
    return y, mean, rstd