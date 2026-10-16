"""Triton vector-add with shape-typed pointers (IDE demo).

Simplified from the stroxler/pyrefly-shapes-for-kernels v1 triton prototype.

This is a demonstration of what Pyrefly support for triton might look like
in the future, although some features we would need to bake into Pyrefly
(like the relationship between kernel and host signature) don't yet exist.

Static type checking only: we don't install triton in this demo because
macOS wheels don't exist, so none of this will run as-is.
"""

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Literal, assert_type, cast

import torch
import triton
import triton.language as tl
from shape_extensions import Int, IntVar

import typed_triton
from typed_triton import constexpr, host_tensor, tlt
from typed_triton.launch_layout import TiledOutputLayout, tiled_output
from typed_triton.torch_views import as_host_tensor, checked_vector

N = IntVar("N")
Block = IntVar("Block")


@typed_triton.jit
def add_kernel(
    x_ptr: tlt.InPointer[[N], [1]],
    y_ptr: tlt.InPointer[[N], [1]],
    output_ptr: tlt.OutPointer[[N], [1]],
    n_elements: Int[N],
    BLOCK_SIZE: constexpr[Int[Block]],
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    # This program's lane offsets; the mask below must share their grid axis.
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def checked_add[Length: IntVar](
    x: host_tensor.Tensor[[Length], [1]],
    y: host_tensor.Tensor[[Length], [1]],
    *,
    block_size: int = 16,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor[[Length]]:
    """Validate host arrays and launch the original vector-add kernel."""
    if type(block_size) is not int or block_size <= 0 or block_size & (block_size - 1):
        raise ValueError("block_size must be a positive power of two")
    if not isinstance(output_dtype, torch.dtype):
        raise ValueError("output_dtype must be a torch.dtype")
    x_ptr, n_elements = checked_vector(x, tlt.InPointer[[Length], [1]])
    y_ptr, y_length = checked_vector(y, tlt.InPointer[[Length], [1]])
    if y_length != n_elements:
        raise ValueError("Input lengths must match")
    if x.device != y.device:
        raise ValueError("Input devices must match")
    output = torch.empty((n_elements,), dtype=output_dtype, device=x.device)
    output_view = as_host_tensor(output, host_tensor.Tensor[[Length], [1]])
    output_ptr, _ = checked_vector(output_view, tlt.OutPointer[[Length], [1]])
    if n_elements:
        layout = tiled_output(
            output_view,
            (block_size,),
            shape_parameters=("n_elements",),
            tile_parameters=("BLOCK_SIZE",),
        )
        device = (
            torch.cuda.device(x.device) if x.device.type == "cuda" else nullcontext()
        )
        with device:
            layout.launch(add_kernel, x_ptr, y_ptr, output_ptr, n_elements, block_size)
    return output


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Upstream tutorial wrapper, with device detection local to the call."""
    output = torch.empty_like(x)
    assert x.device == y.device == output.device
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    cast(Any, add_kernel)[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output