"""Host tensor view carrying shape and element-stride information."""

import torch
from shape_extensions import IntTuple

class Tensor[Shape: IntTuple, Strides: IntTuple](torch.Tensor[Shape]): ...
