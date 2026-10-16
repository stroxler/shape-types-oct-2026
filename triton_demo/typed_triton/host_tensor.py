"""Runtime marker for a checked Torch tensor shape and element strides."""

from shape_extensions import IntTuple


class Tensor[Shape: IntTuple, Strides: IntTuple]:
    """The original Torch tensor after checking its host-side layout."""
