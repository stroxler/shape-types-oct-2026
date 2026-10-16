"""Runtime annotation markers for the static Triton pointer view types."""

from shape_extensions import IntTuple


class InPointer[Shape: IntTuple, Strides: IntTuple]:
    """A checked host tensor presented as a readable Triton pointer."""


class OutPointer[Shape: IntTuple, Strides: IntTuple]:
    """A checked host tensor presented as a writable Triton pointer."""


class InOutPointer[Shape: IntTuple, Strides: IntTuple]:
    """A checked host tensor presented as a readable and writable pointer."""
