# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Static-only overlay stubs are not Buck Python library targets.
# @lint-ignore-every AUTODEPS2

"""Shape-preserving libdevice operations used by Triton's tutorials."""

from shape_extensions import IntTuple
from triton.language import tensor

def asin[Tile: IntTuple](value: tensor[Tile]) -> tensor[Tile]: ...
