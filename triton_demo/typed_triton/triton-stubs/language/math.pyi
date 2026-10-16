# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# This stub is read only by Pyrefly's static-only Triton fixture overlay.
# @lint-ignore-every AUTODEPS2

from shape_extensions import IntTuple

from . import tensor

def exp2[Tile: IntTuple](value: tensor[Tile]) -> tensor[Tile]: ...
def log2[Tile: IntTuple](value: tensor[Tile]) -> tensor[Tile]: ...
