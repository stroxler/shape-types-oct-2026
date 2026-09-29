# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# This adaptation adds tensor shape type annotations for pyrefly.

"""
LearningToPaint Renderer FCN from TorchBenchmark with shape annotations.

Original: pytorch/benchmark/torchbenchmark/models/LearningToPaint/
  baseline/Renderer/model.py (FCN)

Port notes:
- FCN (Fully Connected Network) generates 128x128 brush stroke images from
  10-dimensional stroke parameters (position, color, width, etc.)
- Architecture: Linear MLP → reshape to [B, 16, 16, 16] → Conv2d chain with
  nn.PixelShuffle(2) for 2x upsampling → sigmoid → reshape to [B, 128, 128]
- Uses nn.PixelShuffle(2) for 2x spatial upsampling via channel rearrangement
- Uses x.view for conv-to-spatial reshape and final output reshape
- The original uses `1 - torch.sigmoid(x.view(-1, 128, 128))` — both
  torch.sigmoid and scalar __rsub__ are shape-preserving and work correctly
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from shape_extensions import IntVar
    from torch import Tensor


# ============================================================================
# Renderer FCN
# ============================================================================


class FCN(nn.Module):
    """Fully Connected Network for brush stroke rendering.

    Converts 10-dim stroke parameters into a 128x128 grayscale image.

    Architecture:
    - Linear MLP: 10 → 512 → 1024 → 2048 → 4096
    - Reshape: [B, 4096] → [B, 16, 16, 16]
    - Conv + PixelShuffle chain:
      - Conv2d(16, 32, 3, pad=1) + Conv2d(32, 32, 3, pad=1) + PixelShuffle(2)
        → [B, 8, 32, 32]
      - Conv2d(8, 16, 3, pad=1) + Conv2d(16, 16, 3, pad=1) + PixelShuffle(2)
        → [B, 4, 64, 64]
      - Conv2d(4, 8, 3, pad=1) + Conv2d(8, 4, 3, pad=1) + PixelShuffle(2)
        → [B, 1, 128, 128]
    - Sigmoid → reshape to [B, 128, 128]
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc4 = nn.Linear(2048, 4096)
        self.conv1 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(4, 8, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(8, 4, 3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward[B: IntVar](self, x: Tensor[[B, 10]]) -> Tensor[[B, 128, 128]]:
        # MLP
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))

        # Reshape to spatial: [B, 4096] → [B, 16, 16, 16]
        spatial = h4.view(x.size(0), 16, 16, 16)

        # Stage 1: Conv → Conv → PixelShuffle(2)
        # Conv2d(16→32, 3x3, pad=1): spatial-preserving
        s1_a = F.relu(self.conv1(spatial))
        # Conv2d(32→32, 3x3, pad=1): spatial-preserving
        s1_b = self.conv2(s1_a)
        # PixelShuffle(2): [B, 32, 16, 16] → [B, 8, 32, 32]
        s1_ps = self.pixel_shuffle(s1_b)

        # Stage 2: Conv → Conv → PixelShuffle(2)
        s2_a = F.relu(self.conv3(s1_ps))
        s2_b = self.conv4(s2_a)
        # PixelShuffle(2): [B, 16, 32, 32] → [B, 4, 64, 64]
        s2_ps = self.pixel_shuffle(s2_b)

        # Stage 3: Conv → Conv → PixelShuffle(2)
        s3_a = F.relu(self.conv5(s2_ps))
        s3_b = self.conv6(s3_a)
        # PixelShuffle(2): [B, 4, 64, 64] → [B, 1, 128, 128]
        s3_ps = self.pixel_shuffle(s3_b)

        # Sigmoid and reshape: [B, 1, 128, 128] → [B, 128, 128]
        # Original: 1 - torch.sigmoid(x.view(-1, 128, 128))
        s3_flat = torch.sigmoid(s3_ps.view(x.size(0), 128, 128))
        result = 1 - s3_flat
        return result


# ============================================================================
# Smoke tests
# ============================================================================


def test_fcn():
    """Test FCN: 10-dim stroke params → 128x128 image."""
    renderer = FCN()
    params: Tensor[[4, 10]] = torch.randn(4, 10)
    image = renderer(params)


def test_fcn_single():
    """Test FCN with single stroke."""
    renderer = FCN()
    params: Tensor[[1, 10]] = torch.randn(1, 10)
    image = renderer(params)
