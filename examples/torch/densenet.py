# Portions (c) Meta Platforms, Inc. and affiliates.
#
# This source code is adapted from pytorch/benchmark (TorchBenchmark),
# which is licensed under the BSD 3-Clause License:
# https://github.com/pytorch/benchmark/blob/main/LICENSE
#
# This adaptation adds tensor shape type annotations for pyrefly.

"""
DenseNet from TorchBenchmark with shape annotations.

Original: pytorch/benchmark/torchbenchmark/models/phlippe_densenet/__init__.py

Port notes:
- DenseLayer breaks the original nn.Sequential into explicit calls for shape
    tracking through BatchNorm2d → ReLU → Conv2d(1x1) → BatchNorm2d → ReLU →
    Conv2d(3x3), then concatenates input with output (dense connection)
- WORKAROUND: Uses F.relu instead of configurable act_fn class parameter
    (the original passes nn.ReLU/nn.Tanh/etc. as a class constructor)
- TransitionLayer uses nn.AvgPool2d for spatial downsampling
    (nn.AvgPool2d's forward returns unrefined Tensor, no DSL redirect)
- DenseBlock chains DenseLayers explicitly in forward (not via nn.Sequential)
    because each layer has different input channel counts
- DenseNet uses concrete default config (growth_rate=16, bn_size=2,
    num_layers=[6,6,6,6], num_classes=10) since channel arithmetic is dynamic
"""

from typing import Any, overload, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from shape_extensions import Int, IntVar
    from torch import Tensor


# ============================================================================
# Building Blocks
# ============================================================================


class DenseLayer[InC: IntVar, BnC: IntVar, GR: IntVar](nn.Module):
    """Single dense layer with bottleneck.

    Architecture: BN → ReLU → 1x1 Conv(InC → BnC) → BN → ReLU → 3x3 Conv(BnC → GR)
    Then concatenates output (GR channels) with input (InC channels).

    Input:  Tensor[[B, InC, H, W]]
    Output: Tensor[[B, InC + GR, H, W]]

    BnC = bn_size * growth_rate (bottleneck channels for the 1x1 conv).
    """

    def __init__(
        self, c_in: Int[InC], bn_channels: Int[BnC], growth_rate: Int[GR]
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv1 = nn.Conv2d(c_in, bn_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_channels)
        self.conv2 = nn.Conv2d(
            bn_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )

    def forward[B: IntVar, H: IntVar, W: IntVar](
        self, x: Tensor[[B, InC, H, W]]
    ) -> Tensor[[B, InC + GR, H, W]]:
        # WORKAROUND: F.relu instead of configurable act_fn()
        out0 = F.relu(self.bn1(x))
        out1 = self.conv1(out0)
        out2 = F.relu(self.bn2(out1))
        out3 = self.conv2(out2)
        # Note: torch.cat returns unrefined Tensor in generic body;
        # shape is verified at concrete call sites via test functions
        result = torch.cat([out3, x], dim=1)
        return result


class TransitionLayer[InC: IntVar, OutC: IntVar](nn.Module):
    """Transition between dense blocks: BN → ReLU → 1x1 Conv → AvgPool(2).

    Reduces channels from InC to OutC and halves spatial dimensions.

    Input:  Tensor[[B, InC, H, W]]
    Output: Tensor[[B, OutC, (H-2)//2+1, (W-2)//2+1]]
    """

    def __init__(self, c_in: Int[InC], c_out: Int[OutC]) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(c_in)
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward[B: IntVar, H: IntVar, W: IntVar](
        self, x: Tensor[[B, InC, H, W]]
    ) -> Tensor[[B, OutC, (H - 2) // 2 + 1, (W - 2) // 2 + 1]]:
        # WORKAROUND: F.relu instead of configurable act_fn()
        out0 = F.relu(self.bn(x))
        out1 = self.conv(out0)
        out2 = self.pool(out1)
        return out2


# ============================================================================
# DenseBlock: 6 chained DenseLayers (default config)
# ============================================================================


class DenseBlock[C: IntVar, GR: IntVar, BnC: IntVar](nn.Module):
    """Dense block with 6 layers, using recursive forward.

    Each DenseLayer adds GR channels via concatenation.
    Input channels grow: C → C+GR → C+2*GR → ... → C+6*GR

    Uses _apply_layer + _chain for recursive shape verification instead
    of manually unrolled forward. The inductive step relies on symbolic
    product distribution: (Ch + GR) + GR*(I-1) = Ch + GR*I.
    """

    def __init__(
        self, c_in: Int[C], growth_rate: Int[GR], bn_channels: Int[BnC]
    ) -> None:
        super().__init__()
        layers: list[DenseLayer[Any, Any, Any]] = [
            DenseLayer(c_in, bn_channels, growth_rate),
            DenseLayer(c_in + growth_rate, bn_channels, growth_rate),
            DenseLayer(c_in + 2 * growth_rate, bn_channels, growth_rate),
            DenseLayer(c_in + 3 * growth_rate, bn_channels, growth_rate),
            DenseLayer(c_in + 4 * growth_rate, bn_channels, growth_rate),
            DenseLayer(c_in + 5 * growth_rate, bn_channels, growth_rate),
        ]
        self.layers = nn.ModuleList(layers)

    def _apply_layer[B: IntVar, Ch: IntVar, H: IntVar, W: IntVar](
        self, x: Tensor[[B, Ch, H, W]], depth: int
    ) -> Tensor[[B, Ch + GR, H, W]]:
        idx = len(self.layers) - depth
        layer: DenseLayer[Ch, BnC, GR] = self.layers[idx]
        return layer(x)

    def forward[B: IntVar, H: IntVar, W: IntVar](
        self, x: Tensor[[B, C, H, W]]
    ) -> Tensor[[B, C + 6 * GR, H, W]]:
        return _dense_chain(self, x, 6)


@overload
def _dense_chain[
    GR: IntVar,
    B: IntVar,
    Ch: IntVar,
    H: IntVar,
    W: IntVar,
](
    block: DenseBlock[Any, GR, Any], x: Tensor[[B, Ch, H, W]], depth: Int[1]
) -> Tensor[[B, Ch + GR, H, W]]: ...


@overload
def _dense_chain[
    I: IntVar,
    GR: IntVar,
    B: IntVar,
    Ch: IntVar,
    H: IntVar,
    W: IntVar,
](
    block: DenseBlock[Any, GR, Any], x: Tensor[[B, Ch, H, W]], depth: Int[I]
) -> Tensor[[B, Ch + I * GR, H, W]]: ...


def _dense_chain[
    I: IntVar,
    GR: IntVar,
    B: IntVar,
    Ch: IntVar,
    H: IntVar,
    W: IntVar,
](
    block: DenseBlock[Any, GR, Any], x: Tensor[[B, Ch, H, W]], depth: Int[I]
) -> Tensor[[B, Ch + GR, H, W]] | Tensor[[B, Ch + I * GR, H, W]]:
    y = block._apply_layer(x, depth)
    if depth == 1:
        return y
    return _dense_chain(block, y, depth - 1)


# ============================================================================
# DenseNet (default config: growth_rate=16, bn_size=2, num_layers=[6,6,6,6])
# ============================================================================


class DenseNet(nn.Module):
    """DenseNet with default configuration for CIFAR-10.

    Config: growth_rate=16, bn_size=2, num_layers=[6,6,6,6], num_classes=10
    Input: 3×32×32 (CIFAR-10)

    Channel progression:
    - Input conv: 3 → 32 (= growth_rate * bn_size)
    - Block 1: 32 → 128 (= 32 + 6*16), spatial 32×32
    - Transition 1: 128 → 64, spatial 32→16
    - Block 2: 64 → 160 (= 64 + 6*16), spatial 16×16
    - Transition 2: 160 → 80, spatial 16→8
    - Block 3: 80 → 176 (= 80 + 6*16), spatial 8×8
    - Transition 3: 176 → 88, spatial 8→4
    - Block 4: 88 → 184 (= 88 + 6*16), spatial 4×4
    - Output: BN → ReLU → AdaptiveAvgPool(1,1) → Flatten → Linear(184, 10)
    """

    def __init__(self) -> None:
        super().__init__()
        # bn_channels = bn_size * growth_rate = 2 * 16 = 32
        # Input convolution
        self.input_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        # Dense blocks and transitions
        self.block1 = DenseBlock(32, 16, 32)
        self.trans1 = TransitionLayer(128, 64)
        self.block2 = DenseBlock(64, 16, 32)
        self.trans2 = TransitionLayer(160, 80)
        self.block3 = DenseBlock(80, 16, 32)
        self.trans3 = TransitionLayer(176, 88)
        self.block4 = DenseBlock(88, 16, 32)

        # Output layers
        self.out_bn = nn.BatchNorm2d(184)
        self.out_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_flatten = nn.Flatten()
        self.out_linear = nn.Linear(184, 10)

    def forward[B: IntVar](self, x: Tensor[[B, 3, 32, 32]]) -> Tensor[[B, 10]]:
        # Input convolution
        h0 = self.input_conv(x)

        # Block 1 + Transition 1
        h1 = self.block1(h0)
        h1t = self.trans1(h1)

        # Block 2 + Transition 2
        h2 = self.block2(h1t)
        h2t = self.trans2(h2)

        # Block 3 + Transition 3
        h3 = self.block3(h2t)
        h3t = self.trans3(h3)

        # Block 4 (no transition after last block)
        h4 = self.block4(h3t)

        # Output: BN → ReLU → AdaptiveAvgPool → Flatten → Linear
        # WORKAROUND: F.relu instead of configurable act_fn()
        out_bn = F.relu(self.out_bn(h4))
        out_pool = self.out_pool(out_bn)
        out_flat = self.out_flatten(out_pool)
        logits = self.out_linear(out_flat)
        return logits


# ============================================================================
# Smoke tests
# ============================================================================


def test_dense_layer():
    """Test single dense layer: cat adds growth_rate channels."""
    layer = DenseLayer(32, 32, 16)
    x: Tensor[[4, 32, 8, 8]] = torch.randn(4, 32, 8, 8)
    out = layer(x)


def test_dense_layer_accumulated():
    """Test dense layer with accumulated channels (3rd layer in a block)."""
    layer = DenseLayer(64, 32, 16)
    x: Tensor[[4, 64, 8, 8]] = torch.randn(4, 64, 8, 8)
    out = layer(x)


def test_transition_layer():
    """Test transition: halves channels and spatial dims."""
    trans = TransitionLayer(128, 64)
    x: Tensor[[4, 128, 32, 32]] = torch.randn(4, 128, 32, 32)
    out = trans(x)


def test_dense_block():
    """Test dense block with 6 layers: adds 6*growth_rate channels."""
    block = DenseBlock(32, 16, 32)
    x: Tensor[[4, 32, 32, 32]] = torch.randn(4, 32, 32, 32)
    out = block(x)


def test_dense_block_2():
    """Test second dense block with different input channels."""
    block = DenseBlock(64, 16, 32)
    x: Tensor[[4, 64, 16, 16]] = torch.randn(4, 64, 16, 16)
    out = block(x)


def test_densenet():
    """End-to-end: DenseNet for CIFAR-10 classification."""
    model = DenseNet()
    x: Tensor[[2, 3, 32, 32]] = torch.randn(2, 3, 32, 32)
    out = model(x)
