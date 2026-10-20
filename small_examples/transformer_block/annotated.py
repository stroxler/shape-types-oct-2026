"""The buggy block with a shape-typed interface and inferred intermediates."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from shape_extensions import Int, IntVar
from torch import Tensor


class TransformerBlock[D: IntVar, H: IntVar](nn.Module):
    def __init__(self, dim: Int[D] = 64, heads: Int[H] = 8) -> None:
        super().__init__()
        if dim <= 0 or heads <= 0 or dim % heads != 0:
            raise ValueError(
                "dim must be positive and divisible by positive heads"
            )
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward[B: IntVar, T: IntVar](
        self, x: Tensor[[B, T, D]]
    ) -> Tensor[[B, T, D]]:
        b, t, _ = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, t, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        attn = F.softmax(q @ k.transpose(-1, -2), dim=-1) @ v
        merged = attn.transpose(1, 2).reshape(, self.dim)
        x = x + self.proj(merged)
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x
