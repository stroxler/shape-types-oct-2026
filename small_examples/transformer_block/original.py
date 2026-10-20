"""An unannotated transformer block with a subtle merge mistake."""

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, dim=64, heads=8):
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

    def forward(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, t, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.softmax(q @ k.transpose(-1, -2), dim=-1) @ v
        x = x + self.proj(attn)
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x
