"""The same layer, with tensor shapes on its interface and key intermediates."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F
from shape_extensions import IntVar
from torch import Tensor


# With shape annotations, we can easily see where the bug is

class CrossAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(32, 32)
        self.key = nn.Linear(32, 32)
        self.value = nn.Linear(32, 32)
        self.output = nn.Linear(32, 32)

    def forward[B: IntVar, Q: IntVar, K: IntVar](
        self, query: Tensor[[B, Q, 32]], context: Tensor[[B, K, 32]]
    ) -> Tensor[[B, Q, 32]]:
        batch, query_length, _ = query.shape
        context_length = context.shape[1]

        q: Tensor[[B, 4, Q, 8]] = self.query(query).reshape(
            query_length, batch, 4, 8
        ).transpose(0, 2)
        k: Tensor[[B, 4, K, 8]] = self.key(context).reshape(
            batch, context_length, 4, 8
        ).transpose(0, 2)
        v: Tensor[[B, 4, K, 8]] = self.value(context).reshape(
            batch, context_length, 4, 8
        ).transpose(0, 2)

        scores: Tensor[[B, 4, Q, K]] = q @ k.transpose(-2, -1) / 8**0.5
        weights = F.softmax(scores, dim=-1)
        attended: Tensor[[B, 4, Q, 8]] = weights @ v
        merged = attended.transpose(1, 2).reshape(batch, query_length, 32)
        return self.output(merged)


# And once we see the bug, it's easy to fix

class FixedCrossAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(32, 32)
        self.key = nn.Linear(32, 32)
        self.value = nn.Linear(32, 32)
        self.output = nn.Linear(32, 32)

    def forward[B: IntVar, Q: IntVar, K: IntVar](
        self, query: Tensor[[B, Q, 32]], context: Tensor[[B, K, 32]]
    ) -> Tensor[[B, Q, 32]]:
        batch, query_length, _ = query.shape
        context_length = context.shape[1]

        q: Tensor[[B, 4, Q, 8]] = self.query(query).reshape(
            batch, query_length, 4, 8
        ).transpose(1, 2)
        k: Tensor[[B, 4, K, 8]] = self.key(context).reshape(
            batch, context_length, 4, 8
        ).transpose(1, 2)
        v: Tensor[[B, 4, K, 8]] = self.value(context).reshape(
            batch, context_length, 4, 8
        ).transpose(1, 2)

        scores: Tensor[[B, 4, Q, K]] = q @ k.transpose(-2, -1) / 8**0.5
        weights = F.softmax(scores, dim=-1)
        attended: Tensor[[B, 4, Q, 8]] = weights @ v
        merged = attended.transpose(1, 2).reshape(batch, query_length, 32)
        return self.output(merged)
