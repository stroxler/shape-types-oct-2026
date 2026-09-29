"""A tiny four-head cross-attention layer; try unequal sequence lengths."""

import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.query = nn.Linear(32, 32)
        self.key = nn.Linear(32, 32)
        self.value = nn.Linear(32, 32)
        self.output = nn.Linear(32, 32)

    def forward(self, query, context):
        batch, query_length, _ = query.shape
        context_length = context.shape[1]

        q = self.query(query).reshape(query_length, batch, 4, 8).transpose(1, 2)
        k = self.key(context).reshape(batch, context_length, 4, 8).transpose(0, 2)
        v = self.value(context).reshape(batch, context_length, 4, 8).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / 8**0.5
        weights = F.softmax(scores, dim=-1)
        attended = weights @ v
        merged = attended.transpose(1, 2).reshape(batch, query_length, 32)
        return self.output(merged)
