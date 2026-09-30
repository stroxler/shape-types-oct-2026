"""An unannotated cross-attention layer with two subtle axis mistakes."""

import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, model_width=32, num_heads=4):
        super().__init__()
        if model_width <= 0 or num_heads <= 0 or model_width % num_heads != 0:
            raise ValueError(
                "model_width must be positive and divisible by positive num_heads"
            )
        self.model_width = model_width
        self.num_heads = num_heads
        self.head_dim = model_width // num_heads
        self.query = nn.Linear(model_width, model_width)
        self.key = nn.Linear(model_width, model_width)
        self.value = nn.Linear(model_width, model_width)
        self.output = nn.Linear(model_width, model_width)

    def forward(self, query, context):
        batch, query_length, _ = query.shape
        context_length = context.shape[1]

        q = self.query(query).reshape(
            batch, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k = self.key(context).reshape(
            batch, context_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        v = self.value(context).reshape(
            batch, context_length, self.num_heads, self.head_dim
        ).transpose(1, 2)

        scores = q @ k.transpose(1, 2) / self.head_dim**0.5
        weights = F.softmax(scores, dim=-1)
        attended = weights @ v
        merged = attended.transpose(1, 2).reshape(
            query_length, batch, self.model_width
        )
        return self.output(merged)
