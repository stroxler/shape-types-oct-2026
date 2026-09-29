import torch
import torch.nn as nn
import torch.nn.functional as F
from shape_extensions import Int, IntVar
from torch import Tensor

# A basic multilayer perceptron, demonstrating Sequential support


class MLP[P: IntVar, M: IntVar, K: IntVar](nn.Module):
    def __init__(
        self,
        features: Int[P],
        hidden: Int[M],
        categories: Int[K],
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, categories),
        )

    def forward[N: IntVar](self, x: Tensor[[N, P]]) -> Tensor[[N, K]]:
        logits = self.net(x)
        return logits

    @staticmethod
    def run():
        model = MLP(4, 8, 3)  # P=4 features, M=8 hidden values, K=3 categories
        x = torch.randn(32, 4)
        labels = torch.randint(0, 3, (32,))
        logits = model(x)
        loss = F.cross_entropy(logits, labels)
        return float(loss)


# A tiny convnet in the style of the official MNIST example, generic over
# channels and image size. The convolutions shrink the spatial dimensions
# (H -> H - 4 -> ...), and adaptive pooling collapses to 1x1 so the
# classifier head has a fixed size for any image.


class TinyConvNet[CIn: IntVar, C1: IntVar, C2: IntVar](nn.Module):
    def __init__(self, in_channels: Int[CIn], width1: Int[C1], width2: Int[C2]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, width1, kernel_size=5)
        self.conv2 = nn.Conv2d(width1, width2, kernel_size=5)
        self.fc1 = nn.Linear(width2, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward[N: IntVar, H: IntVar, W: IntVar](
        self, x: Tensor[[N, CIn, H, W]]
    ) -> Tensor[[N, 10]]:
        conv1 = self.conv1(x)
        pooled1 = F.max_pool2d(F.relu(conv1), 2)
        conv2 = self.conv2(pooled1)
        pooled2 = F.max_pool2d(F.relu(conv2), 2)
        pooled3 = F.adaptive_avg_pool2d(pooled2, (1, 1))
        flat = torch.flatten(pooled3, 1)
        hidden = F.relu(self.fc1(flat))
        return self.fc2(hidden)

    @staticmethod
    def run():
        model = TinyConvNet(3, 16, 32)
        x = torch.randn(32, 3, 28, 28)
        labels = torch.randint(0, 10, (32,))
        logits = model(x)
        loss = F.cross_entropy(logits, labels)
        return float(loss)
