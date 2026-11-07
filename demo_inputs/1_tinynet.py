import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(16, 32)
        self.debug = False
        self.extra = nn.Linear(32, 32)
        self.left = nn.Linear(32, 16)
        self.right = nn.Linear(32, 16)
        self.mix = nn.Linear(32, 32)
        self.head = nn.Linear(32, 4)

    def forward(self, x):
        x = self.stem(x)
        x = F.relu(x)
        if self.debug:
            x = self.extra(x)
        a = self.left(x)
        b = self.right(x)
        x = torch.cat([a, b], dim=1)
        h = self.mix(x)
        x = h + x
        x = self.head(x)
        return x


model = TinyNet()
example_input = torch.randn(2, 16)
