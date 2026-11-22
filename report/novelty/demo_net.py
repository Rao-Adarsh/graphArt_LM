import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.bn = nn.BatchNorm2d(c)

    def forward(self, x):
        y = self.bn(self.conv(x))
        return torch.relu(x + y)


class Net(nn.Module):
    def __init__(self, c=8, depth=2):
        super().__init__()
        self.stem = nn.Conv2d(3, c, 3, padding=1)
        self.blocks = nn.Sequential(*[Block(c) for _ in range(depth)])
        self.head = nn.Linear(c, 10)

    def forward(self, x):
        x = self.blocks(self.stem(x))
        return self.head(x.mean((2, 3)))


model = Net()
example_input = torch.randn(1, 3, 16, 16)
