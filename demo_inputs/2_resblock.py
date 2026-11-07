import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block1 = ConvBNReLU(channels, channels)
        self.block2 = ConvBNReLU(channels, channels)

    def forward(self, input):
        out = self.block1(input)
        out = self.block2(out)
        return out + input


model = ResBlock(8)
example_input = torch.randn(1, 8, 16, 16)
