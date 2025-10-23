"""
ResNet implementation for CIFAR-10/100 classification.

Implements a ResNet architecture optimized for 32x32 input images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic residual block with skip connection.

    Args:
        in_channel (int): Input channels
        out_channel (int): Output channels
        stride (int): Stride for first conv
    """

    def __init__(self, in_channel, out_channel, stride):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride

        self.conv1 = nn.Conv2d(
            in_channel, out_channel, kernel_size=3, padding=1, stride=stride, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(
            out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass through the residual block."""
        # First convolution with batch norm and ReLU
        x1 = self.relu(self.bn1(self.conv1(x)))
        # Second convolution with batch norm (no ReLU yet)
        x2 = self.bn2(self.conv2(x1))

        # Shortcut connection to match dimensions
        if self.stride != 1 or self.in_channel != self.out_channel:
            # Downsample spatial dimensions if stride > 1
            shortcut = x[:, :, :: self.stride, :: self.stride]
            # Pad channels if input and output channels differ
            pad = (0, 0, 0, 0, 0, self.out_channel - self.in_channel)
            shortcut = F.pad(shortcut, pad, mode="constant", value=0)
        else:
            shortcut = x

        # Residual connection: F(x) + x, then apply ReLU
        return self.relu(x2 + shortcut)


class ResNet(nn.Module):
    """
    ResNet for CIFAR-10/100 classification.

    Args:
        num_classes (int): Number of output classes (default: 10)
        n (int): Number of blocks per stage (default: 9 for ResNet-56)
    """

    def __init__(self, num_classes=10, n=9):
        super().__init__()
        self.n = n

        # First conv layer
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        # 3 Stages of Residual Blocks
        self.stage1 = self._make_layers(16, 16, num_blocks=self.n, stride=1)
        self.stage2 = self._make_layers(16, 32, num_blocks=self.n, stride=2)
        self.stage3 = self._make_layers(32, 64, num_blocks=self.n, stride=2)

        # Remaining layers for Classification
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, in_channel, out_channel, num_blocks, stride):
        """Create a sequence of residual blocks for a ResNet stage."""
        layers = []
        # First block (handles downsampling and channel change if needed)
        layers.append(BasicBlock(in_channel, out_channel, stride))
        # Remaining blocks (maintain dimensions)
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channel, out_channel, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the ResNet architecture."""
        # Initial convolution layer
        out = self.relu(self.bn(self.conv1(x)))

        # Three stages of residual blocks
        out = self.stage1(out)  # 16 channels, 32x32
        out = self.stage2(out)  # 32 channels, 16x16 (downsampled)
        out = self.stage3(out)  # 64 channels, 8x8 (downsampled)

        # Global average pooling and classification
        out = self.pool(out)  # 64 channels, 1x1
        out = torch.flatten(out, 1)  # Flatten to (batch_size, 64)
        out = self.fc(out)  # Final classification layer

        return out
