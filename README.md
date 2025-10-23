# ResNet for CIFAR Classification

A ResNet implementation optimized for CIFAR-10/100 datasets (32x32 images).

## Features

- ResNet architecture with skip connections
- Configurable depth (ResNet-20, ResNet-56, etc.)
- Proper weight initialization
- CIFAR-optimized design

## Architecture

- **BasicBlock**: Residual block with skip connections
- **ResNet**: Three stages (16→32→64 channels) + classification head

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
import torch
from ResNet import ResNet

# Create ResNet-56 for CIFAR-10
model = ResNet(num_classes=10, n=9)

# Forward pass
x = torch.randn(32, 3, 32, 32)  # Batch of CIFAR images
output = model(x)  # Shape: (32, num_classes)
```

## Model Variants

| Model | n | Layers |
|-------|---|--------|
| ResNet-20 | 3 | 20 |
| ResNet-56 | 9 | 56 |
| ResNet-110 | 18 | 110 |