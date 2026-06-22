from __future__ import annotations

import torch
from torch import nn
from torchvision import models


def create_resnet18_binary(in_channels: int = 2, pretrained: bool = False) -> nn.Module:
    """Create a ResNet18 binary classifier for BF+nucleus input."""
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )

    with torch.no_grad():
        if pretrained:
            mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
            model.conv1.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))
        else:
            nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")

    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

