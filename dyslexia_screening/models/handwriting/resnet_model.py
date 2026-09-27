"""
ResNet18-based handwriting classifier definition.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build a ResNet18 with its final fully-connected layer replaced to
    match the number of classes discovered in the handwriting dataset."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def load_resnet18_checkpoint(checkpoint_path: str, num_classes: int, device: str = "cpu") -> nn.Module:
    model = build_resnet18(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
