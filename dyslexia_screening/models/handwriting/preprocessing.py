"""
Shared image-to-tensor preprocessing for handwriting training and inference.
Using the exact same transform pipeline at train and inference time avoids
train/serve skew.
"""
from __future__ import annotations

import json
import os
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transform(image_size: int = 224, augment: bool = False) -> transforms.Compose:
    """Build the torchvision transform pipeline.

    augment=True adds light training-time augmentation (random rotation and
    slight translation), appropriate for handwriting images where small
    stroke variations shouldn't change the label.
    """
    ops = []

    if augment:
        ops.append(transforms.RandomRotation(degrees=5))
        ops.append(transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)))

    ops.extend(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


def image_to_tensor(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Convert a PIL image (already cropped/resized by image_utils if
    desired) into a normalized model-ready tensor with a batch dimension."""
    transform = build_transform(image_size=image_size, augment=False)
    tensor = transform(image)
    return tensor.unsqueeze(0)


def save_preprocessing_config(save_path: str, image_size: int, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> None:
    config = {"image_size": image_size, "mean": mean, "std": std}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def load_preprocessing_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)
