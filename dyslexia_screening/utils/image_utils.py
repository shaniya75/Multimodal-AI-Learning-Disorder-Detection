"""
Image loading, validation, and preprocessing utilities for the handwriting
classification pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError


class ImageValidationError(Exception):
    """Raised when an uploaded/loaded image fails validation."""


@dataclass
class ImagePreprocessingResult:
    image: Image.Image
    original_size: Tuple[int, int]
    final_size: Tuple[int, int]
    was_cropped: bool


def load_image(file_path_or_buffer) -> Image.Image:
    """Load an image from a path or file-like object."""
    try:
        image = Image.open(file_path_or_buffer)
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError(f"Could not read image: {exc}") from exc
    return image


def validate_image(image: Image.Image, min_dimension: int = 20) -> None:
    """Basic sanity checks on an image before preprocessing."""
    width, height = image.size
    if width < min_dimension or height < min_dimension:
        raise ImageValidationError(
            f"Image is too small ({width}x{height}). "
            f"Minimum dimension is {min_dimension}px."
        )

    arr = np.array(image.convert("L"))
    if arr.std() < 1.0:
        raise ImageValidationError("Image appears to be blank (near-uniform pixel values).")


def convert_to_rgb(image: Image.Image) -> Image.Image:
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def autocrop_borders(image: Image.Image, threshold: int = 250) -> Tuple[Image.Image, bool]:
    """Crop large uniform (near-white) borders around handwriting content.

    Uses a simple threshold-based bounding-box crop. If no meaningful
    content region is detected, the original image is returned unchanged.
    """
    gray = np.array(image.convert("L"))
    mask = gray < threshold  # non-background pixels

    if not mask.any():
        return image, False

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Guard against cropping away (almost) everything.
    height, width = gray.shape
    if (rmax - rmin) < 0.05 * height or (cmax - cmin) < 0.05 * width:
        return image, False

    cropped = image.crop((int(cmin), int(rmin), int(cmax) + 1, int(rmax) + 1))
    return cropped, True


def resize_image(image: Image.Image, target_size: int) -> Image.Image:
    return image.resize((target_size, target_size), Image.BILINEAR)


def preprocess_handwriting(
    file_path_or_buffer,
    target_size: int = 224,
    autocrop: bool = True,
) -> ImagePreprocessingResult:
    """Full preprocessing pipeline for a handwriting image.

    Steps: load -> validate -> convert to RGB -> autocrop borders (optional)
    -> resize. Normalization to model-ready tensors happens separately in
    models/handwriting/preprocessing.py using the same target_size /
    normalization statistics used at training time.
    """
    image = load_image(file_path_or_buffer)
    validate_image(image)
    original_size = image.size

    image = convert_to_rgb(image)

    was_cropped = False
    if autocrop:
        image, was_cropped = autocrop_borders(image)

    image = resize_image(image, target_size)

    return ImagePreprocessingResult(
        image=image,
        original_size=original_size,
        final_size=image.size,
        was_cropped=was_cropped,
    )


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
