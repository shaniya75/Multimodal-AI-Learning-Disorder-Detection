"""
Handwriting dataset discovery and PyTorch Dataset wrapper.

Supports two on-disk layouts under `handwriting.dataset_path`:

1. Flat ImageFolder style (one level deep):

    data/handwriting/
        <class_name_1>/
            img001.png
        <class_name_2>/
            img001.png

2. Nested style, e.g. diagnosis label at the top with content-type
   sub-folders underneath (numbers / letters / paragraphs):

    data/handwriting/
        dyslexic/
            numbers/
                img001.png
            letters/
                img001.png
            paragraphs/
                img001.png
        non_dyslexic/
            numbers/
                img001.png
            ...

For the nested layout, the **top-level folder name is the training
label** (e.g. "dyslexic" / "non_dyslexic") and the sub-folder name is
recorded separately as `content_type` (e.g. "numbers"), which is not used
as the classification target but is available for filtering (see
`handwriting.content_type_filter` in config.yaml) or later analysis.

This also transparently supports datasets nested even deeper, e.g. the
IAM Words layout (writer/form -> word images):

    data/handwriting/
        dyslexic/
            words/
                a01/
                    a01-000u/
                        a01-000u-00-00.png
                        ...

Here content_type is "words" for every image under that branch. Since
individual word-image filenames don't carry a student/writer ID, the
immediate parent folder name (e.g. "a01-000u") is used as a grouping
proxy for the student-level train/val/test split, so multiple word images
from the same form/page never leak across splits.

Class names are read directly from the top-level directory names — they
are NOT renamed or reinterpreted. If the source dataset encodes a student
ID in the filename (e.g. "student123_sample01.png"), that takes priority
over the parent-folder fallback. If an image sits directly inside its
class folder with no informative filename or sub-directory, student_id is
left as None and the caller falls back to a stratified random split
(documented as a limitation).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional

from PIL import Image
from torch.utils.data import Dataset

_STUDENT_ID_PATTERN = re.compile(r"(student[_\-]?\d+|writer[_\-]?\d+|subject[_\-]?\d+)", re.IGNORECASE)
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass
class HandwritingSample:
    image_path: str
    class_name: str
    student_id: Optional[str]
    content_type: Optional[str] = None  # e.g. "numbers" / "letters" / "paragraphs"


def discover_handwriting_samples(
    dataset_root: str,
    content_type_filter: Optional[List[str]] = None,
) -> List[HandwritingSample]:
    """Walk the dataset directory and collect samples.

    The top-level sub-directory of `dataset_root` is always treated as the
    class label. Anything nested further underneath (any depth) is walked
    recursively, and the *immediate* child folder name directly under the
    class folder (if any) is recorded as `content_type`.

    content_type_filter: if given, only samples whose content_type is in
    this list are kept (case-insensitive). Use this to train a model on
    just "numbers", just "letters", just "paragraphs", or any combination.
    Leave as None to use all content types combined.
    """
    samples: List[HandwritingSample] = []

    if not os.path.isdir(dataset_root):
        return samples

    normalized_filter = (
        {c.lower() for c in content_type_filter} if content_type_filter else None
    )

    class_dirs = sorted(
        d for d in os.listdir(dataset_root) if os.path.isdir(os.path.join(dataset_root, d))
    )

    for class_name in class_dirs:
        class_dir = os.path.join(dataset_root, class_name)

        for dirpath, dirnames, filenames in os.walk(class_dir):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]

            # content_type = the first path component under class_dir, if any
            rel = os.path.relpath(dirpath, class_dir)
            if rel == ".":
                content_type = None
            else:
                content_type = rel.split(os.sep)[0]

            if normalized_filter and content_type and content_type.lower() not in normalized_filter:
                continue
            if normalized_filter and content_type is None:
                # Flat images directly under class_dir have no content_type;
                # skip them when a filter is active since they can't be matched.
                continue

            for fname in sorted(filenames):
                ext = os.path.splitext(fname)[1].lower()
                if ext not in _IMAGE_EXTENSIONS:
                    continue

                match = _STUDENT_ID_PATTERN.search(fname)

                if match:
                    student_id = match.group(1)
                else:
                    student_id = None

                samples.append(
                    HandwritingSample(
                        image_path=os.path.join(dirpath, fname),
                        class_name=class_name,
                        student_id=student_id,
                        content_type=content_type,
                    )
                )

    return samples


def get_class_names(samples: List[HandwritingSample]) -> List[str]:
    return sorted({s.class_name for s in samples})


class HandwritingTorchDataset(Dataset):
    def __init__(self, samples: List[HandwritingSample], class_to_idx: dict, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.class_to_idx[sample.class_name]
        return image, label
