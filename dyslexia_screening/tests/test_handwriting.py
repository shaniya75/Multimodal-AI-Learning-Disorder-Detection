import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.image_utils import (
    preprocess_handwriting,
    validate_image,
    ImageValidationError,
    autocrop_borders,
)
from training.handwriting_dataset import discover_handwriting_samples, get_class_names


@pytest.fixture
def sample_image(tmp_path):
    arr = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    path = tmp_path / "sample.png"
    img.save(path)
    return str(path)


@pytest.fixture
def blank_image(tmp_path):
    arr = np.full((300, 300, 3), 255, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    path = tmp_path / "blank.png"
    img.save(path)
    return str(path)


def test_preprocess_handwriting_resizes(sample_image):
    result = preprocess_handwriting(sample_image, target_size=224)
    assert result.final_size == (224, 224)


def test_preprocess_handwriting_rejects_blank_image(blank_image):
    with pytest.raises(ImageValidationError):
        preprocess_handwriting(blank_image, target_size=224)


def test_validate_image_rejects_tiny_image():
    tiny = Image.new("RGB", (5, 5), color=(0, 0, 0))
    with pytest.raises(ImageValidationError):
        validate_image(tiny, min_dimension=20)


def test_discover_handwriting_samples_reads_class_dirs(tmp_path):
    class_a = tmp_path / "class_a"
    class_b = tmp_path / "class_b"
    class_a.mkdir()
    class_b.mkdir()

    Image.new("RGB", (50, 50), color=(10, 10, 10)).save(class_a / "img1.png")
    Image.new("RGB", (50, 50), color=(20, 20, 20)).save(class_b / "img1.png")

    samples = discover_handwriting_samples(str(tmp_path))
    class_names = get_class_names(samples)

    assert len(samples) == 2
    assert class_names == ["class_a", "class_b"]


def test_discover_handwriting_samples_no_dyslexia_relabeling(tmp_path):
    # Class names must come straight from directory names, never rewritten.
    class_dir = tmp_path / "grade_2_sample"
    class_dir.mkdir()
    Image.new("RGB", (50, 50), color=(5, 5, 5)).save(class_dir / "img1.png")

    samples = discover_handwriting_samples(str(tmp_path))
    assert samples[0].class_name == "grade_2_sample"


def test_discover_handwriting_samples_nested_content_type_layout(tmp_path):
    # dyslexic/numbers, dyslexic/letters, dyslexic/paragraphs, and the same
    # under non_dyslexic — the class label is the top-level folder, and the
    # sub-folder name is captured separately as content_type.
    for label in ["dyslexic", "non_dyslexic"]:
        for content_type in ["numbers", "letters", "paragraphs"]:
            d = tmp_path / label / content_type
            d.mkdir(parents=True)
            Image.new("RGB", (50, 50), color=(1, 2, 3)).save(d / "sample1.png")

    samples = discover_handwriting_samples(str(tmp_path))
    class_names = get_class_names(samples)

    assert class_names == ["dyslexic", "non_dyslexic"]
    assert len(samples) == 6
    content_types = {s.content_type for s in samples}
    assert content_types == {"numbers", "letters", "paragraphs"}
    assert all(s.class_name in {"dyslexic", "non_dyslexic"} for s in samples)


def test_discover_handwriting_samples_content_type_filter(tmp_path):
    for label in ["dyslexic", "non_dyslexic"]:
        for content_type in ["numbers", "letters", "paragraphs"]:
            d = tmp_path / label / content_type
            d.mkdir(parents=True)
            Image.new("RGB", (50, 50), color=(1, 2, 3)).save(d / "sample1.png")

    samples = discover_handwriting_samples(str(tmp_path), content_type_filter=["paragraphs"])
    assert len(samples) == 2
    assert all(s.content_type == "paragraphs" for s in samples)


def test_discover_handwriting_samples_iam_style_groups_by_form_id(tmp_path):
    # Mimics IAM Words layout: dyslexic/words/<writer>/<form-id>/*.png
    # The filenames themselves ("a01-000u-00-00.png") carry no
    # student/writer regex match, so the parent folder name should be used
    # as the grouping proxy instead of leaving student_id as None.
    form_dir = tmp_path / "dyslexic" / "words" / "a01" / "a01-000u"
    form_dir.mkdir(parents=True)
    Image.new("RGB", (40, 40), color=(9, 9, 9)).save(form_dir / "a01-000u-00-00.png")
    Image.new("RGB", (40, 40), color=(9, 9, 9)).save(form_dir / "a01-000u-00-01.png")

    samples = discover_handwriting_samples(str(tmp_path))
    assert len(samples) == 2
    assert all(s.content_type == "words" for s in samples)
    assert all(s.student_id == "a01-000u" for s in samples)
    assert all(s.class_name == "dyslexic" for s in samples)
