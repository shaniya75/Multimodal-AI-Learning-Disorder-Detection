import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config, resolve_path, get_project_root
from utils.validation import validate_student_id, validate_age, validate_grade


def test_load_config_has_required_sections():
    config = load_config()
    assert "app" in config
    assert "speech" in config
    assert "handwriting" in config
    assert "artifacts" in config


def test_resolve_path_is_absolute():
    resolved = resolve_path("data/mps")
    assert os.path.isabs(resolved)
    assert resolved.startswith(get_project_root())


def test_validate_student_id_strips_and_rejects_empty():
    assert validate_student_id("  abc123  ") == "abc123"
    with pytest.raises(ValueError):
        validate_student_id("   ")


def test_validate_age_range():
    assert validate_age(7) == 7
    with pytest.raises(ValueError):
        validate_age(2)
    with pytest.raises(ValueError):
        validate_age(19)
    with pytest.raises(ValueError):
        validate_age("not-a-number")


def test_validate_grade_rejects_empty():
    assert validate_grade(" Grade 3 ") == "Grade 3"
    with pytest.raises(ValueError):
        validate_grade("")
