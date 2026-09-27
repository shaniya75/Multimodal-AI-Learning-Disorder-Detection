"""
Shared validation helpers and the artifact-missing messaging used across
the app. Centralizing these strings keeps the "no fake predictions" rule
consistent everywhere a model might not yet be trained.
"""
from __future__ import annotations

import os
from typing import List


class ModelNotTrainedError(Exception):
    """Raised when inference is attempted but no trained artifact exists."""


SPEECH_NOT_TRAINED_MESSAGE = (
    "Speech model has not been trained yet. Please run the speech training "
    "pipeline first (`python training/train_speech.py`)."
)

HANDWRITING_NOT_TRAINED_MESSAGE = (
    "Handwriting model has not been trained yet. Please run the handwriting "
    "training pipeline first (`python training/train_handwriting.py`)."
)


def require_files(paths: List[str], not_trained_message: str) -> None:
    """Raise ModelNotTrainedError if any required artifact file is missing."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise ModelNotTrainedError(
            f"{not_trained_message} Missing files: {', '.join(missing)}"
        )


def validate_student_id(student_id: str) -> str:
    student_id = (student_id or "").strip()
    if not student_id:
        raise ValueError("Student ID cannot be empty.")
    if len(student_id) > 64:
        raise ValueError("Student ID is too long (max 64 characters).")
    return student_id


def validate_age(age) -> int:
    try:
        age_int = int(age)
    except (TypeError, ValueError):
        raise ValueError("Age must be a whole number.")
    if not (3 <= age_int <= 18):
        raise ValueError("Age must be between 3 and 18 for this screening tool.")
    return age_int


def validate_grade(grade: str) -> str:
    grade = (grade or "").strip()
    if not grade:
        raise ValueError("Grade/Class cannot be empty.")
    return grade
