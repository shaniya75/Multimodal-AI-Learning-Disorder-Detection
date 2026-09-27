import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.speech.speech_features import (
    ReadingAlignmentCounts,
    compute_alignment_rates,
    compute_wcpm,
    FEATURE_COLUMNS,
)
from utils.validation import ModelNotTrainedError, require_files


def test_compute_alignment_rates_basic():
    counts = ReadingAlignmentCounts(
        total_reference_words=100, correct_words=90, substitutions=5, deletions=5, insertions=2
    )
    rates = compute_alignment_rates(counts)
    assert rates["word_accuracy"] == 0.9
    assert rates["substitution_rate"] == 0.05
    assert rates["deletion_rate"] == 0.05
    assert rates["miscue_rate"] == 0.10  # substitutions+deletions / total


def test_compute_alignment_rates_zero_division_safe():
    counts = ReadingAlignmentCounts(
        total_reference_words=0, correct_words=0, substitutions=0, deletions=0, insertions=0
    )
    rates = compute_alignment_rates(counts)
    assert rates["word_accuracy"] == 0.0


def test_compute_wcpm():
    wcpm = compute_wcpm(correct_words=60, reading_time_seconds=60)
    assert wcpm == 60.0


def test_feature_columns_no_dyslexia_label_leakage():
    # Ensures the reading-performance feature schema never smuggles in a
    # dyslexia-labeled column at this development stage.
    forbidden = {"dyslexia", "dyslexic", "diagnosis"}
    for col in FEATURE_COLUMNS:
        assert not any(term in col.lower() for term in forbidden)


def test_require_files_raises_when_missing(tmp_path):
    missing_path = str(tmp_path / "does_not_exist.pkl")
    with pytest.raises(ModelNotTrainedError):
        require_files([missing_path], "Speech model has not been trained yet.")


def test_require_files_passes_when_present(tmp_path):
    existing = tmp_path / "exists.pkl"
    existing.write_text("dummy")
    require_files([str(existing)], "Speech model has not been trained yet.")
