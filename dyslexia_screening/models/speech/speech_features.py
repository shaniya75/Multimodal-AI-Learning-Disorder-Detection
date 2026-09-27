"""
Reading-performance feature computation from MPS annotations and audio.

IMPORTANT SCOPE NOTE:
This module computes oral-reading performance metrics (accuracy, WCPM,
substitution/deletion/insertion rates, pause statistics, etc.). It does NOT
assign, threshold, or infer any dyslexia label. Any classification target
used downstream must come directly from a documented field in the MPS
dataset's own annotations (see training/extract_reading_features.py) —
never from an arbitrary threshold on these metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from utils.audio_utils import detect_pauses


@dataclass
class ReadingAlignmentCounts:
    """Word-level alignment counts between a reference transcript and the
    child's actual reading, as provided/derivable from MPS annotations."""
    total_reference_words: int
    correct_words: int
    substitutions: int
    deletions: int
    insertions: int


@dataclass
class ReadingPerformanceFeatures:
    reading_duration: float
    word_accuracy: float
    substitution_rate: float
    deletion_rate: float
    insertion_rate: float
    miscue_rate: float
    wcpm: float
    pause_count: int
    long_pause_count: int
    hesitation_count: int
    speech_duration: float

    def to_dict(self) -> Dict:
        return asdict(self)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator is None or denominator == 0:
        return 0.0
    return numerator / denominator


def compute_alignment_rates(counts: ReadingAlignmentCounts) -> Dict[str, float]:
    """Compute accuracy/substitution/deletion/insertion/miscue rates from
    raw alignment counts. All rates are normalized by total_reference_words,
    matching standard oral-reading fluency conventions."""
    total = counts.total_reference_words
    incorrect_words = counts.substitutions + counts.deletions

    return {
        "word_accuracy": round(_safe_div(counts.correct_words, total), 4),
        "substitution_rate": round(_safe_div(counts.substitutions, total), 4),
        "deletion_rate": round(_safe_div(counts.deletions, total), 4),
        "insertion_rate": round(_safe_div(counts.insertions, total), 4),
        "miscue_rate": round(_safe_div(incorrect_words, total), 4),
    }


def compute_wcpm(correct_words: int, reading_time_seconds: float) -> float:
    """Words Correct Per Minute."""
    reading_minutes = reading_time_seconds / 60.0
    return round(_safe_div(correct_words, reading_minutes), 2)


def compute_reading_performance_features(
    waveform,
    sample_rate: int,
    alignment_counts: ReadingAlignmentCounts,
    reading_duration_seconds: float,
    hesitation_count: Optional[int] = None,
) -> ReadingPerformanceFeatures:
    """Combine alignment-derived rates, WCPM, and audio-derived pause
    statistics into a single feature record for one reading sample."""
    rates = compute_alignment_rates(alignment_counts)
    wcpm = compute_wcpm(alignment_counts.correct_words, reading_duration_seconds)
    pause_stats = detect_pauses(waveform, sample_rate)

    return ReadingPerformanceFeatures(
        reading_duration=round(reading_duration_seconds, 3),
        word_accuracy=rates["word_accuracy"],
        substitution_rate=rates["substitution_rate"],
        deletion_rate=rates["deletion_rate"],
        insertion_rate=rates["insertion_rate"],
        miscue_rate=rates["miscue_rate"],
        wcpm=wcpm,
        pause_count=pause_stats["pause_count"],
        long_pause_count=pause_stats["long_pause_count"],
        hesitation_count=hesitation_count if hesitation_count is not None else pause_stats["pause_count"],
        speech_duration=round(waveform.shape[0] / float(sample_rate), 3),
    )


FEATURE_COLUMNS: List[str] = [
    "reading_duration",
    "word_accuracy",
    "substitution_rate",
    "deletion_rate",
    "insertion_rate",
    "miscue_rate",
    "wcpm",
    "pause_count",
    "long_pause_count",
    "hesitation_count",
    "speech_duration",
]
