"""
Audio loading, validation, and preprocessing utilities for the speech
(oral-reading) pipeline.

The MPS branch of this application is a reading-performance analysis
pipeline, NOT a dyslexia classifier. These utilities only standardize audio
for feature extraction; they do not make any diagnostic claims.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import librosa

logger = logging.getLogger(__name__)


class AudioValidationError(Exception):
    """Raised when an uploaded/loaded audio file fails validation."""


@dataclass
class AudioProcessingResult:
    waveform: np.ndarray
    sample_rate: int
    duration_seconds: float
    original_sample_rate: int
    was_resampled: bool
    was_downmixed: bool


def load_audio(file_path: str) -> Tuple[np.ndarray, int]:
    """Load an audio file from disk.

    Raises
    ------
    AudioValidationError
        If the file cannot be read or is corrupted.
    """
    try:
        waveform, sample_rate = sf.read(file_path, always_2d=False)
    except Exception as exc:  # broad on purpose: many libsndfile errors possible
        raise AudioValidationError(f"Could not read audio file '{file_path}': {exc}") from exc

    if waveform is None or len(waveform) == 0:
        raise AudioValidationError(f"Audio file '{file_path}' is empty or corrupted.")

    return waveform.astype(np.float32), sample_rate


def validate_audio(
    waveform: np.ndarray,
    sample_rate: int,
    min_duration_seconds: float = 1.0,
    max_duration_seconds: float = 120.0,
) -> None:
    """Validate basic audio properties. Raises AudioValidationError on failure."""
    if waveform.size == 0:
        raise AudioValidationError("Audio contains no samples.")

    duration = waveform.shape[0] / float(sample_rate)

    if duration < min_duration_seconds:
        raise AudioValidationError(
            f"Recording is too short ({duration:.2f}s). "
            f"Minimum required is {min_duration_seconds:.2f}s."
        )
    if duration > max_duration_seconds:
        raise AudioValidationError(
            f"Recording is too long ({duration:.2f}s). "
            f"Maximum allowed is {max_duration_seconds:.2f}s."
        )

    if not np.isfinite(waveform).all():
        raise AudioValidationError("Audio contains invalid (NaN/Inf) samples.")

    peak = np.max(np.abs(waveform)) if waveform.size else 0.0
    if peak == 0.0:
        raise AudioValidationError("Audio appears to be silent (all zeros).")


def to_mono(waveform: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Downmix a multi-channel waveform to mono by averaging channels."""
    if waveform.ndim == 1:
        return waveform, False
    mono = np.mean(waveform, axis=1)
    return mono.astype(np.float32), True


def resample_audio(waveform: np.ndarray, orig_sr: int, target_sr: int) -> Tuple[np.ndarray, bool]:
    """Resample waveform to target_sr if needed."""
    if orig_sr == target_sr:
        return waveform, False
    resampled = librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
    return resampled.astype(np.float32), True


def normalize_audio(waveform: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Peak-normalize audio only when it would otherwise clip or is very
    quiet. Genuine pauses inside the recording are left untouched — this
    function only scales amplitude, it does not trim silence."""
    peak = np.max(np.abs(waveform))
    if peak == 0:
        return waveform
    if peak > 1.0 or peak < 0.1:
        return (waveform / peak) * target_peak
    return waveform


def preprocess_audio(
    file_path: str,
    target_sample_rate: int = 16000,
    min_duration_seconds: float = 1.0,
    max_duration_seconds: float = 120.0,
    normalize: bool = True,
) -> AudioProcessingResult:
    """Full preprocessing pipeline for a single audio file.

    Steps: load -> mono -> resample -> validate -> normalize (optional).

    Note: this function deliberately does NOT strip silence/pauses, since
    pause structure is potentially useful oral-reading information (see
    speech_features.py for pause-related feature extraction).
    """
    waveform, orig_sr = load_audio(file_path)
    waveform, was_downmixed = to_mono(waveform)
    waveform, was_resampled = resample_audio(waveform, orig_sr, target_sample_rate)

    validate_audio(
        waveform,
        target_sample_rate,
        min_duration_seconds=min_duration_seconds,
        max_duration_seconds=max_duration_seconds,
    )

    if normalize:
        waveform = normalize_audio(waveform)

    duration = waveform.shape[0] / float(target_sample_rate)

    return AudioProcessingResult(
        waveform=waveform,
        sample_rate=target_sample_rate,
        duration_seconds=duration,
        original_sample_rate=orig_sr,
        was_resampled=was_resampled,
        was_downmixed=was_downmixed,
    )


def detect_pauses(
    waveform: np.ndarray,
    sample_rate: int,
    top_db: float = 30.0,
    min_pause_seconds: float = 0.3,
    long_pause_seconds: float = 1.0,
) -> dict:
    """Estimate pause statistics using energy-based silence detection.

    This is a lightweight heuristic (librosa.effects.split), not a clinical
    measurement. It is intended for exploratory reading-performance analysis
    only.
    """
    intervals = librosa.effects.split(waveform, top_db=top_db)

    if len(intervals) <= 1:
        return {"pause_count": 0, "long_pause_count": 0, "total_pause_seconds": 0.0}

    pause_count = 0
    long_pause_count = 0
    total_pause_seconds = 0.0

    for i in range(1, len(intervals)):
        gap_samples = intervals[i][0] - intervals[i - 1][1]
        gap_seconds = gap_samples / float(sample_rate)
        if gap_seconds >= min_pause_seconds:
            pause_count += 1
            total_pause_seconds += gap_seconds
            if gap_seconds >= long_pause_seconds:
                long_pause_count += 1

    return {
        "pause_count": pause_count,
        "long_pause_count": long_pause_count,
        "total_pause_seconds": round(total_pause_seconds, 3),
    }
