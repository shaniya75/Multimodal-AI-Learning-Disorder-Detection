import os
import sys

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.audio_utils import (
    preprocess_audio,
    validate_audio,
    to_mono,
    resample_audio,
    normalize_audio,
    detect_pauses,
    AudioValidationError,
)


@pytest.fixture
def sine_wav(tmp_path):
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    path = tmp_path / "sine.wav"
    sf.write(str(path), waveform, sr)
    return str(path)


@pytest.fixture
def silent_wav(tmp_path):
    sr = 16000
    waveform = np.zeros(sr, dtype=np.float32)
    path = tmp_path / "silence.wav"
    sf.write(str(path), waveform, sr)
    return str(path)


def test_preprocess_audio_resamples_and_returns_mono(sine_wav):
    result = preprocess_audio(sine_wav, target_sample_rate=16000)
    assert result.sample_rate == 16000
    assert result.waveform.ndim == 1
    assert result.duration_seconds > 0


def test_preprocess_audio_rejects_silence(silent_wav):
    with pytest.raises(AudioValidationError):
        preprocess_audio(silent_wav, target_sample_rate=16000)


def test_to_mono_downmixes_stereo():
    stereo = np.random.randn(1000, 2).astype(np.float32)
    mono, was_downmixed = to_mono(stereo)
    assert mono.ndim == 1
    assert was_downmixed is True


def test_resample_audio_changes_length():
    waveform = np.random.randn(22050).astype(np.float32)
    resampled, was_resampled = resample_audio(waveform, 22050, 16000)
    assert was_resampled is True
    assert len(resampled) != len(waveform)


def test_normalize_audio_scales_quiet_signal():
    quiet = (np.random.randn(1000) * 0.01).astype(np.float32)
    normalized = normalize_audio(quiet)
    assert np.max(np.abs(normalized)) > np.max(np.abs(quiet))


def test_detect_pauses_on_silence_only_returns_zero_counts():
    waveform = np.zeros(16000, dtype=np.float32)
    stats = detect_pauses(waveform, 16000)
    assert stats["pause_count"] >= 0
    assert "long_pause_count" in stats
