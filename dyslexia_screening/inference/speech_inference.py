"""
End-to-end inference for a single WAV file.

Pipeline:
    WAV
      ↓
    Audio preprocessing
      ↓
    Wav2Vec2
      ↓
    768-dimensional speech embedding
      ↓
    Trained speech classifier
      ↓
    Reading Difficulty / No Reading Difficulty

IMPORTANT:
    The live system does not calculate word_accuracy or create a new
    reading-outcome label. The trained classifier directly predicts the
    research-defined reading-performance category from the Wav2Vec2
    representation.

This is NOT a clinical dyslexia diagnosis.
"""
from __future__ import annotations

from typing import Dict

from utils.audio_utils import (
    preprocess_audio,
    AudioValidationError,
)

from utils.config_loader import (
    load_config,
    resolve_path,
)

from models.speech.wav2vec_model import (
    get_cached_embedder,
)

from models.speech.speech_predictor import (
    SpeechPredictor,
)

from utils.validation import (
    ModelNotTrainedError,
)


# =========================================================
# Run speech inference
# =========================================================

def run_speech_inference(
    audio_file_path: str,
) -> Dict:

    # -----------------------------------------------------
    # Load configuration
    # -----------------------------------------------------

    config = load_config()

    sample_rate = config["speech"]["sample_rate"]

    model_name = config["speech"]["model_name"]

    artifacts_dir = resolve_path(
        config["artifacts"]["speech"]
    )

    # -----------------------------------------------------
    # Audio preprocessing
    # -----------------------------------------------------

    try:

        audio_result = preprocess_audio(
            audio_file_path,
            target_sample_rate=sample_rate,
            min_duration_seconds=config["speech"][
                "min_duration_seconds"
            ],
            max_duration_seconds=config["speech"][
                "max_duration_seconds"
            ],
        )

    except AudioValidationError as exc:

        return {
            "error": str(exc)
        }

    # -----------------------------------------------------
    # Load Wav2Vec2
    # -----------------------------------------------------

    try:

        embedder = get_cached_embedder(
            model_name
        )

        embedding = embedder.extract_embedding(
            audio_result.waveform,
            audio_result.sample_rate,
        )

    except Exception as exc:

        return {
            "error": (
                "Unable to extract Wav2Vec2 "
                f"features: {exc}"
            )
        }

    # -----------------------------------------------------
    # Check embedding
    # -----------------------------------------------------

    if embedding is None:

        return {
            "error": (
                "Wav2Vec2 feature extraction "
                "returned no features."
            )
        }

    # -----------------------------------------------------
    # Load trained speech classifier
    # -----------------------------------------------------

    predictor = SpeechPredictor(
        artifacts_dir
    )

    if not predictor.is_trained():

        return {
            "error": (
                "Speech model has not been trained yet. "
                "Please run the speech training pipeline "
                "first."
            )
        }

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    try:

        prediction = predictor.predict_embedding(
            embedding
        )

    except ModelNotTrainedError as exc:

        return {
            "error": str(exc)
        }

    except Exception as exc:

        return {
            "error": (
                "Speech prediction failed: "
                f"{exc}"
            )
        }

    # -----------------------------------------------------
    # Return result
    # -----------------------------------------------------

    return {

        "prediction": prediction,

        "audio_duration": (
            audio_result.duration_seconds
        ),

        "sample_rate": (
            audio_result.sample_rate
        ),

        "embedding_dimension": int(
            len(embedding)
        ),
    }