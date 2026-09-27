"""
End-to-end inference for a single uploaded handwriting image.
"""
from __future__ import annotations

from typing import Dict

from utils.image_utils import preprocess_handwriting, ImageValidationError
from utils.config_loader import load_config, resolve_path
from models.handwriting.handwriting_predictor import HandwritingPredictor
from utils.validation import ModelNotTrainedError


def run_handwriting_inference(image_file_path_or_buffer) -> Dict:
    config = load_config()
    image_size = config["handwriting"]["image_size"]
    artifacts_dir = resolve_path(config["artifacts"]["handwriting"])

    try:
        result = preprocess_handwriting(image_file_path_or_buffer, target_size=image_size)
    except ImageValidationError as exc:
        return {"error": str(exc)}

    predictor = HandwritingPredictor(artifacts_dir)
    if not predictor.is_trained():
        return {
            "error": (
                "Handwriting model has not been trained yet. Please run the "
                "handwriting training pipeline first."
            )
        }

    try:
        prediction = predictor.predict(result.image)
    except ModelNotTrainedError as exc:
        return {"error": str(exc)}

    return {
        "prediction": prediction,
        "preprocessing": {
            "original_size": result.original_size,
            "final_size": result.final_size,
            "was_cropped": result.was_cropped,
        },
    }
