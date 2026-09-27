from __future__ import annotations

import json
import os

import joblib
import numpy as np

from utils.validation import (
    ModelNotTrainedError,
)


class SpeechPredictor:
    """
    Loads the trained speech classifier and performs
    prediction using Wav2Vec2 embeddings.

    The classifier receives only the 768 Wav2Vec2
    features used during training.
    """

    # =====================================================
    # Initialization
    # =====================================================

    def __init__(
        self,
        artifacts_dir: str,
    ):

        self.artifacts_dir = artifacts_dir

        self.model_path = os.path.join(
            artifacts_dir,
            "speech_classifier.pkl",
        )

        self.mapping_path = os.path.join(
            artifacts_dir,
            "class_mapping.json",
        )

        self.feature_columns_path = os.path.join(
            artifacts_dir,
            "feature_columns.json",
        )

        self.model = None

        self.model_name = None

        self.class_mapping = {}

        self.feature_columns = []

        self._load_model()

    # =====================================================
    # Load trained model
    # =====================================================

    def _load_model(self):

        if not os.path.exists(
            self.model_path
        ):

            return

        saved = joblib.load(
            self.model_path
        )

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        if isinstance(
            saved,
            dict
        ):

            self.model = saved.get(
                "model"
            )

            self.model_name = saved.get(
                "model_name"
            )

            self.feature_columns = saved.get(
                "feature_columns",
                []
            )

            self.class_mapping = saved.get(
                "class_mapping",
                {}
            )

        else:

            # Backward compatibility if the model
            # was saved directly instead of as a dictionary.

            self.model = saved

        # -------------------------------------------------
        # Load class mapping if not inside model file
        # -------------------------------------------------

        if (
            not self.class_mapping
            and os.path.exists(
                self.mapping_path
            )
        ):

            with open(
                self.mapping_path,
                "r",
                encoding="utf-8",
            ) as f:

                self.class_mapping = json.load(
                    f
                )

        # -------------------------------------------------
        # Load feature columns if not inside model file
        # -------------------------------------------------

        if (
            not self.feature_columns
            and os.path.exists(
                self.feature_columns_path
            )
        ):

            with open(
                self.feature_columns_path,
                "r",
                encoding="utf-8",
            ) as f:

                self.feature_columns = json.load(
                    f
                )

    # =====================================================
    # Check whether model exists
    # =====================================================

    def is_trained(self) -> bool:

        return self.model is not None

    # =====================================================
    # Predict using Wav2Vec2 embedding
    # =====================================================

    def predict_embedding(
        self,
        embedding: np.ndarray,
    ) -> dict:

        if not self.is_trained():

            raise ModelNotTrainedError(
                "Speech classifier has not been trained."
            )

        # -------------------------------------------------
        # Convert to NumPy array
        # -------------------------------------------------

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        # -------------------------------------------------
        # Make sure it is 2D
        # -------------------------------------------------

        if embedding.ndim == 1:

            embedding = embedding.reshape(
                1,
                -1
            )

        # -------------------------------------------------
        # Validate feature count
        # -------------------------------------------------

        if self.feature_columns:

            expected_features = len(
                self.feature_columns
            )

            actual_features = embedding.shape[1]

            if actual_features != expected_features:

                raise ValueError(
                    "Wav2Vec2 feature dimension mismatch. "
                    f"Model expects {expected_features} "
                    f"features but received "
                    f"{actual_features}."
                )

        # -------------------------------------------------
        # Prediction
        # -------------------------------------------------

        prediction = self.model.predict(
            embedding
        )[0]

        # -------------------------------------------------
        # Convert prediction to integer class ID
        # -------------------------------------------------

        try:

            class_id = int(
                prediction
            )

        except (
            TypeError,
            ValueError,
        ):

            class_id = prediction

        # -------------------------------------------------
        # Convert class ID to label
        # -------------------------------------------------

        prediction_key = str(
            class_id
        )

        label = self.class_mapping.get(
            prediction_key,
            prediction_key,
        )

        # -------------------------------------------------
        # Result
        # -------------------------------------------------

        result = {

            "label": label,

            "class_id": class_id,
        }

        # -------------------------------------------------
        # Probability
        # -------------------------------------------------

        if hasattr(
            self.model,
            "predict_proba",
        ):

            probabilities = (
                self.model.predict_proba(
                    embedding
                )[0]
            )

            probability_dict = {}

            for index, probability in enumerate(
                probabilities
            ):

                probability_dict[
                    str(index)
                ] = float(
                    probability
                )

            result[
                "probabilities"
            ] = probability_dict

            result[
                "confidence"
            ] = float(
                np.max(
                    probabilities
                )
            )

        return result