"""
Train a speech classifier for MPS oral-reading performance.

Target:
    Reading Difficulty
    No Reading Difficulty

IMPORTANT:
    This is NOT a dyslexia classifier.

MPS does not provide a clinical dyslexia label. Therefore, this script
creates a research-defined reading-performance screening label from the
MPS word-level annotations.

Current operational rule:
    reading_accuracy < 90%  -> Reading Difficulty
    reading_accuracy >= 90% -> No Reading Difficulty

The classifier itself uses Wav2Vec2 speech embeddings so that the model
does not directly receive the same word-accuracy/miscue features used
to construct the target.

Outputs:
    artifacts/speech/speech_classifier.pkl
    artifacts/speech/feature_columns.json
    artifacts/speech/class_mapping.json
    artifacts/speech/metrics.json
    artifacts/speech/predictions.csv
"""

from __future__ import annotations

import json
import logging
import os
import sys

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config, resolve_path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# RESEARCH SCREENING THRESHOLD
# ---------------------------------------------------------

READING_ACCURACY_THRESHOLD = 0.90


# ---------------------------------------------------------
# GROUPED TRAIN / TEST SPLIT
# ---------------------------------------------------------

def grouped_train_test_split(
    df: pd.DataFrame,
    group_col: str,
    test_size: float,
    random_state: int
):
    """
    Student-level split.

    The same student should not appear in both training and testing.
    """

    if group_col not in df.columns or df[group_col].isna().all():

        logger.warning(
            "No usable '%s' column for grouped split. "
            "Falling back to row-level split.",
            group_col
        )

        from sklearn.model_selection import train_test_split

        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=test_size,
            random_state=random_state
        )

        return train_idx, test_idx

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state
    )

    groups = df[group_col].fillna("unknown").astype(str)

    train_idx, test_idx = next(
        splitter.split(df, groups=groups)
    )

    return train_idx, test_idx


# ---------------------------------------------------------
# CREATE READING DIFFICULTY LABEL
# ---------------------------------------------------------

def create_reading_difficulty_label(df: pd.DataFrame) -> pd.DataFrame:
    """
        Creates a research-defined reading-performance label.

        Uses the word_accuracy feature already calculated from the
        MPS word-level annotations.

        Current operational rule:
            word_accuracy < 0.90  -> Reading Difficulty
            word_accuracy >= 0.90 -> No Reading Difficulty

        This is a research screening label, NOT a dyslexia diagnosis.
    """

    if "word_accuracy" not in df.columns:

        logger.error(
            "Required column 'word_accuracy' is missing from speech_features.csv."
        )

        logger.error(
            "Available columns are: %s",
            list(df.columns)
        )

        sys.exit(1)

    df = df.copy()

    # Remove missing values
    df = df.dropna(
        subset=["word_accuracy"]
    ).reset_index(drop=True)

    # Make sure accuracy is numeric
    df["word_accuracy"] = pd.to_numeric(
        df["word_accuracy"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["word_accuracy"]
    ).reset_index(drop=True)

    # Create research screening label
    df["reading_outcome"] = np.where(
        df["word_accuracy"] < READING_ACCURACY_THRESHOLD,
        "Reading Difficulty",
        "No Reading Difficulty"
    )

    return df


# ---------------------------------------------------------
# METRICS
# ---------------------------------------------------------

def compute_metrics(
    y_true,
    y_pred,
    y_proba,
    num_classes: int
) -> dict:

    metrics = {
        "accuracy": round(
            float(
                accuracy_score(y_true, y_pred)
            ),
            4
        ),

        "precision": round(
            float(
                precision_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                )
            ),
            4
        ),

        "recall": round(
            float(
                recall_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                )
            ),
            4
        ),

        "f1_score": round(
            float(
                f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0
                )
            ),
            4
        ),
    }

    # -----------------------------------------------------
    # ROC-AUC
    # -----------------------------------------------------

    try:

        if num_classes == 2:

            metrics["roc_auc"] = round(
                float(
                    roc_auc_score(
                        y_true,
                        y_proba[:, 1]
                    )
                ),
                4
            )

        elif num_classes > 2:

            metrics["roc_auc"] = round(
                float(
                    roc_auc_score(
                        y_true,
                        y_proba,
                        multi_class="ovr",
                        average="macro"
                    )
                ),
                4
            )

        else:

            metrics["roc_auc"] = None

    except ValueError:

        metrics["roc_auc"] = None

    # -----------------------------------------------------
    # CONFUSION MATRIX
    # -----------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    metrics["confusion_matrix"] = cm.tolist()

    # -----------------------------------------------------
    # SENSITIVITY / SPECIFICITY
    # -----------------------------------------------------

    if num_classes == 2 and cm.shape == (2, 2):

        tn, fp, fn, tp = cm.ravel()

        sensitivity = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0.0
        )

        metrics["sensitivity"] = round(
            float(sensitivity),
            4
        )

        metrics["specificity"] = round(
            float(specificity),
            4
        )

    else:

        metrics["sensitivity"] = None
        metrics["specificity"] = None

    return metrics


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    config = load_config()

    artifacts_dir = resolve_path(
        config["artifacts"]["speech"]
    )

    feature_table_path = os.path.join(
        artifacts_dir,
        "speech_features.csv"
    )

    # -----------------------------------------------------
    # CHECK FEATURE TABLE
    # -----------------------------------------------------

    if not os.path.exists(feature_table_path):

        logger.error(
            "Speech feature table not found at %s.",
            feature_table_path
        )

        logger.error(
            "Run extract_reading_features.py and "
            "extract_speech_embeddings.py first."
        )

        sys.exit(1)

    # -----------------------------------------------------
    # LOAD DATA
    # -----------------------------------------------------

    df = pd.read_csv(
        feature_table_path
    )

    logger.info(
        "Loaded speech feature table: %d rows, %d columns",
        len(df),
        len(df.columns)
    )

    # -----------------------------------------------------
    # CREATE READING DIFFICULTY LABEL
    # -----------------------------------------------------

    df = create_reading_difficulty_label(df)

    logger.info(
        "Created research screening target using %.0f%% "
        "reading-accuracy threshold.",
        READING_ACCURACY_THRESHOLD * 100
    )

    # -----------------------------------------------------
    # DISPLAY CLASS DISTRIBUTION
    # -----------------------------------------------------

    class_counts = df["reading_outcome"].value_counts()

    logger.info(
        "Reading outcome distribution:\n%s",
        class_counts.to_string()
    )

    # -----------------------------------------------------
    # FIND WAV2VEC2 FEATURES
    # -----------------------------------------------------

    wav2vec_cols = [
        c
        for c in df.columns
        if c.startswith("wav2vec_feature_")
    ]

    if not wav2vec_cols:

        logger.error(
            "No Wav2Vec2 features found."
        )

        logger.error(
            "Expected columns such as "
            "'wav2vec_feature_0', "
            "'wav2vec_feature_1', ..."
        )

        sys.exit(1)

    logger.info(
        "Using %d Wav2Vec2 speech features.",
        len(wav2vec_cols)
    )

    # -----------------------------------------------------
    # IMPORTANT:
    # Only use Wav2Vec2 embeddings for classification.
    #
    # Do NOT include:
    # correct_words
    # total_reference_words
    # substitutions
    # insertions
    # deletions
    # reading_accuracy
    #
    # because these were used to construct the target.
    # -----------------------------------------------------

    feature_columns = wav2vec_cols

    X = (
        df[feature_columns]
        .fillna(0.0)
        .values
        .astype(np.float32)
    )

    # -----------------------------------------------------
    # ENCODE TARGET
    # -----------------------------------------------------

    encoder = LabelEncoder()

    y = encoder.fit_transform(
        df["reading_outcome"].astype(str)
    )

    class_mapping = {
        str(i): label
        for i, label
        in enumerate(encoder.classes_)
    }

    num_classes = len(
        encoder.classes_
    )

    logger.info(
        "Classes: %s",
        class_mapping
    )

    # -----------------------------------------------------
    # CHECK BOTH CLASSES
    # -----------------------------------------------------

    if num_classes < 2:

        logger.error(
            "Only one reading-outcome class exists."
        )

        logger.error(
            "The classifier requires both "
            "'Reading Difficulty' and "
            "'No Reading Difficulty'."
        )

        sys.exit(1)

    # -----------------------------------------------------
    # GROUPED TRAIN / TEST SPLIT
    # -----------------------------------------------------

    test_size = config["speech"].get(
        "test_size",
        0.2
    )

    random_state = config["speech"].get(
        "random_state",
        42
    )

    train_idx, test_idx = grouped_train_test_split(
        df,
        "student_id",
        test_size,
        random_state
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    logger.info(
        "Training samples: %d",
        len(X_train)
    )

    logger.info(
        "Testing samples: %d",
        len(X_test)
    )

    # -----------------------------------------------------
    # CHECK TRAINING CLASSES
    # -----------------------------------------------------

    if len(np.unique(y_train)) < 2:

        logger.error(
            "Training split contains only one class."
        )

        logger.error(
            "Try changing the test_size/random_state "
            "or use a stratified group split."
        )

        sys.exit(1)

    # -----------------------------------------------------
    # CLASSIFIER
    # -----------------------------------------------------

    classifier_type = config["speech"].get(
        "classifier",
        "svm"
    )

    logger.info(
        "Classifier: %s",
        classifier_type
    )

    # -----------------------------------------------------
    # SVM
    # -----------------------------------------------------

    if classifier_type == "svm":

        model = SVC(
            kernel="rbf",
            probability=True,
            random_state=random_state,
            class_weight="balanced"
        )

    # -----------------------------------------------------
    # XGBOOST
    # -----------------------------------------------------

    elif classifier_type == "xgboost":

        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric=(
                "mlogloss"
                if num_classes > 2
                else "logloss"
            ),
            random_state=random_state
        )

    else:

        logger.error(
            "Unknown classifier type '%s'.",
            classifier_type
        )

        sys.exit(1)

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    logger.info(
        "Starting model training..."
    )

    model.fit(
        X_train,
        y_train
    )

    logger.info(
        "Model training completed."
    )

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    y_proba = model.predict_proba(
        X_test
    )

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    metrics = compute_metrics(
        y_test,
        y_pred,
        y_proba,
        num_classes
    )

    metrics["classifier"] = classifier_type

    metrics["num_train_samples"] = int(
        len(X_train)
    )

    metrics["num_test_samples"] = int(
        len(X_test)
    )

    metrics["num_classes"] = int(
        num_classes
    )

    metrics["reading_accuracy_threshold"] = (
        READING_ACCURACY_THRESHOLD
    )

    metrics["target_definition"] = (
        "Reading Difficulty if reading accuracy < 90%; "
        "No Reading Difficulty otherwise."
    )

    metrics["target_scope"] = (
        "Research-defined oral-reading performance screening; "
        "not a dyslexia diagnosis."
    )

    # -----------------------------------------------------
    # TEST PREDICTIONS TABLE
    # -----------------------------------------------------

    test_results = df.iloc[
        test_idx
    ].copy()

    test_results["actual_outcome"] = [
        encoder.inverse_transform(
            [value]
        )[0]
        for value in y_test
    ]

    test_results["predicted_outcome"] = [
        encoder.inverse_transform(
            [value]
        )[0]
        for value in y_pred
    ]

    test_results["prediction_correct"] = (
        test_results["actual_outcome"]
        ==
        test_results["predicted_outcome"]
    )

    predictions_path = os.path.join(
        artifacts_dir,
        "predictions.csv"
    )

    test_results.to_csv(
        predictions_path,
        index=False
    )

    # -----------------------------------------------------
    # SAVE MODEL
    # -----------------------------------------------------

    os.makedirs(
        artifacts_dir,
        exist_ok=True
    )

    model_path = os.path.join(
        artifacts_dir,
        "speech_classifier.pkl"
    )

    joblib.dump(
    {
        "model": model,
        "model_name": (
            "XGBoost"
            if classifier_type == "xgboost"
            else "SVM"
        ),
        "feature_columns": feature_columns,
        "class_mapping": class_mapping,
    },
    model_path
)

    # -----------------------------------------------------
    # SAVE FEATURE COLUMNS
    # -----------------------------------------------------

    with open(
        os.path.join(
            artifacts_dir,
            "feature_columns.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            feature_columns,
            f,
            indent=2
        )

    # -----------------------------------------------------
    # SAVE CLASS MAPPING
    # -----------------------------------------------------

    with open(
        os.path.join(
            artifacts_dir,
            "class_mapping.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            class_mapping,
            f,
            indent=2
        )

    # -----------------------------------------------------
    # SAVE METRICS
    # -----------------------------------------------------

    metrics_path = os.path.join(
        artifacts_dir,
        "metrics.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    # -----------------------------------------------------
    # PRINT FINAL RESULTS
    # -----------------------------------------------------

    logger.info(
        "=========================================="
    )

    logger.info(
        "SPEECH READING-DIFFICULTY RESULTS"
    )

    logger.info(
        "=========================================="
    )

    logger.info(
        "Accuracy     : %.4f",
        metrics["accuracy"]
    )

    logger.info(
        "Precision    : %.4f",
        metrics["precision"]
    )

    logger.info(
        "Recall       : %.4f",
        metrics["recall"]
    )

    logger.info(
        "F1 Score     : %.4f",
        metrics["f1_score"]
    )

    logger.info(
        "ROC-AUC      : %s",
        metrics["roc_auc"]
    )

    logger.info(
        "Sensitivity  : %s",
        metrics["sensitivity"]
    )

    logger.info(
        "Specificity  : %s",
        metrics["specificity"]
    )

    logger.info(
        "Confusion Matrix: %s",
        metrics["confusion_matrix"]
    )

    logger.info(
        "=========================================="
    )

    logger.info(
        "Model saved to: %s",
        model_path
    )

    logger.info(
        "Metrics saved to: %s",
        metrics_path
    )

    logger.info(
        "Predictions saved to: %s",
        predictions_path
    )


if __name__ == "__main__":
    main()