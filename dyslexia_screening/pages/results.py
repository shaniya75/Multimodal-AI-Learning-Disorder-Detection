import json
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.config_loader import resolve_path
from utils.session_utils import has_student_info
from inference.speech_inference import run_speech_inference
from inference.handwriting_inference import run_handwriting_inference


def _load_metrics(artifacts_dir: str) -> dict | None:
    path = os.path.join(artifacts_dir, "metrics.json")

    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _metric_or_na(metrics: dict | None, key: str):
    if metrics is None:
        return "N/A"

    value = metrics.get(key)

    return "N/A" if value is None else value


def _get_class_names(class_mapping: dict, modality: str) -> list:
    """
    Convert the saved class mapping into display names.

    Supports both formats:

    {
        "0": "class_name",
        "1": "class_name"
    }

    and

    {
        "class_name": 0,
        "class_name": 1
    }
    """

    # ---------------------------------------------------------
    # First convert the saved mapping into index -> class name
    # ---------------------------------------------------------
    index_to_class = {}

    # Format:
    # {"0": "non_dyslexic", "1": "dyslexic"}
    if all(str(i) in class_mapping for i in range(len(class_mapping))):

        for i in range(len(class_mapping)):
            index_to_class[i] = str(class_mapping[str(i)])

    else:
        # Format:
        # {"non_dyslexic": 0, "dyslexic": 1}

        for class_name, class_index in class_mapping.items():
            try:
                index_to_class[int(class_index)] = str(class_name)
            except (ValueError, TypeError):
                continue

    # ---------------------------------------------------------
    # Convert model names into user-friendly names
    # ---------------------------------------------------------

    if modality == "Handwriting":

        display_map = {
            "dyslexic": "Dyslexia",
            "non_dyslexic": "Non-Dyslexia",
            "dyslexia": "Dyslexia",
            "non-dyslexic": "Non-Dyslexia",
            "non dyslexic": "Non-Dyslexia",
            "class 0": "Non-Dyslexia",
            "class 1": "Dyslexia",
            "0": "Non-Dyslexia",
            "1": "Dyslexia",
        }

    else:

        display_map = {
            "reading_difficulty": "Reading Difficulty",
            "no_reading_difficulty": "No Reading Difficulty",
            "reading difficulty": "Reading Difficulty",
            "no reading difficulty": "No Reading Difficulty",
            "class 0": "No Reading Difficulty",
            "class 1": "Reading Difficulty",
            "0": "No Reading Difficulty",
            "1": "Reading Difficulty",
        }

    class_names = []

    for i in sorted(index_to_class.keys()):

        original_name = index_to_class[i]

        normalized_name = original_name.strip().lower()

        display_name = display_map.get(
            normalized_name,
            original_name.replace("_", " ").title()
        )

        class_names.append(display_name)

    return class_names


def _format_probability_labels(probabilities: dict, modality: str) -> dict:
    """
    Convert probability dictionary keys into user-friendly labels.
    """

    if not probabilities:
        return {}

    if modality == "Handwriting":

        label_map = {
            "class_0": "Dyslexia",
            "class_1": "Non-Dyslexia",
    
        }

    else:

        label_map = {
            "class_0": "Dyslexia",
            "class_1": "Non-Dyslexia",
            "0": "No Reading Difficulty",
            "1": "Reading Difficulty",
            "class 0": "No Reading Difficulty",
            "class 1": "Reading Difficulty",
            "no_reading_difficulty": "No Reading Difficulty",
            "reading_difficulty": "Reading Difficulty",
            "no reading difficulty": "No Reading Difficulty",
            "reading difficulty": "Reading Difficulty",
        }

    formatted = {}

    for label, probability in probabilities.items():

        normalized_label = str(label).strip().lower()

        display_label = label_map.get(
            normalized_label,
            str(label)
        )

        formatted[display_label] = probability

    return formatted


def _format_prediction_label(prediction, modality: str) -> str:
    """
    Convert a model prediction into a user-friendly display label.
    """

    if prediction is None:
        return "Unknown"

    prediction_text = str(prediction).strip()

    normalized_prediction = prediction_text.lower()

    if modality == "Handwriting":

        label_map = {
            "class_0": "Dyslexia",
            "class_1": "Non-Dyslexia",

        }

    else:

        label_map = {
            "class_0": "Dyslexia",
            "class_1": "Non-Dyslexia",
            "0": "No Reading Difficulty",
            "1": "Reading Difficulty",
            "class 0": "No Reading Difficulty",
            "class 1": "Reading Difficulty",
            "no_reading_difficulty": "No Reading Difficulty",
            "reading_difficulty": "Reading Difficulty",
            "no reading difficulty": "No Reading Difficulty",
            "reading difficulty": "Reading Difficulty",
        }

    return label_map.get(
        normalized_prediction,
        prediction_text
    )


def _plot_confusion_matrix(cm, class_names, title):

    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=class_names,
            y=class_names,
            colorscale="Blues",
            showscale=True,
            text=cm,
            texttemplate="%{text}",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=380,
    )

    return fig


def _render_probabilities(probabilities: dict):

    df = pd.DataFrame(
        {
            "Class": list(probabilities.keys()),
            "Probability": list(probabilities.values()),
        }
    )

    st.bar_chart(
        df.set_index("Class")
    )


def render(config: dict) -> None:

    st.title("Results")

    if not has_student_info():

        st.warning(
            "Please complete Student Information first."
        )

        return

    st.caption(
        "Model outputs only — no combined score, no AI-generated interpretation, "
        "and no dyslexia diagnosis is produced at this development stage."
    )

    speech_artifacts = resolve_path(
        config["artifacts"]["speech"]
    )

    handwriting_artifacts = resolve_path(
        config["artifacts"]["handwriting"]
    )

    speech_result = None
    handwriting_result = None

    # =========================================================
    # SPEECH
    # =========================================================

    st.header("MPS Oral-Reading Model Result")

    if (
        st.session_state.speech_audio_path
        and os.path.exists(
            st.session_state.speech_audio_path
        )
    ):

        with st.spinner(
            "Running speech inference..."
        ):

            speech_result = run_speech_inference(
                st.session_state.speech_audio_path
            )

        st.session_state.speech_prediction = speech_result

    else:

        st.info(
            "No speech recording uploaded for this session."
        )

    if speech_result:

        if (
            "error" in speech_result
            and "prediction" not in speech_result
        ):

            st.warning(
                speech_result["error"]
            )

        else:

            if "error" in speech_result:

                st.warning(
                    speech_result["error"]
                )

            if "prediction" in speech_result:

                pred = speech_result["prediction"]

                predicted_class = _format_prediction_label(
                    pred.get(
                        "prediction",
                        pred.get(
                            "label",
                            "Unknown"
                        )
                    ),
                    "Speech",
                )

                st.metric(
                    "Predicted Class",
                    predicted_class
                )

                model_name = pred.get(
                    "model",
                    "Speech classifier"
                )

                st.caption(
                    f"Model: {model_name}"
                )

                probabilities = pred.get(
                    "probabilities"
                )

                if probabilities:

                    probabilities = _format_probability_labels(
                        probabilities,
                        "Speech"
                    )

                    _render_probabilities(
                        probabilities
                    )

            if "features" in speech_result:

                with st.expander(
                    "Extracted Reading Features"
                ):

                    st.json(
                        speech_result["features"]
                    )

        # -----------------------------------------------------
        # Speech evaluation metrics
        # -----------------------------------------------------

        speech_metrics = _load_metrics(
            speech_artifacts
        )

        if speech_metrics:

            with st.expander(
                "Speech Model Evaluation Metrics"
            ):

                st.write(
                    {
                        "Accuracy": _metric_or_na(
                            speech_metrics,
                            "accuracy",
                        ),
                        "Precision": _metric_or_na(
                            speech_metrics,
                            "precision",
                        ),
                        "Recall": _metric_or_na(
                            speech_metrics,
                            "recall",
                        ),
                        "F1 Score": _metric_or_na(
                            speech_metrics,
                            "f1_score",
                        ),
                        "ROC-AUC": _metric_or_na(
                            speech_metrics,
                            "roc_auc",
                        ),
                        "Sensitivity": _metric_or_na(
                            speech_metrics,
                            "sensitivity",
                        ),
                        "Specificity": _metric_or_na(
                            speech_metrics,
                            "specificity",
                        ),
                    }
                )

                cm = speech_metrics.get(
                    "confusion_matrix"
                )

                class_mapping_path = os.path.join(
                    speech_artifacts,
                    "class_mapping.json",
                )

                if (
                    cm
                    and os.path.exists(
                        class_mapping_path
                    )
                ):

                    with open(
                        class_mapping_path,
                        "r",
                        encoding="utf-8",
                    ) as f:

                        class_mapping = json.load(f)

                    class_names = _get_class_names(
                        class_mapping,
                        "Speech",
                    )

                    st.plotly_chart(
                        _plot_confusion_matrix(
                            cm,
                            class_names,
                            "Speech Confusion Matrix",
                        ),
                        use_container_width=True,
                    )

    st.divider()

    # =========================================================
    # HANDWRITING
    # =========================================================

    st.header("Handwriting Model Result")

    if (
        st.session_state.handwriting_image_path
        and os.path.exists(
            st.session_state.handwriting_image_path
        )
    ):

        with st.spinner(
            "Running handwriting inference..."
        ):

            handwriting_result = run_handwriting_inference(
                st.session_state.handwriting_image_path
            )

        st.session_state.handwriting_prediction = (
            handwriting_result
        )

    else:

        st.info(
            "No handwriting sample uploaded for this session."
        )

    if handwriting_result:

        if (
            "error" in handwriting_result
            and "prediction" not in handwriting_result
        ):

            st.warning(
                handwriting_result["error"]
            )

        else:

            pred = handwriting_result["prediction"]

            predicted_class = _format_prediction_label(
                pred.get(
                    "prediction",
                    "Unknown"
                ),
                "Handwriting",
            )

            st.metric(
                "Predicted Class",
                predicted_class
            )

            st.caption(
                f"Model: {pred.get('model', 'Handwriting classifier')}"
            )

            probabilities = pred.get(
                "probabilities",
                {}
            )

            probabilities = _format_probability_labels(
                probabilities,
                "Handwriting",
            )

            _render_probabilities(
                probabilities
            )

        # -----------------------------------------------------
        # Handwriting evaluation metrics
        # -----------------------------------------------------

        handwriting_metrics = _load_metrics(
            handwriting_artifacts
        )

        if handwriting_metrics:

            with st.expander(
                "Handwriting Model Evaluation Metrics"
            ):

                st.write(
                    {
                        "Accuracy": _metric_or_na(
                            handwriting_metrics,
                            "accuracy",
                        ),
                        "Precision": _metric_or_na(
                            handwriting_metrics,
                            "precision",
                        ),
                        "Recall": _metric_or_na(
                            handwriting_metrics,
                            "recall",
                        ),
                        "F1 Score": _metric_or_na(
                            handwriting_metrics,
                            "f1_score",
                        ),
                        "ROC-AUC": _metric_or_na(
                            handwriting_metrics,
                            "roc_auc",
                        ),
                        "Sensitivity": _metric_or_na(
                            handwriting_metrics,
                            "sensitivity",
                        ),
                        "Specificity": _metric_or_na(
                            handwriting_metrics,
                            "specificity",
                        ),
                    }
                )

                cm = handwriting_metrics.get(
                    "confusion_matrix"
                )

                class_mapping_path = os.path.join(
                    handwriting_artifacts,
                    "class_mapping.json",
                )

                if (
                    cm
                    and os.path.exists(
                        class_mapping_path
                    )
                ):

                    with open(
                        class_mapping_path,
                        "r",
                        encoding="utf-8",
                    ) as f:

                        class_mapping = json.load(f)

                    class_names = _get_class_names(
                        class_mapping,
                        "Handwriting",
                    )

                    st.plotly_chart(
                        _plot_confusion_matrix(
                            cm,
                            class_names,
                            "Handwriting Confusion Matrix",
                        ),
                        use_container_width=True,
                    )

    st.divider()

    # =========================================================
    # PREDICTION MATRIX
    # =========================================================

    st.header("Prediction Matrix")

    rows = []

    # ---------------------------------------------------------
    # Speech prediction
    # ---------------------------------------------------------

    if (
        speech_result
        and "prediction" in speech_result
    ):

        p = speech_result["prediction"]

        probabilities = _format_probability_labels(
            p.get(
                "probabilities",
                {}
            ),
            "Speech",
        )

        predicted_class = _format_prediction_label(
            p.get(
                "prediction",
                p.get(
                    "label",
                    "Unknown"
                )
            ),
            "Speech",
        )

        rows.append(
            {
                "Modality": "Speech",
                "Model": p.get(
                    "model",
                    "Speech classifier"
                ),
                "Predicted Class": predicted_class,
                "Class Probabilities": ", ".join(
                    f"{k}: {v}"
                    for k, v in probabilities.items()
                ),
            }
        )

    # ---------------------------------------------------------
    # Handwriting prediction
    # ---------------------------------------------------------

    if (
        handwriting_result
        and "prediction" in handwriting_result
    ):

        p = handwriting_result["prediction"]

        probabilities = _format_probability_labels(
            p.get(
                "probabilities",
                {}
            ),
            "Handwriting",
        )

        predicted_class = _format_prediction_label(
            p.get(
                "prediction",
                "Unknown"
            ),
            "Handwriting",
        )

        rows.append(
            {
                "Modality": "Handwriting",
                "Model": p.get(
                    "model",
                    "Handwriting classifier"
                ),
                "Predicted Class": predicted_class,
                "Class Probabilities": ", ".join(
                    f"{k}: {v}"
                    for k, v in probabilities.items()
                ),
            }
        )

    if rows:

        st.table(
            pd.DataFrame(rows)
        )

    else:

        st.caption(
            "No predictions available yet."
        )

    # =========================================================
    # EVALUATION MATRIX
    # =========================================================

    st.header("Evaluation Matrix")

    speech_metrics = _load_metrics(
        speech_artifacts
    )

    handwriting_metrics = _load_metrics(
        handwriting_artifacts
    )

    eval_rows = []

    metrics_to_display = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1_score"),
        ("ROC-AUC", "roc_auc"),
        ("Sensitivity", "sensitivity"),
        ("Specificity", "specificity"),
    ]

    for label, key in metrics_to_display:

        eval_rows.append(
            {
                "Metric": label,
                "Speech": _metric_or_na(
                    speech_metrics,
                    key,
                ),
                "Handwriting": _metric_or_na(
                    handwriting_metrics,
                    key,
                ),
            }
        )

    st.table(
        pd.DataFrame(eval_rows)
    )
