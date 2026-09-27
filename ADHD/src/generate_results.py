"""
Performance Analysis for ADHD Eye-Tracking Classification

Evaluates the existing 13 eye-tracking features using:
    1. Support Vector Machine (RBF)
    2. Random Forest

Participant-level Stratified Group K-Fold cross-validation is used
so that recordings belonging to the same participant never appear
in both training and testing folds.

Outputs:
    ADHD/results/metrics/
    ADHD/results/figures/
    ADHD/results/predictions/
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = PROJECT_ROOT / "ADHD" / "results" / "features"
RESULTS_DIR = PROJECT_ROOT / "ADHD" / "results"

METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# EXISTING 13 FEATURES
# ============================================================

FEATURE_COLS = [
    "n_fixations",
    "fixation_rate_per_sec",
    "fix_duration_mean",
    "fix_duration_std",
    "fix_duration_median",
    "fix_dispersion_mean",
    "n_saccades",
    "saccade_rate_per_sec",
    "sacc_amplitude_mean",
    "sacc_amplitude_std",
    "sacc_peak_vel_mean",
    "sacc_duration_mean",
    "tracker_loss_prop",
]


# ============================================================
# FEATURE FILES
# ============================================================

VIDEO_FILES = [
    "features_Despicable_Me.csv",
    "features_Diary_of_a_Wimpy_Kid_Trailer.csv",
    "features_Fractals.csv",
    "features_The_Present.csv",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    frames = []

    for filename in VIDEO_FILES:

        path = FEATURE_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Feature file not found:\n{path}"
            )

        df = pd.read_csv(path)

        if "Patient_ID" not in df.columns:
            raise ValueError(
                f"'Patient_ID' column missing from {filename}"
            )

        if "label" not in df.columns:
            raise ValueError(
                f"'label' column missing from {filename}"
            )

        df["source_video"] = (
            filename
            .replace("features_", "")
            .replace(".csv", "")
        )

        frames.append(df)

    data = pd.concat(
        frames,
        ignore_index=True
    )

    return data


# ============================================================
# MODELS
# ============================================================

def create_models():

    models = {

        "SVM": Pipeline([
            (
                "impute",
                SimpleImputer(strategy="median")
            ),

            (
                "scale",
                StandardScaler()
            ),

            (
                "classifier",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    probability=True,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ]),

        "Random Forest": Pipeline([
            (
                "impute",
                SimpleImputer(strategy="median")
            ),

            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=5,
                    class_weight="balanced",
                    random_state=42
                )
            )
        ])
    }

    return models


# ============================================================
# PLOT CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    y_true,
    y_pred,
    model_name
):

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    im = ax.imshow(cm)

    ax.set_title(
        f"{model_name} - Confusion Matrix"
    )

    ax.set_xlabel(
        "Predicted Label"
    )

    ax.set_ylabel(
        "Actual Label"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Typical", "ADHD"]
    )

    ax.set_yticklabels(
        ["Typical", "ADHD"]
    )

    for i in range(2):
        for j in range(2):

            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    fig.colorbar(im, ax=ax)

    fig.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
    )

    fig.savefig(
        FIGURES_DIR /
        f"confusion_matrix_{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# PLOT ROC CURVE
# ============================================================

def save_roc_curve(
    y_true,
    probabilities,
    model_name
):

    fpr, tpr, _ = roc_curve(
        y_true,
        probabilities
    )

    auc_value = roc_auc_score(
        y_true,
        probabilities
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{model_name} (AUC = {auc_value:.3f})"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Chance"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        f"ROC Curve - {model_name}"
    )

    ax.legend(
        loc="lower right"
    )

    ax.grid(
        alpha=0.3
    )

    fig.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
    )

    fig.savefig(
        FIGURES_DIR /
        f"roc_curve_{filename}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    model,
    model_name
):

    if model_name != "Random Forest":
        return

    rf = model.named_steps[
        "classifier"
    ]

    importances = rf.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": importances
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=True
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    ax.set_xlabel(
        "Feature Importance"
    )

    ax.set_ylabel(
        "Eye-Tracking Feature"
    )

    ax.set_title(
        "Random Forest Feature Importance"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR /
        "feature_importance_random_forest.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    importance_df.sort_values(
        "Importance",
        ascending=False
    ).to_csv(
        METRICS_DIR /
        "feature_importance.csv",
        index=False
    )


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print()
    print("=" * 75)
    print("              ADHD PERFORMANCE ANALYSIS")
    print("=" * 75)
    print()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data = load_data()

    X = data[FEATURE_COLS]
    y = data["label"].astype(int)
    groups = data["Patient_ID"]

    print(
        f"Total recordings : {len(data)}"
    )

    print(
        f"Participants     : {groups.nunique()}"
    )

    print(
        f"ADHD recordings  : {int(y.sum())}"
    )

    print(
        f"Typical          : {int((y == 0).sum())}"
    )

    print()

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    models = create_models()

    all_metrics = []
    all_fold_results = []
    all_predictions = []

    # --------------------------------------------------------
    # Evaluate each model
    # --------------------------------------------------------

    for model_name, model in models.items():

        print("-" * 75)
        print(f"Evaluating {model_name}")
        print("-" * 75)

        y_true_all = []
        y_pred_all = []
        y_prob_all = []

        # --------------------------------------------
        # Five folds
        # --------------------------------------------

        for fold, (train_idx, test_idx) in enumerate(
            cv.split(
                X,
                y,
                groups=groups
            ),
            start=1
        ):

            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]

            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            probabilities = model.predict_proba(
                X_test
            )[:, 1]

            y_true_all.extend(
                y_test.tolist()
            )

            y_pred_all.extend(
                predictions.tolist()
            )

            y_prob_all.extend(
                probabilities.tolist()
            )

            fold_accuracy = accuracy_score(
                y_test,
                predictions
            )

            fold_precision = precision_score(
                y_test,
                predictions,
                zero_division=0
            )

            fold_recall = recall_score(
                y_test,
                predictions,
                zero_division=0
            )

            fold_f1 = f1_score(
                y_test,
                predictions,
                zero_division=0
            )

            fold_auc = roc_auc_score(
                y_test,
                probabilities
            )

            all_fold_results.append({
                "Model": model_name,
                "Fold": fold,
                "Accuracy": fold_accuracy,
                "Precision": fold_precision,
                "Recall": fold_recall,
                "F1": fold_f1,
                "ROC_AUC": fold_auc
            })

            # Save OOF predictions
            for idx, prediction, probability in zip(
                test_idx,
                predictions,
                probabilities
            ):

                all_predictions.append({
                    "Model": model_name,
                    "Patient_ID": data.iloc[idx]["Patient_ID"],
                    "Video": data.iloc[idx]["source_video"],
                    "Actual": int(data.iloc[idx]["label"]),
                    "Prediction": int(prediction),
                    "ADHD_Probability": probability,
                    "Fold": fold
                })

        # ----------------------------------------------------
        # Overall out-of-fold performance
        # ----------------------------------------------------

        y_true_all = np.array(
            y_true_all
        )

        y_pred_all = np.array(
            y_pred_all
        )

        y_prob_all = np.array(
            y_prob_all
        )

        accuracy = accuracy_score(
            y_true_all,
            y_pred_all
        )

        precision = precision_score(
            y_true_all,
            y_pred_all,
            zero_division=0
        )

        recall = recall_score(
            y_true_all,
            y_pred_all,
            zero_division=0
        )

        f1 = f1_score(
            y_true_all,
            y_pred_all,
            zero_division=0
        )

        auc_value = roc_auc_score(
            y_true_all,
            y_prob_all
        )

        all_metrics.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "ROC_AUC": auc_value
        })

        print(
            f"Accuracy  : {accuracy:.3f}"
        )

        print(
            f"Precision : {precision:.3f}"
        )

        print(
            f"Recall    : {recall:.3f}"
        )

        print(
            f"F1-score  : {f1:.3f}"
        )

        print(
            f"ROC-AUC   : {auc_value:.3f}"
        )

        print()

        # ----------------------------------------------------
        # Figures
        # ----------------------------------------------------

        save_confusion_matrix(
            y_true_all,
            y_pred_all,
            model_name
        )

        save_roc_curve(
            y_true_all,
            y_prob_all,
            model_name
        )

        # ----------------------------------------------------
        # Train full model for feature importance
        # ----------------------------------------------------

        model.fit(
            X,
            y
        )

        save_feature_importance(
            model,
            model_name
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    metrics_df = pd.DataFrame(
        all_metrics
    )

    fold_df = pd.DataFrame(
        all_fold_results
    )

    predictions_df = pd.DataFrame(
        all_predictions
    )

    metrics_df.to_csv(
        METRICS_DIR /
        "performance_summary.csv",
        index=False
    )

    fold_df.to_csv(
        METRICS_DIR /
        "fold_results.csv",
        index=False
    )

    predictions_df.to_csv(
        PREDICTIONS_DIR /
        "out_of_fold_predictions.csv",
        index=False
    )

    # ========================================================
    # PRINT FINAL TABLE
    # ========================================================

    print("=" * 75)
    print("                 FINAL PERFORMANCE")
    print("=" * 75)
    print()

    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}"
        )
    )

    print()
    print("=" * 75)
    print("Results saved to:")
    print(
        f"  {METRICS_DIR}"
    )
    print(
        f"  {FIGURES_DIR}"
    )
    print(
        f"  {PREDICTIONS_DIR}"
    )
    print("=" * 75)
    print()


if __name__ == "__main__":
    main()