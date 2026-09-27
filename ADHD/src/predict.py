"""
ADHD Eye-Tracking Prediction

Trains a Random Forest using the existing eye-tracking features
and generates participant-level ADHD risk predictions.

This script is intended for screening/demo purposes.
"""

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = PROJECT_ROOT / "ADHD" / "results" / "features"


# ============================================================
# EXISTING FEATURES
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
# DATASETS
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
            print(f"WARNING: Could not find {filename}")
            continue

        df = pd.read_csv(path)

        # Keep track of the video source
        if "video" not in df.columns:
            df["video"] = filename.replace(
                "features_", ""
            ).replace(".csv", "")

        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No feature CSV files were found."
        )

    data = pd.concat(frames, ignore_index=True)

    return data


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(X, y):

    model = Pipeline([
        (
            "impute",
            SimpleImputer(strategy="median")
        ),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=5,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    model.fit(X, y)

    return model


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 78)
    print("                 ADHD EYE-TRACKING PREDICTION")
    print("=" * 78)
    print()

    # --------------------------------------------------------
    # Load all recordings
    # --------------------------------------------------------

    data = load_data()

    print(f"Total recordings loaded : {len(data)}")
    print(f"Number of features      : {len(FEATURE_COLS)}")
    print()

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    X = data[FEATURE_COLS]
    y = data["label"].astype(int)

    print(
        f"Training data           : "
        f"ADHD={int(y.sum())}, "
        f"Typical={int((y == 0).sum())}"
    )

    print()

    # --------------------------------------------------------
    # Train Random Forest
    # --------------------------------------------------------

    model = train_model(X, y)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    # Find which probability column corresponds to ADHD
    classes = model.named_steps["clf"].classes_

    adhd_index = list(classes).index(1)

    adhd_probability = probabilities[:, adhd_index]

    # --------------------------------------------------------
    # Create result table
    # --------------------------------------------------------

    results = pd.DataFrame({
        "Patient_ID": data["Patient_ID"],
        "Video": data["video"],
        "Actual": data["label"].map({
            0: "Typical",
            1: "ADHD"
        }),
        "Prediction": pd.Series(predictions).map({
            0: "Typical",
            1: "ADHD Risk"
        }),
        "ADHD Probability": adhd_probability
    })

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("=" * 78)
    print("                         PREDICTIONS")
    print("=" * 78)
    print()

    header = (
        f"{'Recording':<48}"
        f"{'Actual':<12}"
        f"{'Prediction':<15}"
        f"{'Risk':>8}"
    )

    print(header)
    print("-" * 78)

    for _, row in results.iterrows():

        recording = (
            str(row["Patient_ID"]) +
            str(row["Video"])
        )

        # Keep the table aligned
        if len(recording) > 46:
            recording = recording[:46]

        actual = row["Actual"]
        prediction = row["Prediction"]
        risk = row["ADHD Probability"] * 100

        print(
            f"{recording:<48}"
            f"{actual:<12}"
            f"{prediction:<15}"
            f"{risk:>6.1f}%"
        )

    # --------------------------------------------------------
    # Overall training-set agreement
    # --------------------------------------------------------

    accuracy = (predictions == y.values).mean()

    print()
    print("=" * 78)
    print("                       SUMMARY")
    print("=" * 78)

    print()
    print(f"Total recordings       : {len(results)}")
    print(
        f"Predicted ADHD Risk    : "
        f"{sum(predictions == 1)}"
    )
    print(
        f"Predicted Typical      : "
        f"{sum(predictions == 0)}"
    )
    print(
        f"Training-set agreement : "
        f"{accuracy * 100:.1f}%"
    )

    # --------------------------------------------------------
    # Top Random Forest features
    # --------------------------------------------------------

    rf = model.named_steps["clf"]

    importance_df = pd.DataFrame({
        "Feature": FEATURE_COLS,
        "Importance": rf.feature_importances_
    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    print()
    print("=" * 78)
    print("                 TOP CONTRIBUTING FEATURES")
    print("=" * 78)
    print()

    for i, (_, row) in enumerate(
        importance_df.head(5).iterrows(),
        start=1
    ):

        print(
            f"{i}. "
            f"{row['Feature']:<30}"
            f"{row['Importance']:.3f}"
        )

    print()
    print("=" * 78)
    print("NOTE: Predictions are screening estimates, not clinical diagnoses.")
    print("=" * 78)
    print()


if __name__ == "__main__":
    main()
    