"""
ADHD eye-tracking feature extraction

This script takes the raw eye-tracking recordings from the
ECML-ADHD dataset and converts them into a table of
eye-movement features.

Pipeline:

Raw gaze data
    ↓
Basic cleaning
    ↓
Pixel coordinates → visual angle
    ↓
Fixation / saccade detection
    ↓
Eye-movement features
    ↓
Subject-video feature table
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# DATASET LOCATION
# ============================================================

# CHANGE THIS to the location of your downloaded ecml-ADHD folder.
DATASET_DIR = Path(r"C:\Users\shani\ecml-ADHD")


# The original ECML-ADHD repository contains the eye-movement
# preprocessing functions we use here.
UTILS_DIR = DATASET_DIR / "DNNmodel" / "DataGeneration"
sys.path.insert(0, str(UTILS_DIR))

from utils import pix2deg, get_sacc_fix_lists_dispersion


# ============================================================
# SCREEN INFORMATION
# ============================================================

SCREEN_PX_X = 800
SCREEN_PX_Y = 600

SCREEN_CM_X = 33.8
SCREEN_CM_Y = 27.0

DISTANCE_CM = 63.5


# ============================================================
# VIDEO NAMES
# ============================================================

VIDEO_NAME_MAP = {
    "video_Diary_of_a_Wimpy_Kid_Trailer":
        "Diary_of_a_Wimpy_Kid_Trailer",

    "video_Fractals":
        "Fractals",

    "video_Despicable_Me":
        "Despicable_Me",

    "video_The_Present":
        "The_Present",
}


# ============================================================
# LOAD RAW GAZE
# ============================================================

def load_raw_gaze(patient_id, video_name):
    """
    Load one participant's eye-tracking recording.
    """

    path = (
        DATASET_DIR
        / "Data"
        / "X_px"
        / f"X_px_{patient_id}_Video_{video_name}.npy"
    )

    if not path.exists():
        return None

    arr = np.load(path, allow_pickle=True)

    df = pd.DataFrame(
        arr,
        columns=[
            "video_frame",
            "time",
            "x_px",
            "y_px"
        ]
    )

    # Remove invalid frame identifiers
    df = df[df["video_frame"] >= 0].reset_index(drop=True)

    return df


# ============================================================
# PREPROCESSING
# ============================================================

def clean_and_convert(df, sampling_rate):
    """
    Clean gaze coordinates and convert pixels to
    degrees of visual angle.
    """

    df = df.copy()

    # (0, 0) represents tracker loss / missing gaze
    tracker_loss = (
        (df["x_px"] == 0) &
        (df["y_px"] == 0)
    )

    df.loc[
        tracker_loss,
        ["x_px", "y_px"]
    ] = np.nan

    tracker_loss_prop = tracker_loss.mean()

    # Pixel → degrees of visual angle
    df["x_deg"] = pix2deg(
        df["x_px"],
        SCREEN_PX_X,
        SCREEN_CM_X,
        DISTANCE_CM,
        adjust_origin=True
    )

    df["y_deg"] = pix2deg(
        df["y_px"],
        SCREEN_PX_Y,
        SCREEN_CM_Y,
        DISTANCE_CM,
        adjust_origin=True,
        reverse_axis=True
    )

    df.attrs["tracker_loss_prop"] = tracker_loss_prop
    df.attrs["sampling_rate"] = sampling_rate

    return df


# ============================================================
# FIXATION / SACCADE DETECTION
# ============================================================

def detect_events(df):
    """
    Identify fixations and saccades using the I-DT
    implementation supplied with the ECML-ADHD repository.
    """

    sampling_rate = df.attrs["sampling_rate"]

    events, event_df = get_sacc_fix_lists_dispersion(
        df["x_deg"].values,
        df["y_deg"].values,
        sampling_rate=sampling_rate
    )

    return events, event_df


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(df, events):

    sampling_rate = df.attrs["sampling_rate"]

    x = df["x_deg"].values
    y = df["y_deg"].values

    # --------------------------------------------------------
    # Fixations
    # --------------------------------------------------------

    fixation_durations = []
    fixation_dispersions = []

    for indices in events["fixations"]:

        if len(indices) < 2:
            continue

        duration_ms = (
            len(indices) / sampling_rate
        ) * 1000

        fx = x[indices]
        fy = y[indices]

        dispersion = (
            np.nanmax(fx) - np.nanmin(fx)
            +
            np.nanmax(fy) - np.nanmin(fy)
        )

        fixation_durations.append(duration_ms)
        fixation_dispersions.append(dispersion)

    # --------------------------------------------------------
    # Saccades
    # --------------------------------------------------------

    saccade_amplitudes = []
    saccade_peak_velocities = []
    saccade_durations = []

    for indices in events["saccades"]:

        if len(indices) < 2:
            continue

        duration_ms = (
            len(indices) / sampling_rate
        ) * 1000

        sx = x[indices]
        sy = y[indices]

        amplitude = np.sqrt(
            (sx[-1] - sx[0]) ** 2
            +
            (sy[-1] - sy[0]) ** 2
        )

        dx = np.diff(sx)
        dy = np.diff(sy)

        velocity = (
            np.sqrt(dx ** 2 + dy ** 2)
            * sampling_rate
        )

        peak_velocity = (
            np.nanmax(velocity)
            if len(velocity) > 0
            else np.nan
        )

        saccade_amplitudes.append(amplitude)
        saccade_peak_velocities.append(peak_velocity)
        saccade_durations.append(duration_ms)

    # --------------------------------------------------------
    # Recording duration
    # --------------------------------------------------------

    duration_sec = (
        len(df) / sampling_rate
        if sampling_rate
        else np.nan
    )

    # --------------------------------------------------------
    # Final feature dictionary
    # --------------------------------------------------------

    features = {

        "n_fixations":
            len(fixation_durations),

        "fixation_rate_per_sec":
            (
                len(fixation_durations) / duration_sec
                if duration_sec
                else np.nan
            ),

        "fix_duration_mean":
            (
                np.nanmean(fixation_durations)
                if fixation_durations
                else np.nan
            ),

        "fix_duration_std":
            (
                np.nanstd(fixation_durations)
                if fixation_durations
                else np.nan
            ),

        "fix_duration_median":
            (
                np.nanmedian(fixation_durations)
                if fixation_durations
                else np.nan
            ),

        "fix_dispersion_mean":
            (
                np.nanmean(fixation_dispersions)
                if fixation_dispersions
                else np.nan
            ),

        "n_saccades":
            len(saccade_amplitudes),

        "saccade_rate_per_sec":
            (
                len(saccade_amplitudes) / duration_sec
                if duration_sec
                else np.nan
            ),

        "sacc_amplitude_mean":
            (
                np.nanmean(saccade_amplitudes)
                if saccade_amplitudes
                else np.nan
            ),

        "sacc_amplitude_std":
            (
                np.nanstd(saccade_amplitudes)
                if saccade_amplitudes
                else np.nan
            ),

        "sacc_peak_vel_mean":
            (
                np.nanmean(saccade_peak_velocities)
                if saccade_peak_velocities
                else np.nan
            ),

        "sacc_duration_mean":
            (
                np.nanmean(saccade_durations)
                if saccade_durations
                else np.nan
            ),

        "tracker_loss_prop":
            df.attrs["tracker_loss_prop"],

        "n_samples":
            len(df),
    }

    return features


# ============================================================
# BUILD DATASET FOR ONE VIDEO
# ============================================================

def build_dataset(video_column):

    label_file = (
        DATASET_DIR
        / "Data"
        / "sub_info"
        / "sub_sel_classif.csv"
    )

    sub_info = pd.read_csv(
        label_file,
        sep="\t"
    )

    sub_info = sub_info[
        sub_info[video_column] == True
    ].reset_index(drop=True)

    video_name = VIDEO_NAME_MAP[video_column]

    rows = []

    for _, participant in tqdm(
        sub_info.iterrows(),
        total=len(sub_info),
        desc=video_name
    ):

        patient_id = participant["Patient_ID"]

        sampling_rate = int(
            participant["sampling_rate"]
        )

        raw = load_raw_gaze(
            patient_id,
            video_name
        )

        if raw is None:
            continue

        # Require at least approximately one second
        if len(raw) < sampling_rate:
            continue

        clean = clean_and_convert(
            raw,
            sampling_rate
        )

        # Remove recordings with excessive tracker loss
        if clean.attrs["tracker_loss_prop"] > 0.5:
            continue

        events, _ = detect_events(clean)

        features = extract_features(
            clean,
            events
        )

        features["Patient_ID"] = patient_id
        features["video"] = video_name
        features["label"] = int(participant["label"])

        rows.append(features)

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    output_directory = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "features"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    for video_column, video_name in VIDEO_NAME_MAP.items():

        print("\n" + "=" * 60)
        print(f"Processing: {video_name}")
        print("=" * 60)

        df = build_dataset(video_column)

        output_file = (
            output_directory
            / f"features_{video_name}.csv"
        )

        df.to_csv(
            output_file,
            index=False
        )

        print(
            f"\nSaved {len(df)} recordings to:"
            f"\n{output_file}"
        )

        print("\nLabels:")

        print(
            df["label"]
            .value_counts()
            .sort_index()
        )