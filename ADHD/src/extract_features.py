"""
Eye-Tracking Dataset pipeline for ADHD detection (ecml-ADHD data)
==================================================================
Stage 1: Preprocessing  -> clean raw gaze coordinates, px->deg conversion
Stage 2: Fixation/saccade detection -> I-DT dispersion algorithm (reuses repo's own utils.py)
Stage 3: Feature extraction -> statistical features (saccade length, fixation count, ...)
Output: one row per subject-video with engineered features + label, ready for SVM/RF.

Run from anywhere; just point REPO_DIR at your local clone of
https://github.com/aeye-lab/ecml-ADHD
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

REPO_DIR = "/home/claude/ecml-ADHD"          # <-- change if your clone lives elsewhere
sys.path.insert(0, os.path.join(REPO_DIR, "DNNmodel", "DataGeneration"))
from utils import pix2deg, get_sacc_fix_lists_dispersion  # reuse the authors' own I-DT code

# ---- screen geometry constants, taken directly from write_Datafiles.py in the repo ----
SCREEN_PX_X, SCREEN_PX_Y = 800, 600
SCREEN_CM_X, SCREEN_CM_Y = 33.8, 27.0
DISTANCE_CM = 63.5

VIDEO_NAME_MAP = {
    "video_Diary_of_a_Wimpy_Kid_Trailer": "Diary_of_a_Wimpy_Kid_Trailer",
    "video_Fractals": "Fractals",
    "video_Despicable_Me": "Despicable_Me",
    "video_The_Present": "The_Present",
}


def load_raw_gaze(patient_id: str, video_file_name: str) -> pd.DataFrame:
    """Stage 1a: load raw sample-level gaze for one subject-video pair."""
    path = os.path.join(
        REPO_DIR, "Data", "X_px",
        f"X_px_{patient_id}_Video_{video_file_name}.npy",
    )
    if not os.path.exists(path):
        return None
    arr = np.load(path, allow_pickle=True)
    df = pd.DataFrame(arr, columns=["video_frame", "time", "x_px", "y_px"])
    df = df[df.video_frame >= 0].reset_index(drop=True)
    return df


def clean_and_convert(df: pd.DataFrame, sampling_rate: int) -> pd.DataFrame:
    """Stage 1b: cleaning raw gaze coordinates + px -> degrees of visual angle."""
    df = df.copy()
    # (0,0) is the tracker-loss / blink sentinel in this dataset, not a real gaze point
    tracker_loss_mask = (df.x_px == 0) & (df.y_px == 0)
    df.loc[tracker_loss_mask, ["x_px", "y_px"]] = np.nan

    tracker_loss_prop = tracker_loss_mask.mean()

    df["x_deg"] = pix2deg(df.x_px, SCREEN_PX_X, SCREEN_CM_X, DISTANCE_CM, adjust_origin=True)
    df["y_deg"] = pix2deg(df.y_px, SCREEN_PX_Y, SCREEN_CM_Y, DISTANCE_CM,
                           adjust_origin=True, reverse_axis=True)
    df.attrs["tracker_loss_prop"] = tracker_loss_prop
    df.attrs["sampling_rate"] = sampling_rate
    return df


def detect_events(df: pd.DataFrame):
    """Stage 2: I-DT fixation/saccade detection (reuses the repo's own algorithm)."""
    sampling_rate = df.attrs["sampling_rate"]
    events, event_df = get_sacc_fix_lists_dispersion(
        df.x_deg.values, df.y_deg.values,
        sampling_rate=sampling_rate,
    )
    return events, event_df


def extract_features(df: pd.DataFrame, events: dict) -> dict:
    """Stage 3: statistical features per subject-video."""
    sr = df.attrs["sampling_rate"]
    x, y = df.x_deg.values, df.y_deg.values

    # ---- fixation features ----
    fix_durations, fix_dispersions = [], []
    for idxs in events["fixations"]:
        if len(idxs) < 2:
            continue
        dur_ms = (len(idxs) / sr) * 1000.0
        fx, fy = x[idxs], y[idxs]
        disp = (np.nanmax(fx) - np.nanmin(fx)) + (np.nanmax(fy) - np.nanmin(fy))
        fix_durations.append(dur_ms)
        fix_dispersions.append(disp)

    # ---- saccade features ----
    sacc_amplitudes, sacc_peak_vels, sacc_durations = [], [], []
    for idxs in events["saccades"]:
        if len(idxs) < 2:
            continue
        dur_ms = (len(idxs) / sr) * 1000.0
        sx, sy = x[idxs], y[idxs]
        amplitude = np.sqrt((sx[-1] - sx[0]) ** 2 + (sy[-1] - sy[0]) ** 2)  # "saccade length"
        # sample-to-sample velocity within the saccade (deg/sec)
        dx = np.diff(sx); dy = np.diff(sy)
        step_vel = np.sqrt(dx ** 2 + dy ** 2) * sr
        peak_vel = np.nanmax(step_vel) if len(step_vel) else np.nan
        sacc_amplitudes.append(amplitude)
        sacc_peak_vels.append(peak_vel)
        sacc_durations.append(dur_ms)

    duration_sec = len(df) / sr if sr else np.nan

    feats = {
        "n_fixations": len(fix_durations),
        "fixation_rate_per_sec": len(fix_durations) / duration_sec if duration_sec else np.nan,
        "fix_duration_mean": np.nanmean(fix_durations) if fix_durations else np.nan,
        "fix_duration_std": np.nanstd(fix_durations) if fix_durations else np.nan,
        "fix_duration_median": np.nanmedian(fix_durations) if fix_durations else np.nan,
        "fix_dispersion_mean": np.nanmean(fix_dispersions) if fix_dispersions else np.nan,
        "n_saccades": len(sacc_amplitudes),
        "saccade_rate_per_sec": len(sacc_amplitudes) / duration_sec if duration_sec else np.nan,
        "sacc_amplitude_mean": np.nanmean(sacc_amplitudes) if sacc_amplitudes else np.nan,
        "sacc_amplitude_std": np.nanstd(sacc_amplitudes) if sacc_amplitudes else np.nan,
        "sacc_peak_vel_mean": np.nanmean(sacc_peak_vels) if sacc_peak_vels else np.nan,
        "sacc_duration_mean": np.nanmean(sacc_durations) if sacc_durations else np.nan,
        "tracker_loss_prop": df.attrs["tracker_loss_prop"],
        "n_samples": len(df),
    }
    return feats


def build_dataset(video_col: str, max_subjects: int = None) -> pd.DataFrame:
    """
    Runs the full Stage1->Stage2->Stage3 pipeline for every subject that has
    a recording for the given video, and returns a feature table with labels.
    video_col: one of the VIDEO_NAME_MAP keys, e.g. 'video_Despicable_Me'
    """
    sub_info = pd.read_csv(
        os.path.join(REPO_DIR, "Data", "sub_info", "sub_sel_classif.csv"), sep="\t"
    )
    sub_info = sub_info[sub_info[video_col] == True].reset_index(drop=True)
    if max_subjects:
        sub_info = sub_info.iloc[:max_subjects]

    video_file_name = VIDEO_NAME_MAP[video_col]
    rows = []
    for _, row in tqdm(sub_info.iterrows(), total=len(sub_info), desc=video_file_name):
        pid = row["Patient_ID"]
        sr = int(row["sampling_rate"])
        raw = load_raw_gaze(pid, video_file_name)
        if raw is None or len(raw) < sr:  # need at least ~1 sec of data
            continue
        clean = clean_and_convert(raw, sampling_rate=sr)
        # skip subjects with too much tracker loss to be reliable
        if clean.attrs["tracker_loss_prop"] > 0.5:
            continue
        events, _ = detect_events(clean)
        feats = extract_features(clean, events)
        feats["Patient_ID"] = pid
        feats["label"] = int(row["label"])  # 1 = ADHD, 0 = control
        rows.append(feats)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # Despicable Me has the most labeled subjects (315) -> best video to start with
    df = build_dataset("video_Despicable_Me")
    out_path = os.path.join(os.path.dirname(__file__), "features_despicable_me.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} subject feature rows to {out_path}")
    print(df["label"].value_counts())
