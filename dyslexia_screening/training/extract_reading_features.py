"""
Build the MPS reading-performance feature table (without Wav2Vec embeddings
— see extract_speech_embeddings.py for that half).

Usage:
    python training/extract_reading_features.py

Output:
    artifacts/speech/reading_features.csv

IMPORTANT: The prediction target column ('reading_outcome' by default, see
config.yaml -> speech.target_column) must be derived from an actual
documented field in the MPS annotations (e.g. a pass/fail or
correct/incorrect outcome field that the dataset itself provides). This
script raises an error rather than inventing one if that field is missing.
Do NOT threshold WCPM/accuracy here to manufacture a dyslexia label.

FIELD NAMES: this script reads alignment-count fields (total reference
words, correct words, etc.) using the name mapping in config.yaml's
speech.annotation_fields, not hard-coded field names, since the real
data.json schema may differ. If it reports 0 usable rows, check the
"Example record keys" line logged by training/mps_dataset.py (it prints
the actual keys found in your data.json) and update
speech.annotation_fields to match.
"""
from __future__ import annotations

import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config, resolve_path
from utils.audio_utils import preprocess_audio, AudioValidationError
from models.speech.speech_features import (
    ReadingAlignmentCounts,
    compute_reading_performance_features,
)
from training.mps_dataset import discover_mps_records, MPSRecord

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ANNOTATION_FIELDS = {
    "total_reference_words": "total_reference_words",
    "correct_words": "correct_words",
    "substitutions": "substitutions",
    "deletions": "deletions",
    "insertions": "insertions",
    "reading_duration_seconds": "reading_duration_seconds",
    "hesitation_count": "hesitation_count",
}


def _extract_alignment_counts(
    record: MPSRecord,
    field_map: dict,
) -> ReadingAlignmentCounts | None:

    ann = record.reading_annotations

    total_key = field_map["total_reference_words"]
    correct_key = field_map["correct_words"]

    if total_key not in ann or correct_key not in ann:
        return None

    return ReadingAlignmentCounts(
        total_reference_words=int(
            ann[total_key]
        ),
        correct_words=int(
            ann[correct_key]
        ),
        substitutions=int(
            ann.get(
                field_map["substitutions"],
                0,
            ) or 0
        ),
        deletions=int(
            ann.get(
                field_map["deletions"],
                0,
            ) or 0
        ),
        insertions=int(
            ann.get(
                field_map["insertions"],
                0,
            ) or 0
        ),
    )


def build_feature_table(
    dataset_root: str, sample_rate: int, target_column: str, field_map: dict
) -> pd.DataFrame:
    records = discover_mps_records(dataset_root)
    rows = []
    skipped_no_annotation = 0
    skipped_bad_audio = 0
    
    #skipped_no_target = 0
    
    example_keys_logged = False

    for record in records:
        alignment = _extract_alignment_counts(record, field_map)
        if alignment is None:
            skipped_no_annotation += 1
            if not example_keys_logged and record.reading_annotations:
                logger.info(
                    "Example record keys (first record with an annotation, for "
                    "fixing speech.annotation_fields in config.yaml): %s",
                    list(record.reading_annotations.keys()),
                )
                example_keys_logged = True
            continue

        try:
            result = preprocess_audio(record.audio_path, target_sample_rate=sample_rate)
        except AudioValidationError as exc:
            logger.warning("Skipping %s: %s", record.audio_path, exc)
            skipped_bad_audio += 1
            continue

        duration_key = field_map["reading_duration_seconds"]
        reading_duration = record.reading_annotations.get(duration_key, result.duration_seconds)

        features = compute_reading_performance_features(
            waveform=result.waveform,
            sample_rate=result.sample_rate,
            alignment_counts=alignment,
            reading_duration_seconds=float(reading_duration),
            hesitation_count=record.reading_annotations.get(field_map["hesitation_count"]),
        )

        row = {
    "student_id": record.student_id,
    "audio_path": record.audio_path,
    "grade": record.grade,
    "prompt": record.prompt,
    **features.to_dict(),
}
        rows.append(row)

    logger.info(
    "Feature extraction summary: %d usable rows, %d skipped (no annotation), "
    "%d skipped (bad audio)",
    len(rows),
    skipped_no_annotation,
    skipped_bad_audio,
)

    return pd.DataFrame(rows)


def main():
    config = load_config()
    dataset_root = resolve_path(config["speech"]["dataset_path"])
    sample_rate = config["speech"]["sample_rate"]
    target_column = config["speech"]["target_column"]
    artifacts_dir = resolve_path(config["artifacts"]["speech"])
    field_map = {**DEFAULT_ANNOTATION_FIELDS, **config["speech"].get("annotation_fields", {})}

    os.makedirs(artifacts_dir, exist_ok=True)

    df = build_feature_table(dataset_root, sample_rate, target_column, field_map)

    if df.empty:
        logger.error(
            "No usable MPS records found. Check that %s contains audio files "
            "matched to annotations (via data.json or per-file annotations) "
            "that include '%s' and the fields mapped in "
            "speech.annotation_fields ('%s', '%s') in config.yaml. See the "
            "'Example record keys' log line above (if any) for your actual "
            "data.json field names.",
            dataset_root,
            target_column,
            field_map["total_reference_words"],
            field_map["correct_words"],
        )
        sys.exit(1)

    out_path = os.path.join(artifacts_dir, "reading_features.csv")
    df.to_csv(out_path, index=False)
    logger.info("Saved reading-performance feature table to %s (%d rows)", out_path, len(df))


if __name__ == "__main__":
    main()
