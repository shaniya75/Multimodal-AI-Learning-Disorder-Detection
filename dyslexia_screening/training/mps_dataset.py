"""
MPS dataset discovery and loading.

Supports the actual MPS data.json structure:

{
    "audioID": "...",
    "audioPath": "audios/....wav",
    "metaData": {
        "speakerID": "...",
        "gender": "...",
        "grade": "...",
        "storyID": "...",
        "paragraphID": "..."
    },
    "manualTranscript": "...",
    "promptText": "...",
    "textAlignment": [
        {
            "promptTextWord": "...",
            "manualTranscriptWord": "...",
            "miscueLabel": "c/s/i"
        }
    ]
}

This module does NOT invent dyslexia labels.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg"}
ANNOTATION_EXTENSIONS = {".json", ".csv", ".txt", ".tsv"}
CENTRAL_INDEX_FILENAME = "data.json"

# Actual MPS field + common alternatives.
_AUDIO_FIELD_KEYS = [
    "audioPath",
    "audio_path",
    "audio_file",
    "audio",
    "filename",
    "file",
    "wav_file",
    "wav",
]

_MPS_FILENAME_PATTERN = re.compile(
    r"^(?P<student_id>[^_]+)_(?P<prompt>[^_]+)_(?P<attempt>\d+)\.\w+$"
)


@dataclass
class MPSRecord:
    audio_path: str
    student_id: Optional[str] = None
    prompt: Optional[str] = None
    transcript: Optional[str] = None
    grade: Optional[str] = None
    annotation_path: Optional[str] = None
    reading_annotations: Dict = field(default_factory=dict)
    word_alignment: Optional[List[Dict]] = None


def _prune_hidden_dirs(dirnames: List[str]) -> None:
    dirnames[:] = [d for d in dirnames if not d.startswith(".")]


def _find_audio_files(root: str) -> List[str]:
    audio_files = []

    for dirpath, dirnames, filenames in os.walk(root):
        _prune_hidden_dirs(dirnames)

        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()

            if ext in AUDIO_EXTENSIONS:
                audio_files.append(os.path.join(dirpath, fname))

    return sorted(audio_files)


def _find_central_index_files(root: str) -> List[str]:
    matches = []

    for dirpath, dirnames, filenames in os.walk(root):
        _prune_hidden_dirs(dirnames)

        for fname in filenames:
            if fname.lower() == CENTRAL_INDEX_FILENAME:
                matches.append(os.path.join(dirpath, fname))

    return sorted(matches)


def _resolve_audio_path(
    dataset_root: str,
    index_path: str,
    audio_value: str,
) -> Optional[str]:
    """
    Resolve the audioPath from data.json to an actual file.

    MPS example:

        audioPath = "audios/5d44c_EN-OL-RC-538_2.wav"

    and data.json is located at:

        dataset_root/data.json

    Therefore the expected file is:

        dataset_root/audios/5d44c_EN-OL-RC-538_2.wav
    """

    if not audio_value:
        return None

    audio_value = str(audio_value).replace("\\", os.sep).replace("/", os.sep)

    # Absolute path.
    if os.path.isabs(audio_value):
        if os.path.isfile(audio_value):
            return os.path.normpath(audio_value)

    # First try relative to the data.json directory.
    index_dir = os.path.dirname(index_path)

    candidate = os.path.normpath(
        os.path.join(index_dir, audio_value)
    )

    if os.path.isfile(candidate):
        return candidate

    # Try relative to dataset root.
    candidate = os.path.normpath(
        os.path.join(dataset_root, audio_value)
    )

    if os.path.isfile(candidate):
        return candidate

    # Finally search by basename.
    basename = os.path.basename(audio_value)

    for dirpath, dirnames, filenames in os.walk(dataset_root):
        _prune_hidden_dirs(dirnames)

        if basename in filenames:
            return os.path.join(dirpath, basename)

    return None


def _alignment_to_counts(
    alignment: Optional[List[Dict]],
) -> Dict[str, int]:
    """
    Convert MPS textAlignment records into alignment counts.

    MPS labels:
        c = correct
        s = substitution
        i = insertion

    Deletions are represented by a reference word without a
    corresponding spoken word. In the supplied MPS schema,
    these appear as an alignment entry whose manualTranscriptWord
    is <eps>.

    The function does not infer dyslexia or any diagnostic label.
    """

    counts = {
        "total_reference_words": 0,
        "correct_words": 0,
        "substitutions": 0,
        "deletions": 0,
        "insertions": 0,
    }

    if not isinstance(alignment, list):
        return counts

    for item in alignment:
        if not isinstance(item, dict):
            continue

        reference_word = str(
            item.get("promptTextWord", "")
        ).strip()

        spoken_word = str(
            item.get("manualTranscriptWord", "")
        ).strip()

        label = str(
            item.get("miscueLabel", "")
        ).strip().lower()

        # <eps> on the prompt/reference side means there is
        # no reference word. Therefore this is an insertion.
        if reference_word == "<eps>":
            counts["insertions"] += 1
            continue

        # Every non-epsilon prompt word is a reference word.
        counts["total_reference_words"] += 1

        if label == "c":
            counts["correct_words"] += 1

        elif label == "s":
            counts["substitutions"] += 1

        elif label == "i":
            counts["insertions"] += 1

        elif label == "d":
            counts["deletions"] += 1

        # Defensive handling for an alignment where the spoken
        # side is explicitly marked as <eps>.
        elif spoken_word == "<eps>":
            counts["deletions"] += 1

    return counts


def _load_central_index(
    paths: List[str],
    dataset_root: str,
) -> Dict[str, Dict]:
    """
    Load central data.json files.

    The returned lookup uses the actual audio filename as the key.
    """

    index: Dict[str, Dict] = {}

    for path in paths:

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

        except Exception as exc:
            logger.warning(
                "Could not parse central index %s as JSON: %s",
                path,
                exc,
            )
            continue

        if isinstance(data, list):

            if data and isinstance(data[0], dict):
                logger.info(
                    "Central index %s: list of %d records. "
                    "Example record keys: %s",
                    path,
                    len(data),
                    list(data[0].keys()),
                )

            added = 0

            for record in data:

                if not isinstance(record, dict):
                    continue

                audio_value = None

                for key in _AUDIO_FIELD_KEYS:
                    if record.get(key):
                        audio_value = record[key]
                        break

                if audio_value is None:
                    continue

                resolved = _resolve_audio_path(
                    dataset_root,
                    path,
                    str(audio_value),
                )

                if resolved is not None:
                    audio_name = os.path.basename(resolved)
                else:
                    audio_name = os.path.basename(
                        str(audio_value)
                    )

                index[audio_name] = record
                index[
                    os.path.splitext(audio_name)[0]
                ] = record

                added += 1

            logger.info(
                "Central index %s: matched %d/%d records "
                "to an audio filename field.",
                path,
                added,
                len(data),
            )

        elif isinstance(data, dict):

            sample_key = next(iter(data), None)
            sample_val = (
                data.get(sample_key)
                if sample_key is not None
                else None
            )

            logger.info(
                "Central index %s: dict with %d top-level keys. "
                "Example key: %r, example value keys: %s",
                path,
                len(data),
                sample_key,
                list(sample_val.keys())
                if isinstance(sample_val, dict)
                else type(sample_val),
            )

            for key, record in data.items():

                if not isinstance(record, dict):
                    continue

                base = os.path.basename(str(key))

                index[base] = record
                index[
                    os.path.splitext(base)[0]
                ] = record

        else:

            logger.warning(
                "Unrecognized JSON structure in %s.",
                path,
            )

    return index


def _parse_filename_metadata(audio_path: str) -> Dict:
    fname = os.path.basename(audio_path)

    match = _MPS_FILENAME_PATTERN.match(fname)

    if not match:
        return {}

    return {
        "student_id": match.group("student_id"),
        "prompt": match.group("prompt"),
        "attempt": match.group("attempt"),
    }


def _find_matching_annotation(
    audio_path: str,
) -> Optional[str]:

    base_dir = os.path.dirname(audio_path)

    base_name = os.path.splitext(
        os.path.basename(audio_path)
    )[0]

    candidate_dirs = [
        base_dir,
        os.path.join(
            os.path.dirname(base_dir),
            "annotations",
        ),
        os.path.join(
            os.path.dirname(base_dir),
            "labels",
        ),
        os.path.join(
            os.path.dirname(base_dir),
            "transcripts",
        ),
    ]

    for cand_dir in candidate_dirs:

        if not os.path.isdir(cand_dir):
            continue

        for ext in ANNOTATION_EXTENSIONS:

            candidate = os.path.join(
                cand_dir,
                base_name + ext,
            )

            if os.path.exists(candidate):
                return candidate

    return None


def _parse_annotation_file(
    annotation_path: str,
) -> Dict:

    ext = os.path.splitext(
        annotation_path
    )[1].lower()

    try:

        if ext == ".json":

            with open(
                annotation_path,
                "r",
                encoding="utf-8",
            ) as f:
                return json.load(f)

        with open(
            annotation_path,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:
            return {"raw_text": f.read()}

    except Exception as exc:

        logger.warning(
            "Failed to parse annotation file %s: %s",
            annotation_path,
            exc,
        )

        return {}


def _infer_student_id(
    audio_path: str,
    annotation: Dict,
) -> Optional[str]:

    if annotation.get("student_id"):
        return str(annotation["student_id"])

    metadata = annotation.get("metaData")

    if isinstance(metadata, dict):

        if metadata.get("speakerID"):
            return str(metadata["speakerID"])

    parent = os.path.basename(
        os.path.dirname(audio_path)
    )

    if parent and parent.lower() not in {
        "audio",
        "wav",
        "data",
        "mps",
        "audios",
    }:
        return parent

    return None


def _prepare_mps_annotation(
    annotation: Dict,
) -> Dict:
    """
    Normalize the actual MPS record while preserving the
    original annotation fields.
    """

    annotation = dict(annotation)

    metadata = annotation.get("metaData")

    if not isinstance(metadata, dict):
        metadata = {}

    # Normalize metadata.
    if metadata.get("speakerID") is not None:
        annotation.setdefault(
            "student_id",
            str(metadata["speakerID"]),
        )

    if metadata.get("grade") is not None:
        annotation.setdefault(
            "grade",
            str(metadata["grade"]),
        )

    if annotation.get("promptText") is not None:
        annotation.setdefault(
            "prompt",
            annotation["promptText"],
        )

        annotation.setdefault(
            "transcript",
            annotation["manualTranscript"],
        )

    alignment = annotation.get("textAlignment")

    if isinstance(alignment, list):

        annotation["word_alignment"] = alignment

        counts = _alignment_to_counts(alignment)

        for key, value in counts.items():
            annotation.setdefault(key, value)

    return annotation


def discover_mps_records(
    dataset_root: str,
) -> List[MPSRecord]:

    if not os.path.isdir(dataset_root):

        logger.warning(
            "MPS dataset path does not exist: %s",
            dataset_root,
        )

        return []

    audio_files = _find_audio_files(dataset_root)

    logger.info(
        "Discovered %d audio files under %s",
        len(audio_files),
        dataset_root,
    )

    central_index_paths = _find_central_index_files(
        dataset_root
    )

    central_index: Dict[str, Dict] = {}

    if central_index_paths:

        logger.info(
            "Found %d central index file(s): %s",
            len(central_index_paths),
            central_index_paths,
        )

        central_index = _load_central_index(
            central_index_paths,
            dataset_root,
        )

    else:

        logger.info(
            "No central data.json index found under %s.",
            dataset_root,
        )

    records: List[MPSRecord] = []

    matched_central = 0
    matched_per_file = 0
    matched_none = 0

    for audio_path in audio_files:

        base_name = os.path.basename(audio_path)

        stem = os.path.splitext(base_name)[0]

        annotation: Dict = {}

        annotation_path: Optional[str] = None

        source = "none"

        if base_name in central_index:

            annotation = dict(
                central_index[base_name]
            )

            source = "central_index"

        elif stem in central_index:

            annotation = dict(
                central_index[stem]
            )

            source = "central_index"

        else:

            annotation_path = _find_matching_annotation(
                audio_path
            )

            if annotation_path:

                annotation = _parse_annotation_file(
                    annotation_path
                )

                source = "per_file"

        # Normalize actual MPS fields.
        annotation = _prepare_mps_annotation(
            annotation
        )

        # Filename fallback.
        filename_meta = _parse_filename_metadata(
            audio_path
        )

        for key, value in filename_meta.items():
            annotation.setdefault(key, value)

        if source == "central_index":
            matched_central += 1

        elif source == "per_file":
            matched_per_file += 1

        else:
            matched_none += 1

        record = MPSRecord(
            audio_path=audio_path,

            student_id=_infer_student_id(
                audio_path,
                annotation,
            ),

            prompt=annotation.get(
                "prompt"
            ),

            transcript=(
                annotation.get("transcript")
                or annotation.get("reference_text")
            ),

            grade=annotation.get(
                "grade"
            ),

            annotation_path=annotation_path,

            reading_annotations=annotation,

            word_alignment=annotation.get(
                "word_alignment"
            ),
        )

        records.append(record)

    logger.info(
        "MPS discovery complete: %d records total — "
        "%d matched via central index, "
        "%d matched via per-file annotation, "
        "%d unmatched.",
        len(records),
        matched_central,
        matched_per_file,
        matched_none,
    )

    return records