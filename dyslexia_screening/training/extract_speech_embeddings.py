
"""
Extract Wav2Vec2 embeddings for every audio file referenced in
artifacts/speech/reading_features.csv.

Output:
    artifacts/speech/speech_features.csv
    artifacts/speech/embeddings/*.npy
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config, resolve_path
from utils.audio_utils import preprocess_audio, AudioValidationError
from models.speech.wav2vec_model import Wav2VecEmbedder


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def main():

    # ---------------------------------------------------------
    # 1. Load configuration
    # ---------------------------------------------------------

    config = load_config()

    artifacts_dir = resolve_path(
        config["artifacts"]["speech"]
    )

    embeddings_dir = os.path.join(
        artifacts_dir,
        "embeddings"
    )

    os.makedirs(
        embeddings_dir,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # 2. Load reading feature table
    # ---------------------------------------------------------

    reading_features_path = os.path.join(
        artifacts_dir,
        "reading_features.csv"
    )

    if not os.path.exists(reading_features_path):

        logger.error(
            "Reading feature table not found at %s. "
            "Run training/extract_reading_features.py first.",
            reading_features_path,
        )

        sys.exit(1)

    df = pd.read_csv(
        reading_features_path
    )

    logger.info(
        "Loaded reading feature table: %d rows",
        len(df)
    )

    if "audio_path" not in df.columns:

        logger.error(
            "Column 'audio_path' is missing from reading_features.csv."
        )

        sys.exit(1)

    # ---------------------------------------------------------
    # 3. Load speech model
    # ---------------------------------------------------------

    sample_rate = config["speech"]["sample_rate"]
    model_name = config["speech"]["model_name"]

    logger.info(
        "Loading Wav2Vec2 model '%s' for embedding extraction...",
        model_name,
    )

    embedder = Wav2VecEmbedder(
        model_name=model_name
    )

    logger.info(
        "Embedding dimension: %d",
        embedder.embedding_dim
    )

    # ---------------------------------------------------------
    # 4. Extract embeddings
    # ---------------------------------------------------------

    embedding_rows = []
    valid_indices = []

    total = len(df)

    logger.info(
        "Starting embedding extraction for %d audio files...",
        total
    )

    for position, (idx, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        audio_path = row["audio_path"]

        # Print progress every file
        logger.info(
            "Processing %d/%d: %s",
            position,
            total,
            os.path.basename(audio_path),
        )

        try:

            result = preprocess_audio(
                audio_path,
                target_sample_rate=sample_rate
            )

            embedding = embedder.extract_embedding(
                result.waveform,
                result.sample_rate
            )

            embedding = np.asarray(
                embedding,
                dtype=np.float32
            ).reshape(-1)

        except AudioValidationError as exc:

            logger.warning(
                "Skipping audio validation failure: %s -> %s",
                audio_path,
                exc,
            )

            continue

        except Exception as exc:

            logger.warning(
                "Unexpected error embedding %s: %s",
                audio_path,
                exc,
            )

            continue

        # -----------------------------------------------------
        # Save individual embedding
        # -----------------------------------------------------

        cache_name = (
            f"{idx}_"
            f"{os.path.splitext(os.path.basename(audio_path))[0]}"
            ".npy"
        )

        cache_path = os.path.join(
            embeddings_dir,
            cache_name
        )

        np.save(
            cache_path,
            embedding
        )

        embedding_rows.append(
            embedding
        )

        valid_indices.append(
            idx
        )

        # -----------------------------------------------------
        # Progress summary every 25 files
        # -----------------------------------------------------

        if position % 25 == 0:

            logger.info(
                "Progress: %d/%d processed | %d successful",
                position,
                total,
                len(embedding_rows),
            )

    # ---------------------------------------------------------
    # 5. Check extraction result
    # ---------------------------------------------------------

    if not embedding_rows:

        logger.error(
            "No embeddings were successfully extracted."
        )

        sys.exit(1)

    logger.info(
        "Embedding extraction complete: %d/%d successful",
        len(embedding_rows),
        total,
    )

    # ---------------------------------------------------------
    # 6. Create embedding matrix
    # ---------------------------------------------------------

    embedding_matrix = np.vstack(
        embedding_rows
    )

    logger.info(
        "Embedding matrix shape: %s",
        embedding_matrix.shape
    )

    embedding_cols = [
        f"wav2vec_feature_{i + 1}"
        for i in range(
            embedding_matrix.shape[1]
        )
    ]

    embedding_df = pd.DataFrame(
        embedding_matrix,
        columns=embedding_cols,
        index=valid_indices,
    )

    # ---------------------------------------------------------
    # 7. Combine reading features + Wav2Vec embeddings
    # ---------------------------------------------------------

    combined = df.loc[
        valid_indices
    ].reset_index(
        drop=True
    )

    embedding_df = embedding_df.reset_index(
        drop=True
    )

    combined = pd.concat(
        [
            combined,
            embedding_df
        ],
        axis=1
    )

    # ---------------------------------------------------------
    # 8. Save final speech feature table
    # ---------------------------------------------------------

    out_path = os.path.join(
        artifacts_dir,
        "speech_features.csv"
    )

    combined.to_csv(
        out_path,
        index=False
    )

    logger.info(
        "Saved combined speech feature table to %s",
        out_path
    )

    logger.info(
        "Final table: %d rows, %d Wav2Vec2 dimensions",
        len(combined),
        embedding_matrix.shape[1],
    )


if __name__ == "__main__":
    main()
