"""
Wav2Vec 2.0 wrapper used for English oral-reading speech feature extraction.

This module only produces embeddings from audio. It makes no predictions
and has no notion of "dyslexia" — it is a general-purpose speech
representation extractor.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Tuple

import numpy as np
import torch
from transformers import Wav2Vec2Model, Wav2Vec2Processor

logger = logging.getLogger(__name__)


class Wav2VecEmbedder:
    """Loads a pretrained Wav2Vec 2.0 model/processor and extracts
    fixed-length embeddings from raw waveforms via mean pooling."""

    def __init__(self, model_name: str = "facebook/wav2vec2-base", device: str = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading Wav2Vec2 processor/model: %s", model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Determine embedding dimension dynamically from the loaded model's
        # config rather than hard-coding it (different Wav2Vec2 checkpoints
        # have different hidden sizes).
        self.embedding_dim = self.model.config.hidden_size

    @torch.no_grad()
    def extract_embedding(self, waveform: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extract a single fixed-length embedding vector for one waveform
        using mean pooling over the temporal dimension of the last hidden
        state.
        """
        inputs = self.processor(
            waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(self.device)

        outputs = self.model(input_values)
        hidden_states = outputs.last_hidden_state  # (batch, time, hidden)

        embedding = hidden_states.mean(dim=1)  # mean pooling -> (batch, hidden)
        return embedding.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def extract_embeddings_batch(
        self, waveforms: list, sample_rate: int = 16000
    ) -> np.ndarray:
        """Extract embeddings for a list of waveforms (looped for simplicity
        and to tolerate variable-length audio; batching with padding masks
        can be added later if throughput becomes a bottleneck)."""
        embeddings = [self.extract_embedding(wf, sample_rate) for wf in waveforms]
        return np.vstack(embeddings)


@lru_cache(maxsize=1)
def get_cached_embedder(model_name: str = "facebook/wav2vec2-base") -> Wav2VecEmbedder:
    """Return a process-wide cached embedder so Streamlit doesn't reload the
    model on every interaction."""
    return Wav2VecEmbedder(model_name=model_name)
