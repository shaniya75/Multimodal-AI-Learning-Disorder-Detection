"""
Loads a trained ResNet18 handwriting classifier and produces structured
predictions. Class names are loaded from class_mapping.json (derived from
the actual dataset) — never hard-coded or renamed into dyslexia categories.
"""
from __future__ import annotations

import json
import os
from typing import Dict

import torch
import torch.nn.functional as F
from PIL import Image

from models.handwriting.resnet_model import load_resnet18_checkpoint
from models.handwriting.preprocessing import image_to_tensor, load_preprocessing_config
from utils.validation import HANDWRITING_NOT_TRAINED_MESSAGE, require_files


class HandwritingPredictor:
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = artifacts_dir
        self.checkpoint_path = os.path.join(artifacts_dir, "resnet18.pth")
        self.class_mapping_path = os.path.join(artifacts_dir, "class_mapping.json")
        self.preprocessing_path = os.path.join(artifacts_dir, "preprocessing.json")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._class_mapping: Dict[str, str] = {}
        self._image_size = 224

    def is_trained(self) -> bool:
        return all(
            os.path.exists(p)
            for p in [self.checkpoint_path, self.class_mapping_path, self.preprocessing_path]
        )

    def load(self) -> None:
        require_files(
            [self.checkpoint_path, self.class_mapping_path, self.preprocessing_path],
            HANDWRITING_NOT_TRAINED_MESSAGE,
        )

        with open(self.class_mapping_path, "r", encoding="utf-8") as f:
            self._class_mapping = json.load(f)

        preprocessing_cfg = load_preprocessing_config(self.preprocessing_path)
        self._image_size = preprocessing_cfg.get("image_size", 224)

        num_classes = len(self._class_mapping)
        self._model = load_resnet18_checkpoint(
            self.checkpoint_path, num_classes=num_classes, device=self.device
        )

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict:
        if self._model is None:
            self.load()

        tensor = image_to_tensor(image, image_size=self._image_size).to(self.device)
        logits = self._model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(probs.argmax())
        pred_label = self._class_mapping.get(str(pred_idx), str(pred_idx))

        probabilities = {
            self._class_mapping.get(str(i), f"class_{i}"): float(round(p, 4))
            for i, p in enumerate(probs)
        }

        return {
            "prediction": pred_label,
            "probabilities": probabilities,
            "model": "ResNet18",
        }
