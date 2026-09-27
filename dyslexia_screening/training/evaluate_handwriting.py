"""
Standalone confusion-matrix visualization for the trained handwriting
classifier, reading the metrics.json produced by train_handwriting.py.

Usage:
    python training/evaluate_handwriting.py
"""
from __future__ import annotations

import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def plot_confusion_matrix(cm: np.ndarray, class_names: list, out_path: str, title: str) -> None:
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(5, n * 1.1), max(4.5, n)))
    im = ax.imshow(cm, cmap="Purples")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    config = load_config()
    artifacts_dir = resolve_path(config["artifacts"]["handwriting"])
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    class_mapping_path = os.path.join(artifacts_dir, "class_mapping.json")

    if not os.path.exists(metrics_path):
        logger.error("metrics.json not found. Run training/train_handwriting.py first.")
        sys.exit(1)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    with open(class_mapping_path, "r", encoding="utf-8") as f:
        class_mapping = json.load(f)

    if all(str(i) in class_mapping for i in range(len(class_mapping))):
        # Format: {"0": "non_dyslexic", "1": "dyslexic"}
        class_names = [
            class_mapping[str(i)]
            for i in range(len(class_mapping))
        ]
    else:
        # Format: {"non_dyslexic": 0, "dyslexic": 1}
        class_names = [
            class_name
            for class_name, class_index in sorted(
                class_mapping.items(),
                key=lambda item: int(item[1])
            )
        ]

    # Make the displayed names user-friendly
    class_names = [
        {
            "dyslexic": "Dyslexia",
            "non_dyslexic": "Non-Dyslexia",
        }.get(str(name).lower(), str(name).replace("_", " ").title())
        for name in class_names
    ]    

    cm = np.array(metrics["confusion_matrix"])

    out_path = os.path.join(artifacts_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, out_path, title="Handwriting Model")

    logger.info("Saved confusion matrix plot to %s", out_path)
    logger.info("Metrics summary: %s", {k: v for k, v in metrics.items() if k != "confusion_matrix"})


if __name__ == "__main__":
    main()
