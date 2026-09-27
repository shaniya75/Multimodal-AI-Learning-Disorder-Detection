"""
Standalone re-evaluation / confusion-matrix visualization for the trained
speech classifier. Re-uses the saved model and the held-out split logic
from train_speech.py so metrics.json and the plotted confusion matrix stay
consistent with what was reported during training.

Usage:
    python training/evaluate_speech.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_loader import load_config, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def print_metrics_table(metrics: dict) -> None:
    rows = []
    for name, value in metrics.items():
        if name == "confusion_matrix":
            continue
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        rows.append((str(name), "N/A" if value is None else str(value)))

    metric_width = max(len("Metric"), *(len(name) for name, _ in rows))
    value_width = 72
    border = f"+{'-' * (metric_width + 2)}+{'-' * (value_width + 2)}+"

    print("\nMetrics Summary:")
    print(border)
    print(f"| {'Metric':<{metric_width}} | {'Value':<{value_width}} |")
    print(border)
    for name, value in rows:
        wrapped_value = textwrap.wrap(value, width=value_width) or [""]
        for index, line in enumerate(wrapped_value):
            metric = name if index == 0 else ""
            print(f"| {metric:<{metric_width}} | {line:<{value_width}} |")
    print(border)


def plot_confusion_matrix(cm: np.ndarray, class_names: list, out_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
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
    artifacts_dir = resolve_path(config["artifacts"]["speech"])
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    class_mapping_path = os.path.join(artifacts_dir, "class_mapping.json")

    if not os.path.exists(metrics_path):
        logger.error("metrics.json not found. Run training/train_speech.py first.")
        sys.exit(1)

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    with open(class_mapping_path, "r", encoding="utf-8") as f:
        class_mapping = json.load(f)

    class_names = [class_mapping[str(i)] for i in range(len(class_mapping))]
    cm = np.array(metrics["confusion_matrix"])

    out_path = os.path.join(artifacts_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names, out_path, title="Speech (Reading-Performance) Model")

    logger.info("Saved confusion matrix plot to %s", out_path)
    print_metrics_table(metrics)


if __name__ == "__main__":
    main()
