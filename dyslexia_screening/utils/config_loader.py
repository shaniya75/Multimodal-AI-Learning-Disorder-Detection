"""
Central configuration loader.

All modules should read paths and hyperparameters through this loader rather
than hard-coding values, so that the dataset locations, model names, and
training parameters can be changed in one place (config.yaml).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict

import yaml

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
)


@lru_cache(maxsize=1)
def load_config(config_path: str = _DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load and cache the YAML configuration file.

    Parameters
    ----------
    config_path: str
        Path to config.yaml. Defaults to the project-root config.yaml.

    Returns
    -------
    dict
        Parsed configuration.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_project_root() -> str:
    """Return the absolute path to the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(relative_path: str) -> str:
    """Resolve a path from config.yaml (which is stored relative to project
    root) into an absolute path usable from any working directory."""
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(get_project_root(), relative_path)
