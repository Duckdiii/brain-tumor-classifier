"""Configuration loading.

Replaces the hardcoded ``D:\\Dataset\\...`` / ``/content/...`` paths that were
scattered across the original ``args.py`` files. Every path is declared once
in ``configs/paths.yaml`` (relative to the project root) and can be
relocated with the ``BRAIN_TUMOR_DATA_ROOT`` / ``BRAIN_TUMOR_WEIGHTS_ROOT``
environment variables, e.g. when the dataset lives on another drive or when
running on Colab.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"


def load_yaml(name_or_path: str | Path) -> dict[str, Any]:
    """Load a YAML file, resolving bare file names against ``configs/``."""
    path = Path(name_or_path)
    if not path.is_absolute() and not path.exists():
        path = CONFIGS_DIR / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _root_for(kind: str) -> Path:
    env_var = f"BRAIN_TUMOR_{kind.upper()}_ROOT"
    override = os.environ.get(env_var)
    return Path(override) if override else PROJECT_ROOT


@dataclass(frozen=True)
class Paths:
    """Resolved, absolute filesystem paths for data and model weights."""

    raw_images: Path
    raw_labels: Path
    classification_train: Path
    classification_val: Path
    classification_test: Path
    cnn_weights: Path
    vit_weights: Path
    yolov8_weights: Path
    yolo11_weights: Path
    yolov8_runs: Path
    yolo11_runs: Path

    @classmethod
    def load(cls, config_path: str | Path = "paths.yaml") -> "Paths":
        raw = load_yaml(config_path)
        data_root = _root_for("data")
        weights_root = _root_for("weights")

        def data(rel: str) -> Path:
            return data_root / rel

        def weights(rel: str) -> Path:
            return weights_root / rel

        return cls(
            raw_images=data(raw["raw"]["images"]),
            raw_labels=data(raw["raw"]["labels"]),
            classification_train=data(raw["processed"]["classification"]["train"]),
            classification_val=data(raw["processed"]["classification"]["val"]),
            classification_test=data(raw["processed"]["classification"]["test"]),
            cnn_weights=weights(raw["weights"]["cnn"]),
            vit_weights=weights(raw["weights"]["vit"]),
            yolov8_weights=weights(raw["weights"]["yolov8"]),
            yolo11_weights=weights(raw["weights"]["yolo11"]),
            yolov8_runs=weights(raw["runs"]["yolov8"]),
            yolo11_runs=weights(raw["runs"]["yolo11"]),
        )
