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
    cnn_train: Path
    cnn_val: Path
    cnn_test: Path
    vit_train: Path
    vit_val: Path
    vit_test: Path
    yolo_train: Path
    yolo_val: Path
    yolo_test: Path
    cnn_weights: Path
    vit_weights: Path
    yolov8_weights: Path
    yolov10_weights: Path
    yolov8_runs: Path
    yolov10_runs: Path

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
            cnn_train=data(raw["processed"]["cnn"]["train"]),
            cnn_val=data(raw["processed"]["cnn"]["val"]),
            cnn_test=data(raw["processed"]["cnn"]["test"]),
            vit_train=data(raw["processed"]["vit"]["train"]),
            vit_val=data(raw["processed"]["vit"]["val"]),
            vit_test=data(raw["processed"]["vit"]["test"]),
            yolo_train=data(raw["processed"]["yolo"]["train"]),
            yolo_val=data(raw["processed"]["yolo"]["val"]),
            yolo_test=data(raw["processed"]["yolo"]["test"]),
            cnn_weights=weights(raw["weights"]["cnn"]),
            vit_weights=weights(raw["weights"]["vit"]),
            yolov8_weights=weights(raw["weights"]["yolov8"]),
            yolov10_weights=weights(raw["weights"]["yolov10"]),
            yolov8_runs=weights(raw["runs"]["yolov8"]),
            yolov10_runs=weights(raw["runs"]["yolov10"]),
        )


def yolo_dataset_yaml() -> Path:
    """Path to the Ultralytics dataset descriptor shared by YOLOv8/v10."""
    return CONFIGS_DIR / "yolo_dataset.yaml"
