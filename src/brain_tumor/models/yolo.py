"""Thin wrappers around ``ultralytics.YOLO`` for the classifier models.

YOLOv8 and YOLOv11 share the exact same Ultralytics API surface; only the
base checkpoint (``yolov8m-cls.pt`` vs ``yolo11m-cls.pt``) differs, so both
are served by the same loader.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def load_yolo(weights_path: Path | str) -> YOLO:
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    return YOLO(str(weights_path))
