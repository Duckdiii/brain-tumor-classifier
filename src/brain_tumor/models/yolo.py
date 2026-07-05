"""Thin wrappers around ``ultralytics.YOLO`` for the detection models.

YOLOv8 and YOLOv10 share the exact same Ultralytics API surface; only the
base checkpoint (``yolov8m.pt`` vs ``yolov10m.pt``) differs, so both are
served by the same loader.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


def load_yolo(weights_path: Path | str) -> YOLO:
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
    return YOLO(str(weights_path))
