"""Evaluation / inference helpers for YOLOv8 and YOLOv10 detectors.

Consolidates ``evaluate_yolov8``/``evaluate_yolov10`` and
``test_yolov8_model_predict``/``test_yolov10_model_predict`` from the
original ``Library.py`` into version-agnostic functions, since both models
share the same Ultralytics API.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from brain_tumor.constants import IMAGE_EXTENSIONS
from brain_tumor.models.yolo import load_yolo

ProgressCallback = Optional[Callable[[int, str], None]]


def _report(callback: ProgressCallback, percent: int, message: str) -> None:
    if callback is not None:
        callback(percent, message)


def _scalar(value) -> float:
    if isinstance(value, (list, np.ndarray)):
        return float(np.mean(value)) if len(value) > 0 else 0.0
    return float(value)


def evaluate_yolo(
    weights_path: Path | str,
    data_yaml: Path | str,
    project: Path | str,
    imgsz: int = 640,
    batch: int = 32,
    conf: float = 0.5,
    iou: float = 0.5,
    device: int | str = "cpu",
) -> dict:
    """Run validation and return mAP50 / mAP50-95 / precision / recall."""
    model = load_yolo(weights_path)
    results = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou=iou,
        plots=True,
        save_json=True,
        split="val",
        verbose=True,
        save_txt=True,
        save_conf=True,
        name="result_val",
        project=str(project),
    )
    return {
        "model_name": os.path.basename(str(weights_path)),
        "mAP50": _scalar(results.box.map50),
        "mAP50-95": _scalar(results.box.map),
        "precision": _scalar(results.box.p),
        "recall": _scalar(results.box.r),
    }


def predict_on_dir(
    weights_path: Path | str,
    images_dir: Path | str,
    project: Path | str,
    imgsz: int = 640,
    conf: float = 0.5,
    progress_callback: ProgressCallback = None,
) -> list[str]:
    """Run detection over every image in ``images_dir`` and save annotated outputs.

    Returns the list of annotated image paths written under
    ``<project>/result_test``.
    """
    images_dir = Path(images_dir)
    model = load_yolo(weights_path)
    output_dir = Path(project) / "result_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    total = len(image_files)

    for idx, image_path in enumerate(image_files):
        model.predict(
            source=str(image_path),
            imgsz=imgsz,
            conf=conf,
            save=True,
            project=str(project),
            name="result_test",
            exist_ok=True,
            verbose=False,
        )
        percent = int((idx + 1) / total * 100) if total else 100
        _report(progress_callback, percent, f"Dang du doan... {percent}%")

    return [str(p) for p in output_dir.rglob("*") if p.suffix.lower() in (*IMAGE_EXTENSIONS, ".bmp")]
