"""Evaluation / inference helpers for the YOLOv8 and YOLOv11 classifiers.

Consolidates ``evaluate_yolov8``/``evaluate_yolov10`` from the original
``Library.py`` into version-agnostic functions, since both models share the
same Ultralytics API. ``evaluate_yolo``/``test_yolo`` mirror the shape of
``evaluate_cnn``/``evaluate_vit`` so all four models can be compared with the
same "precision" metric.
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
    data_dir: Path | str,
    project: Path | str,
    imgsz: int = 224,
    batch: int = 32,
    device: int | str = "cpu",
) -> dict:
    """Run validation and return top1/top5 accuracy (``precision`` = top1)."""
    model = load_yolo(weights_path)
    results = model.val(
        data=str(data_dir),
        imgsz=imgsz,
        batch=batch,
        device=device,
        split="val",
        verbose=True,
        name="result_val",
        project=str(project),
    )
    top1 = _scalar(results.top1)
    return {
        "model_name": os.path.basename(str(weights_path)),
        "top1": top1,
        "top5": _scalar(results.top5),
        "precision": top1,
    }


def test_yolo(
    weights_path: Path | str,
    test_dir: Path | str,
    device: int | str = "cpu",
    progress_callback: ProgressCallback = None,
) -> tuple[list[dict], float]:
    """Per-image test loop over a class-subfoldered directory.

    Returns ``(per_image_results, accuracy_percent)`` in the same shape as
    ``evaluate_cnn.test`` / ``evaluate_vit.test``.
    """
    test_dir = Path(test_dir)
    model = load_yolo(weights_path)

    image_files = [
        (class_dir.name, image_path)
        for class_dir in sorted(test_dir.iterdir())
        if class_dir.is_dir()
        for image_path in sorted(class_dir.iterdir())
        if image_path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    results: list[dict] = []
    correct = 0
    total = len(image_files)

    for idx, (true_label, image_path) in enumerate(image_files):
        prediction = model.predict(source=str(image_path), device=device, verbose=False)[0]
        predicted_label = prediction.names[int(prediction.probs.top1)]
        is_correct = predicted_label == true_label
        correct += is_correct
        results.append(
            {
                "filename": str(image_path),
                "true_label": true_label,
                "predicted_label": predicted_label,
                "is_correct": is_correct,
            }
        )
        percent = int((idx + 1) / total * 100) if total else 100
        _report(progress_callback, percent, f"Dang test... {percent}%")

    accuracy = 100 * correct / total if total else 0.0
    return results, accuracy
