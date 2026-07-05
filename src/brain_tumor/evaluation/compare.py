"""Cross-model comparison (from ``compare_model`` in the original ``Library.py``).

Returns a plain :class:`pandas.DataFrame` of precision per model with no UI
or plotting side effects; see :mod:`brain_tumor.evaluation.reporting` for
turning that into charts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch

from brain_tumor.config import Paths, yolo_dataset_yaml
from brain_tumor.constants import CLASS_NAMES
from brain_tumor.evaluation.evaluate_cnn import evaluate as evaluate_cnn
from brain_tumor.evaluation.evaluate_vit import evaluate as evaluate_vit
from brain_tumor.evaluation.evaluate_yolo import evaluate_yolo
from brain_tumor.models.cnn import GeneralCNN
from brain_tumor.models.vit import build_vit_model

ProgressCallback = Optional[Callable[[int, str], None]]


def _report(callback: ProgressCallback, percent: int, message: str) -> None:
    if callback is not None:
        callback(percent, message)


def compare_models(
    paths: Paths,
    device: str | torch.device = "cpu",
    progress_callback: ProgressCallback = None,
) -> pd.DataFrame:
    """Evaluate CNN, ViT, YOLOv8 and YOLOv10 on their respective val splits."""
    _report(progress_callback, 0, "Dang tai mo hinh CNN...")
    cnn_model = GeneralCNN(num_classes=len(CLASS_NAMES))
    cnn_model.load_state_dict(torch.load(paths.cnn_weights, map_location=device))
    cnn_model.to(device)

    _report(progress_callback, 15, "Dang danh gia CNN...")
    _, cnn_precision = evaluate_cnn(
        cnn_model, paths.cnn_val, paths.cnn_train, CLASS_NAMES, device=device
    )

    _report(progress_callback, 30, "Dang tai ViT...")
    vit_model = build_vit_model(num_classes=len(CLASS_NAMES))
    vit_model.load_state_dict(torch.load(paths.vit_weights, map_location=device))
    vit_model.to(device)

    _report(progress_callback, 45, "Dang danh gia ViT...")
    vit_precision = evaluate_vit(vit_model, paths.vit_val, paths.vit_train, device=device)

    _report(progress_callback, 60, "Dang danh gia YOLOv8...")
    yolov8_result = evaluate_yolo(
        paths.yolov8_weights, yolo_dataset_yaml(), paths.yolov8_runs, device=device
    )

    _report(progress_callback, 75, "Dang danh gia YOLOv10...")
    yolov10_result = evaluate_yolo(
        paths.yolov10_weights, yolo_dataset_yaml(), paths.yolov10_runs, device=device
    )

    _report(progress_callback, 100, "Hoan tat so sanh.")
    return pd.DataFrame(
        [
            {"model_name": "CNN", "precision": cnn_precision / 100},
            {"model_name": "ViT", "precision": vit_precision / 100},
            {"model_name": "YOLOv8", "precision": yolov8_result["precision"]},
            {"model_name": "YOLOv10", "precision": yolov10_result["precision"]},
        ]
    )
