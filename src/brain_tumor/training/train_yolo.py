"""Training entry point for YOLOv8 / YOLOv10 detectors.

Both share the Ultralytics ``model.train(**kwargs)`` API, so a single
function serves both by taking the base checkpoint name (``yolov8m.pt`` or
``yolov10m.pt``) from ``configs/yolov8.yaml`` / ``configs/yolov10.yaml``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO


def train(
    base_model: str,
    data_yaml: Path | str,
    project: Path | str,
    weights_out: Path | str,
    epochs: int = 200,
    imgsz: int = 640,
    batch: int = 32,
    device: int | str = "cpu",
    **extra_train_kwargs,
) -> Path:
    """Train a YOLO detector and copy the best checkpoint to ``weights_out``.

    ``extra_train_kwargs`` forwards any remaining keys from
    ``configs/yolov8.yaml`` / ``configs/yolov10.yaml`` (optimizer, lr0,
    warmup_epochs, ...) straight to ``ultralytics.YOLO.train``.
    """
    model = YOLO(base_model)
    model.train(
        data=str(data_yaml),
        project=str(project),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        **extra_train_kwargs,
    )

    run_name = extra_train_kwargs.get("name", "result_train")
    best_weights = Path(project) / run_name / "weights" / "best.pt"
    weights_out = Path(weights_out)
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_weights, weights_out)
    return weights_out
