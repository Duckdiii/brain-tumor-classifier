"""Training entry point for YOLOv8 / YOLOv11 classifiers.

Both share the Ultralytics ``model.train(**kwargs)`` API, so a single
function serves both by taking the base checkpoint name (``yolov8m-cls.pt``
or ``yolo11m-cls.pt``) from ``configs/yolov8.yaml`` / ``configs/yolov11.yaml``.
``data_dir`` is the classification dataset root (containing ``train/`` and
``valid/`` or ``val/`` subfolders per class) shared with CNN/ViT.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ultralytics import YOLO


def train(
    base_model: str,
    data_dir: Path | str,
    project: Path | str,
    weights_out: Path | str,
    epochs: int = 200,
    imgsz: int = 224,
    batch: int = 32,
    device: int | str = "cpu",
    **extra_train_kwargs,
) -> Path:
    """Train a YOLO classifier and copy the best checkpoint to ``weights_out``.

    ``extra_train_kwargs`` forwards any remaining keys from
    ``configs/yolov8.yaml`` / ``configs/yolov11.yaml`` (optimizer, lr0,
    warmup_epochs, ...) straight to ``ultralytics.YOLO.train``.
    """
    model = YOLO(base_model)
    model.train(
        data=str(data_dir),
        task="classify",
        project=str(project),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        **extra_train_kwargs,
    )

    # ``model.trainer.best`` is the actual save path Ultralytics used (it
    # auto-increments the run folder name, e.g. result_train2, if a folder
    # from a previous run already exists).
    best_weights = Path(model.trainer.best)
    weights_out = Path(weights_out)
    weights_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best_weights, weights_out)
    return weights_out
