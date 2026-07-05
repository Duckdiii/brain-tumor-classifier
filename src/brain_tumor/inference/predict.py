"""Single-image inference for the four classifier models (CNN/ViT/YOLOv8/YOLOv11).

The original ``GUI_version2.py`` had an "upload image" panel that only ever
displayed a hardcoded example result (``Nhan du doan: [Vi du: Cho]``); the
functions below implement the real prediction it was meant to show.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from brain_tumor.constants import CLASS_NAMES
from brain_tumor.data.transforms import build_transform
from brain_tumor.models.yolo import load_yolo


def predict_image_classifier(
    model: torch.nn.Module,
    image: Image.Image,
    mean: list[float],
    std: list[float],
    image_size: int,
    device: str | torch.device = "cpu",
    class_names: list[str] = CLASS_NAMES,
) -> tuple[str, float]:
    """Run a CNN/ViT classifier on a single PIL image.

    Returns ``(predicted_class_name, confidence)``.
    """
    transform = build_transform(image_size, mean, std)
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        confidence, predicted_idx = torch.max(probs, dim=0)

    return class_names[predicted_idx.item()], confidence.item()


def predict_image_yolo(
    weights_path: Path | str, image: Path | str | Image.Image
) -> tuple[str, float]:
    """Run a YOLOv8/YOLOv11 classifier on a single image.

    Returns ``(predicted_class_name, confidence)``, matching
    ``predict_image_classifier``.
    """
    model = load_yolo(weights_path)
    result = model.predict(source=image, verbose=False)[0]
    predicted_idx = int(result.probs.top1)
    return result.names[predicted_idx], float(result.probs.top1conf)
