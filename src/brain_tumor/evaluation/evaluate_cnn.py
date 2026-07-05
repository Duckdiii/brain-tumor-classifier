"""Evaluation helpers for the CNN classifier (from ``cnn_evaluate``/``cnn_test``).

These are UI-agnostic: pass a ``progress_callback(percent, message)`` if you
want progress reporting (the Streamlit app wires one in).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from brain_tumor.data.transforms import build_transform, compute_mean_std

ProgressCallback = Optional[Callable[[int, str], None]]


def _report(callback: ProgressCallback, percent: int, message: str) -> None:
    if callback is not None:
        callback(percent, message)


def evaluate(
    model: nn.Module,
    test_dir: Path | str,
    train_dir_for_stats: Path | str,
    classes: list[str],
    image_size: int = 640,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    progress_callback: ProgressCallback = None,
) -> tuple[list[dict], float]:
    """Batched evaluation. Returns (per-image predictions, accuracy percent)."""
    mean, std = compute_mean_std(train_dir_for_stats, image_size)
    transform = build_transform(image_size, mean, std)
    dataset = ImageFolder(root=str(test_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    results: list[dict] = []
    correct = 0
    total = 0
    total_batches = len(loader)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(len(images)):
                global_idx = batch_idx * loader.batch_size + i
                image_path = dataset.samples[global_idx][0]
                results.append(
                    {"filename": image_path, "predicted_class": classes[predicted[i].item()]}
                )
            percent = int((batch_idx + 1) / total_batches * 100)
            _report(progress_callback, percent, f"Dang danh gia... {percent}%")

    precision = 100 * correct / total
    return results, precision


def test(
    model: nn.Module,
    test_dir: Path | str,
    train_dir_for_stats: Path | str,
    classes: list[str],
    image_size: int = 640,
    device: str | torch.device = "cpu",
    progress_callback: ProgressCallback = None,
) -> tuple[list[dict], float]:
    """Per-image test loop reporting true/predicted label and correctness."""
    mean, std = compute_mean_std(train_dir_for_stats, image_size)
    transform = build_transform(image_size, mean, std)
    dataset = ImageFolder(root=str(test_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.to(device)
    model.eval()
    results: list[dict] = []
    correct = 0
    total = 0
    total_batches = len(loader)

    with torch.no_grad():
        for idx, (image, label) in enumerate(loader):
            image, label = image.to(device), label.to(device)
            output = model(image)
            _, pred = torch.max(output, 1)
            is_correct = bool((pred == label).item())
            correct += is_correct
            total += 1
            image_path = dataset.samples[idx][0]
            results.append(
                {
                    "filename": image_path,
                    "true_label": classes[label.item()],
                    "predicted_label": classes[pred.item()],
                    "is_correct": is_correct,
                }
            )
            percent = int((idx + 1) / total_batches * 100)
            _report(progress_callback, percent, f"Dang test... {percent}%")

    accuracy = 100 * correct / total
    return results, accuracy
