"""Evaluation helpers for the ViT classifier (from ``evaluate_model_ViT``/``test_model_ViT``)."""

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
    image_size: int = 224,
    batch_size: int = 32,
    device: str | torch.device = "cpu",
    progress_callback: ProgressCallback = None,
) -> float:
    """Batched accuracy evaluation. Returns accuracy percent."""
    mean, std = compute_mean_std(train_dir_for_stats, image_size)
    transform = build_transform(image_size, mean, std)
    dataset = ImageFolder(root=str(test_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model.eval()
    correct = 0
    total = 0
    total_batches = len(loader)

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            percent = int((idx + 1) / total_batches * 100)
            _report(progress_callback, percent, f"Dang danh gia... {percent}%")

    return 100 * correct / total


def test(
    model: nn.Module,
    test_dir: Path | str,
    train_dir_for_stats: Path | str,
    classes: list[str],
    image_size: int = 224,
    device: str | torch.device = "cpu",
    progress_callback: ProgressCallback = None,
) -> tuple[list[dict], float]:
    """Per-image test loop reporting true/predicted label and correctness."""
    model.to(device)
    model.eval()

    mean, std = compute_mean_std(train_dir_for_stats, image_size)
    transform = build_transform(image_size, mean, std)
    dataset = ImageFolder(root=str(test_dir), transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

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
