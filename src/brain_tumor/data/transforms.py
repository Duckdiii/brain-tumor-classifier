"""Image transforms and dataset normalization statistics.

Unifies ``compute_mean_std_Cnn`` and ``calculate_mean_std_Vit`` from the
original ``Library.py`` (identical logic, only the default image size
differed) into a single ``compute_mean_std`` helper.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def compute_mean_std(train_dir: Path | str, image_size: int, batch_size: int = 32) -> tuple[list[float], list[float]]:
    """Compute per-channel mean/std over a training split for normalization."""
    base_transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )
    dataset = ImageFolder(root=str(train_dir), transform=base_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples
    mean /= total_images
    std /= total_images
    return mean.tolist(), std.tolist()


def build_transform(image_size: int, mean: list[float], std: list[float]) -> transforms.Compose:
    """Standard resize + tensor + normalize pipeline used for eval/inference."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
