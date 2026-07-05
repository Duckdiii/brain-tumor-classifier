"""Training loop for the custom ``GeneralCNN`` classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from brain_tumor.data.transforms import build_transform, compute_mean_std
from brain_tumor.models.cnn import GeneralCNN

ProgressCallback = Optional[Callable[[int, str], None]]


def train(
    train_dir: Path | str,
    val_dir: Path | str,
    checkpoint_out: Path | str,
    num_classes: int,
    image_size: int = 640,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str | torch.device = "cpu",
    progress_callback: ProgressCallback = None,
) -> dict[str, list[float]]:
    """Train ``GeneralCNN`` and save the best-val-accuracy checkpoint.

    Returns the per-epoch train/val loss and accuracy history.
    """
    mean, std = compute_mean_std(train_dir, image_size, batch_size)
    transform = build_transform(image_size, mean, std)

    train_dataset = ImageFolder(root=str(train_dir), transform=transform)
    val_dataset = ImageFolder(root=str(val_dir), transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = GeneralCNN(num_classes=num_classes, image_size=image_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_accuracy = -1.0
    checkpoint_out = Path(checkpoint_out)
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
        val_loss /= len(val_dataset)
        val_accuracy = correct / len(val_dataset)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_out)

        if progress_callback is not None:
            percent = int((epoch + 1) / epochs * 100)
            progress_callback(
                percent,
                f"Epoch {epoch + 1}/{epochs} - train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}",
            )

    return history
