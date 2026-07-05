"""Train/val/test splitting for the raw YOLO-format dataset.

The raw dataset ships as ``images/*.jpg`` + YOLO ``labels/*.txt`` pairs (one
class id per file, since each image contains a single tumor region). Two
downstream layouts are derived from it:

- Classifier layout (CNN / ViT): ``<split>/<class_name>/*.jpg``
  (``torchvision.datasets.ImageFolder`` compatible).
- Detector layout (YOLOv8 / YOLOv10): ``<split>/images`` + ``<split>/labels``.

This mirrors ``split_dataset_CNN`` / ``split_dataset_ViT`` / ``split_dataset_Yolo``
from the original ``Library.py``, deduplicated into one classifier-layout
implementation and stripped of the Streamlit/UI coupling. Progress is
reported through an optional callback instead of ``st.progress``.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Callable, Optional

from brain_tumor.constants import CLASS_ID_TO_NAME, IMAGE_EXTENSIONS

ProgressCallback = Optional[Callable[[int, str], None]]


def _label_class_id(label_path: Path) -> Optional[int]:
    lines = label_path.read_text().splitlines()
    if not lines:
        return None
    return int(lines[0].split()[0])


def _report(callback: ProgressCallback, percent: int, message: str) -> None:
    if callback is not None:
        callback(percent, message)


def split_classification_dataset(
    images_dir: Path | str,
    labels_dir: Path | str,
    output_dir: Path | str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    class_names: dict[int, str] = CLASS_ID_TO_NAME,
    seed: int | None = None,
    progress_callback: ProgressCallback = None,
) -> dict[str, int]:
    """Split the raw dataset into an ``ImageFolder`` layout for CNN/ViT training.

    Ratios are expressed as percentages (0-100) and must sum to 100.
    Returns the number of images copied per split.
    """
    images_dir, labels_dir, output_dir = Path(images_dir), Path(labels_dir), Path(output_dir)
    if round(train_ratio + val_ratio + test_ratio) != 100:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 100")

    for split in ("train", "val", "test"):
        for class_name in class_names.values():
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    label_files = [f for f in labels_dir.iterdir() if f.suffix == ".txt"]
    rng = random.Random(seed)
    rng.shuffle(label_files)

    total = len(label_files)
    train_count = int(total * train_ratio / 100)
    val_count = int(total * val_ratio / 100)

    splits = {
        "train": label_files[:train_count],
        "val": label_files[train_count : train_count + val_count],
        "test": label_files[train_count + val_count :],
    }

    processed = 0
    total_to_process = sum(len(v) for v in splits.values())
    counts: dict[str, int] = {}

    for split, files in splits.items():
        copied = 0
        for label_file in files:
            image_path = images_dir / (label_file.stem + ".jpg")
            processed += 1
            percent = int(processed / total_to_process * 100)
            if not image_path.exists():
                _report(progress_callback, percent, f"Bo qua (khong co anh): {label_file.name}")
                continue
            class_id = _label_class_id(label_file)
            if class_id is None:
                _report(progress_callback, percent, f"Bo qua (nhan rong): {label_file.name}")
                continue
            class_name = class_names.get(class_id, f"class_{class_id}")
            shutil.copy(image_path, output_dir / split / class_name / image_path.name)
            copied += 1
            _report(progress_callback, percent, f"Dang chia {split}... {percent}%")
        counts[split] = copied

    return counts


def split_detection_dataset(
    images_dir: Path | str,
    labels_dir: Path | str,
    output_dir: Path | str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
    progress_callback: ProgressCallback = None,
) -> dict[str, int]:
    """Split the raw dataset into a YOLO ``images/`` + ``labels/`` layout.

    Ratios are expressed as percentages (0-100) and must sum to 100.
    Returns the number of images copied per split.
    """
    images_dir, labels_dir, output_dir = Path(images_dir), Path(labels_dir), Path(output_dir)
    if round(train_ratio + val_ratio + test_ratio) != 100:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 100")

    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    rng = random.Random(seed)
    rng.shuffle(image_files)

    total = len(image_files)
    train_count = int(total * train_ratio / 100)
    val_count = int(total * val_ratio / 100)

    splits = {
        "train": image_files[:train_count],
        "valid": image_files[train_count : train_count + val_count],
        "test": image_files[train_count + val_count :],
    }

    for split in splits:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    processed = 0
    total_to_process = total
    counts: dict[str, int] = {}

    for split, files in splits.items():
        for image_path in files:
            shutil.copy(image_path, output_dir / split / "images" / image_path.name)
            label_path = labels_dir / (image_path.stem + ".txt")
            if label_path.exists():
                shutil.copy(label_path, output_dir / split / "labels" / label_path.name)
            processed += 1
            percent = int(processed / total_to_process * 100)
            _report(progress_callback, percent, f"Dang chia dataset... {percent}%")
        counts[split] = len(files)

    return counts


def check_labels_and_images(images_dir: Path | str, labels_dir: Path | str) -> dict[str, list[str]]:
    """Report images without a matching label file and vice versa."""
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    image_stems = {f.stem for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS}
    label_stems = {f.stem for f in labels_dir.iterdir() if f.suffix == ".txt"}

    return {
        "images_without_labels": sorted(image_stems - label_stems),
        "labels_without_images": sorted(label_stems - image_stems),
        "num_images": len(image_stems),
        "num_labels": len(label_stems),
    }
