from pathlib import Path

from brain_tumor.data.splitting import (
    check_labels_and_images,
    split_classification_dataset,
    split_detection_dataset,
)

CLASS_NAMES = {0: "Glioma", 1: "Meningioma", 2: "No Tumor", 3: "Pituitary"}


def _make_raw_dataset(root: Path, num_per_class: int = 4) -> tuple[Path, Path]:
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    counter = 0
    for class_id in CLASS_NAMES:
        for _ in range(num_per_class):
            name = f"img_{counter}"
            (images_dir / f"{name}.jpg").write_bytes(b"fake-image")
            (labels_dir / f"{name}.txt").write_text(f"{class_id} 0.5 0.5 0.2 0.2\n")
            counter += 1
    return images_dir, labels_dir


def test_split_classification_dataset_distributes_all_images(tmp_path):
    images_dir, labels_dir = _make_raw_dataset(tmp_path)
    output_dir = tmp_path / "output" / "cnn"

    counts = split_classification_dataset(
        images_dir, labels_dir, output_dir, 50, 25, 25, class_names=CLASS_NAMES, seed=0
    )

    assert sum(counts.values()) == 16
    copied = list(output_dir.rglob("*.jpg"))
    assert len(copied) == 16


def test_split_detection_dataset_keeps_image_label_pairs(tmp_path):
    images_dir, labels_dir = _make_raw_dataset(tmp_path)
    output_dir = tmp_path / "output" / "yolo"

    counts = split_detection_dataset(images_dir, labels_dir, output_dir, 50, 25, 25, seed=0)

    assert sum(counts.values()) == 16
    for split in ("train", "valid", "test"):
        images = list((output_dir / split / "images").glob("*.jpg"))
        labels = list((output_dir / split / "labels").glob("*.txt"))
        assert len(images) == len(labels) == counts[split]


def test_check_labels_and_images_flags_mismatch(tmp_path):
    images_dir, labels_dir = _make_raw_dataset(tmp_path, num_per_class=1)
    (images_dir / "orphan.jpg").write_bytes(b"fake-image")

    report = check_labels_and_images(images_dir, labels_dir)

    assert "orphan" in report["images_without_labels"]
    assert report["num_images"] == 5
    assert report["num_labels"] == 4
