#!/usr/bin/env python
"""CLI: split the raw dataset into CNN/ViT (ImageFolder) or YOLO layouts.

Usage:
    python scripts/split_dataset.py --target cnn --train 70 --val 15 --test 15
    python scripts/split_dataset.py --target yolo --train 70 --val 15 --test 15
"""

from __future__ import annotations

import argparse

from brain_tumor.config import Paths
from brain_tumor.data.splitting import split_classification_dataset, split_detection_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=["cnn", "vit", "yolo"], required=True)
    parser.add_argument("--train", type=float, default=70.0, help="Train ratio (0-100)")
    parser.add_argument("--val", type=float, default=15.0, help="Val ratio (0-100)")
    parser.add_argument("--test", type=float, default=15.0, help="Test ratio (0-100)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--paths-config", default="paths.yaml")
    args = parser.parse_args()

    paths = Paths.load(args.paths_config)

    def progress(percent: int, message: str) -> None:
        print(f"[{percent:3d}%] {message}")

    if args.target == "yolo":
        output_dir = paths.yolo_train.parent
        counts = split_detection_dataset(
            paths.raw_images,
            paths.raw_labels,
            output_dir,
            args.train,
            args.val,
            args.test,
            seed=args.seed,
            progress_callback=progress,
        )
    else:
        output_dir = (paths.cnn_train if args.target == "cnn" else paths.vit_train).parent
        counts = split_classification_dataset(
            paths.raw_images,
            paths.raw_labels,
            output_dir,
            args.train,
            args.val,
            args.test,
            seed=args.seed,
            progress_callback=progress,
        )

    print("Hoan tat chia dataset:")
    for split, count in counts.items():
        print(f"  {split}: {count} anh")


if __name__ == "__main__":
    main()
