#!/usr/bin/env python
"""CLI: verify every raw image has a matching YOLO label file and vice versa.

Usage:
    python scripts/check_dataset.py
"""

from __future__ import annotations

import argparse

from brain_tumor.config import Paths
from brain_tumor.data.splitting import check_labels_and_images


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-config", default="paths.yaml")
    args = parser.parse_args()

    paths = Paths.load(args.paths_config)
    report = check_labels_and_images(paths.raw_images, paths.raw_labels)

    print(f"Tong so hinh anh: {report['num_images']}")
    print(f"Tong so nhan: {report['num_labels']}")
    print(f"Hinh anh khong co nhan: {len(report['images_without_labels'])}")
    for name in report["images_without_labels"]:
        print(f"  - {name}")
    print(f"Nhan khong co hinh anh: {len(report['labels_without_images'])}")
    for name in report["labels_without_images"]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
