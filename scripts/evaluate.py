#!/usr/bin/env python
"""CLI: evaluate a single trained model on its validation/test split.

Usage:
    python scripts/evaluate.py --model cnn --split val
    python scripts/evaluate.py --model yolov8 --split val
"""

from __future__ import annotations

import argparse

import torch

from brain_tumor.config import Paths, yolo_dataset_yaml
from brain_tumor.constants import CLASS_NAMES, NUM_CLASSES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cnn", "vit", "yolov8", "yolov10"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--paths-config", default="paths.yaml")
    args = parser.parse_args()

    paths = Paths.load(args.paths_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def progress(percent: int, message: str) -> None:
        print(f"[{percent:3d}%] {message}")

    if args.model == "cnn":
        from brain_tumor.evaluation.evaluate_cnn import evaluate
        from brain_tumor.models.cnn import GeneralCNN

        model = GeneralCNN(num_classes=NUM_CLASSES)
        model.load_state_dict(torch.load(paths.cnn_weights, map_location=device))
        model.to(device)
        split_dir = paths.cnn_val if args.split == "val" else paths.cnn_test
        _, precision = evaluate(
            model, split_dir, paths.cnn_train, CLASS_NAMES, device=device, progress_callback=progress
        )
        print(f"CNN {args.split} precision: {precision:.2f}%")

    elif args.model == "vit":
        from brain_tumor.evaluation.evaluate_vit import evaluate
        from brain_tumor.models.vit import load_vit_checkpoint

        model = load_vit_checkpoint(paths.vit_weights, num_classes=NUM_CLASSES, device=device)
        split_dir = paths.vit_val if args.split == "val" else paths.vit_test
        precision = evaluate(model, split_dir, paths.vit_train, device=device, progress_callback=progress)
        print(f"ViT {args.split} precision: {precision:.2f}%")

    else:
        from brain_tumor.evaluation.evaluate_yolo import evaluate_yolo

        weights = paths.yolov8_weights if args.model == "yolov8" else paths.yolov10_weights
        project = paths.yolov8_runs if args.model == "yolov8" else paths.yolov10_runs
        result = evaluate_yolo(weights, yolo_dataset_yaml(), project, device=0 if device == "cuda" else "cpu")
        print(result)


if __name__ == "__main__":
    main()
