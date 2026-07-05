#!/usr/bin/env python
"""CLI: train one of the four models using its ``configs/<model>.yaml``.

Usage:
    python scripts/train.py --model cnn
    python scripts/train.py --model vit
    python scripts/train.py --model yolov8
    python scripts/train.py --model yolov11
"""

from __future__ import annotations

import argparse

import torch

from brain_tumor.config import Paths, load_yaml
from brain_tumor.constants import NUM_CLASSES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["cnn", "vit", "yolov8", "yolov11"], required=True)
    parser.add_argument("--paths-config", default="paths.yaml")
    args = parser.parse_args()

    paths = Paths.load(args.paths_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def progress(percent: int, message: str) -> None:
        print(f"[{percent:3d}%] {message}")

    if args.model == "cnn":
        from brain_tumor.training.train_cnn import train

        cfg = load_yaml("cnn.yaml")
        train(
            paths.classification_train,
            paths.classification_val,
            paths.cnn_weights,
            num_classes=NUM_CLASSES,
            image_size=cfg["image_size"],
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            device=device,
            progress_callback=progress,
        )
    elif args.model == "vit":
        from brain_tumor.training.train_vit import train

        cfg = load_yaml("vit.yaml")
        train(
            paths.classification_train,
            paths.classification_val,
            paths.vit_weights,
            num_classes=NUM_CLASSES,
            model_name=cfg["model"],
            image_size=cfg["image_size"],
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            warmup_epochs=cfg["warmup_epochs"],
            gradient_clipping=cfg["gradient_clipping"],
            device=device,
            progress_callback=progress,
        )
    else:
        from brain_tumor.training.train_yolo import train

        cfg = load_yaml(f"{args.model}.yaml")
        runs = paths.yolov8_runs if args.model == "yolov8" else paths.yolo11_runs
        weights_out = paths.yolov8_weights if args.model == "yolov8" else paths.yolo11_weights
        extra = {
            k: v
            for k, v in cfg.items()
            if k not in {"task", "model", "epochs", "imgsz", "batch", "device"}
        }
        train(
            base_model=cfg["model"],
            data_dir=paths.classification_train.parent,
            project=runs,
            weights_out=weights_out,
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            device=cfg.get("device", device),
            **extra,
        )

    print("Training hoan tat.")


if __name__ == "__main__":
    main()
