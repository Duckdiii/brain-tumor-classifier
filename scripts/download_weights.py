#!/usr/bin/env python
"""CLI: download pretrained model weights from Hugging Face Hub into weights/.

Usage:
    python scripts/download_weights.py
    python scripts/download_weights.py --model cnn
    python scripts/download_weights.py --force
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

from brain_tumor.config import Paths

MODELS = {
    "cnn": {"repo_id": "Lomuto/cnn-brain-tumor-clasification", "filename": "best_model.pth"},
    "vit": {"repo_id": "Lomuto/vit-brain-tumor-classification", "filename": "best_model.pth"},
    "yolov8": {"repo_id": "Lomuto/yolov8-brain-tumor-classification", "filename": "best.pt"},
    "yolov11": {"repo_id": "Lomuto/yolov11-brain-tumor-classification", "filename": "best.pt"},
}


def download_one(name: str, target: Path, force: bool) -> None:
    if target.exists() and not force:
        print(f"[{name}] da co san: {target} (dung --force de tai lai)")
        return

    info = MODELS[name]
    target.parent.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=info["repo_id"],
            filename=info["filename"],
            local_dir=target.parent,
        )
    )
    if downloaded != target:
        downloaded.replace(target)
    print(f"[{name}] da tai ve: {target}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--force", action="store_true", help="Tai lai du da co file")
    parser.add_argument("--paths-config", default="paths.yaml")
    args = parser.parse_args()

    paths = Paths.load(args.paths_config)
    targets = {
        "cnn": paths.cnn_weights,
        "vit": paths.vit_weights,
        "yolov8": paths.yolov8_weights,
        "yolov11": paths.yolo11_weights,
    }

    names = list(MODELS) if args.model == "all" else [args.model]
    for name in names:
        download_one(name, targets[name], args.force)


if __name__ == "__main__":
    main()
