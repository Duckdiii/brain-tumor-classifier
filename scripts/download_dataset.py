#!/usr/bin/env python
"""CLI: download the pre-split classification dataset from Hugging Face Hub.

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --force
"""

from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download

from brain_tumor.config import Paths

REPO_ID = "Lomuto/brain-tumor-mri-dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Tai lai du da co du lieu")
    parser.add_argument("--paths-config", default="paths.yaml")
    args = parser.parse_args()

    paths = Paths.load(args.paths_config)
    target = paths.classification_train.parent

    if target.exists() and any(target.iterdir()) and not args.force:
        print(f"Da co san: {target} (dung --force de tai lai)")
        return

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=target,
    )
    print(f"Da tai dataset ve: {target}")


if __name__ == "__main__":
    main()
