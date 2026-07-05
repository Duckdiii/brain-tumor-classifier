"""Vision Transformer classifier built on top of ``timm``."""

from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn


def build_vit_model(
    num_classes: int,
    model_name: str = "vit_base_patch16_224",
    pretrained: bool = True,
) -> nn.Module:
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)


def load_vit_checkpoint(
    checkpoint_path: Path | str,
    num_classes: int,
    model_name: str = "vit_base_patch16_224",
    device: str | torch.device = "cpu",
) -> nn.Module:
    model = build_vit_model(num_classes=num_classes, model_name=model_name, pretrained=True)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)
