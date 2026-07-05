"""Custom CNN classifier (``GeneralCNN`` from the original ``cnn.py``).

The feature-map size is now derived from ``image_size`` instead of being
hardcoded to 640, so the same class also works with the smaller crops used
during quick experiments.
"""

from __future__ import annotations

import torch.nn as nn


class GeneralCNN(nn.Module):
    def __init__(self, num_classes: int, image_size: int = 640):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        feature_output_size = 128 * (image_size // 8) * (image_size // 8)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_output_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
