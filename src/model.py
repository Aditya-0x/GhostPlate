"""
Lightweight Depthwise-Separable PyTorch CNN for License Plate Character Recognition
===================================================================================
Project: Automated License Plate Character Recognition under Rain, Fog, and Glare
Team: Group 6 — Aditya Verma (12408804), Varun Singh Rajpurohit (12409154),
      Damandeep Saini (12403195)

Architecture:
- Input: (N, 1, 32, 32) normalized grayscale/binary character crop
- 4 compact convolutional blocks using Depthwise Separable Convolutions
  with BatchNorm2d, ReLU, MaxPool2d, and Dropout for fast training & low latency
  on both Google Colab T4 GPU and standard laptop CPUs (68,004 parameters).
- Output: 36-class logits covering digits 0-9 and uppercase letters A-Z.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHAR_VOCAB: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
NUM_CLASSES: int = len(CHAR_VOCAB)
CHAR_TO_IDX: Dict[str, int] = {ch: idx for idx, ch in enumerate(CHAR_VOCAB)}
IDX_TO_CHAR: Dict[int, str] = {idx: ch for idx, ch in enumerate(CHAR_VOCAB)}


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise 3x3 convolution followed by Pointwise 1x1 convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = [
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class LightweightPlateCNN(nn.Module):
    """
    Compact 4-block CNN for 36-class (0-9, A-Z) alphanumeric recognition.
    Total trainable parameters: 68,004 (~275 KB serialized checkpoint).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, dropout_fc: float = 0.30) -> None:
        super().__init__()

        # Block 1: Standard stem 1 -> 32 (32x32 -> 16x16)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: Depthwise Separable 32 -> 64 (16x16 -> 8x8)
        self.block2 = DepthwiseSeparableBlock(32, 64, pool=True, dropout=0.10)

        # Block 3: Depthwise Separable 64 -> 128 (8x8 -> 4x4)
        self.block3 = DepthwiseSeparableBlock(64, 128, pool=True, dropout=0.15)

        # Block 4: Depthwise Separable 128 -> 192 (4x4 -> 4x4) + Global Avg Pool
        self.block4 = DepthwiseSeparableBlock(128, 192, pool=False, dropout=0.15)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Compact Classifier Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_fc),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.gap(x)
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def save_checkpoint(
    model: LightweightPlateCNN,
    filepath: str,
    val_accuracy: float,
    epoch: int,
) -> None:
    """Save PyTorch model state_dict and metadata."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": CHAR_VOCAB,
            "val_accuracy": float(val_accuracy),
            "epoch": int(epoch),
            "param_count": model.count_parameters(),
        },
        filepath,
    )


def load_model(
    checkpoint_path: str | None = None,
    device: torch.device | str | None = None,
) -> Tuple[LightweightPlateCNN, torch.device]:
    """Instantiate LightweightPlateCNN and load trained weights if present."""
    if device is None:
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)

    model = LightweightPlateCNN(num_classes=NUM_CLASSES).to(resolved_device)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=resolved_device, weights_only=True)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    model.eval()
    return model, resolved_device


@torch.no_grad()
def predict_character_batch(
    model: LightweightPlateCNN,
    char_crops: List[np.ndarray],
    device: torch.device,
) -> Tuple[str, float, List[Tuple[str, float]]]:
    """
    Run batched PyTorch inference on a list of 32x32 uint8 character crops.

    Returns:
        plate_string: Concatenated predicted characters.
        mean_confidence: Average top-1 softmax probability across characters.
        per_char_results: List of (predicted_char, confidence) tuples.
    """
    if not char_crops:
        return "", 0.0, []

    batch_np = np.stack(char_crops, axis=0).astype(np.float32) / 255.0
    batch_np = (batch_np - 0.5) / 0.5  # Normalize to [-1, 1]
    tensor = torch.from_numpy(batch_np[:, None, :, :]).to(device)

    logits = model(tensor)
    probs = F.softmax(logits, dim=1)
    confs, preds = torch.max(probs, dim=1)

    per_char: List[Tuple[str, float]] = []
    for pred_idx, conf in zip(preds.cpu().tolist(), confs.cpu().tolist()):
        per_char.append((IDX_TO_CHAR[pred_idx], float(conf)))

    plate_string = "".join(ch for ch, _ in per_char)
    mean_confidence = float(np.mean([c for _, c in per_char])) if per_char else 0.0
    return plate_string, mean_confidence, per_char


if __name__ == "__main__":
    net = LightweightPlateCNN()
    sample = torch.randn(8, 1, 32, 32)
    out = net(sample)
    print(f"LightweightPlateCNN OK | Params: {net.count_parameters():,} | Output shape: {tuple(out.shape)}")
