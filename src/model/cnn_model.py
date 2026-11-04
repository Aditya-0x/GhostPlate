"""
Lightweight Character Recognition CNN Architecture in PyTorch
Optimized for high-speed inference on laptop CPU and Google Colab free tiers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 36 alphanumeric classes: 0-9 and A-Z
CHARS = [str(i) for i in range(10)] + [chr(i) for i in range(ord("A"), ord("Z") + 1)]
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
IDX2CHAR = {i: c for i, c in enumerate(CHARS)}


class LightweightCharCNN(nn.Module):
    """
    Lightweight Deep CNN for 36-class alphanumeric character recognition.
    Features:
      - 3 convolutional blocks with Batch Normalization
      - LeakyReLU activations to prevent dead neurons
      - Dropout regularization to avoid overfitting
      - < 300k parameters for sub-millisecond CPU latency
    """

    def __init__(self, num_classes: int = 36, dropout_rate: float = 0.3):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))  # 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        logits = self.classifier(x)
        return logits

    @torch.no_grad()
    def predict_char(self, char_tensor: torch.Tensor, device: str = "cpu") -> tuple[str, float]:
        """
        Runs inference on a single normalized character tensor (1, 1, 32, 32).
        Returns predicted character string and confidence probability.
        """
        self.eval()
        char_tensor = char_tensor.to(device)
        logits = self.forward(char_tensor)
        probs = F.softmax(logits, dim=-1)
        conf, pred_idx = torch.max(probs, dim=-1)
        char = IDX2CHAR[pred_idx.item()]
        return char, conf.item()

