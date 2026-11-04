"""
Training Script for Accurate License Plate Character Recognizer
Trains on synthetic vehicle alphanumeric characters + Kaggle real characters.
"""

import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset

from src.model import LightweightCharCNN, SyntheticCharDataset, CHARS
from src.model.dataset import PlateCharDataset


def train_plate_model(
    kaggle_dir: str = "data/kaggle_chars/CNN letter Dataset",
    synthetic_samples: int = 350,
    epochs: int = 8,
    batch_size: int = 128,
    lr: float = 1.5e-3,
    save_path: str = "models/char_cnn_lightweight.pth"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Training Accurate Character Recognizer on: {device}")

    # 1. Dataset: High-quality synthetic vehicle plate typography (0-9, A-Z)
    syn_dataset = SyntheticCharDataset(samples_per_class=synthetic_samples)
    print(f"[*] Generated {len(syn_dataset)} synthetic vehicle typography samples")

    # 2. Blend with balanced Kaggle real character dataset if present
    subsets = [syn_dataset]
    if os.path.exists(kaggle_dir):
        kag_dataset = PlateCharDataset(kaggle_dir)
        print(f"[*] Loaded {len(kag_dataset)} Kaggle real character samples")
        class_indices = {}
        for idx, (_, label) in enumerate(kag_dataset.samples):
            class_indices.setdefault(label, []).append(idx)
        
        selected_indices = []
        for label, indices in class_indices.items():
            selected = random.sample(indices, min(len(indices), 250))
            selected_indices.extend(selected)
        
        kaggle_balanced = Subset(kag_dataset, selected_indices)
        print(f"[*] Balanced Kaggle subset: {len(kaggle_balanced)} samples")
        subsets.append(kaggle_balanced)

    full_dataset = ConcatDataset(subsets)
    print(f"[*] Total training samples: {len(full_dataset)}")

    val_size = int(len(full_dataset) * 0.12)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = LightweightCharCNN(num_classes=len(CHARS)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        train_acc = (correct_train / total_train) * 100.0
        val_acc = (val_correct / val_total) * 100.0
        print(f"[Epoch {epoch:02d}/{epochs:02d}] Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

    print(f"[+] Training complete. Best Val Acc: {best_val_acc:.2f}%. Model saved to: {save_path}")


if __name__ == "__main__":
    train_plate_model()
