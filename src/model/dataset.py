"""
Dataset Loaders and Character Data Generators
Provides both synthetic alphanumeric generator and Kaggle dataset loaders
with consistent binary polarity (White character strokes on Black background).
"""

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .cnn_model import CHARS, CHAR2IDX


class SyntheticCharDataset(Dataset):
    """
    On-the-fly synthetic character generator for robust training.
    Applies diverse fonts, rotations, noise, elastic deformations, and contrast variations.
    Produces White character strokes (1.0) on Black background (0.0).
    """

    def __init__(self, samples_per_class: int = 500, img_size: int = 32):
        self.samples_per_class = samples_per_class
        self.img_size = img_size
        self.total_samples = len(CHARS) * samples_per_class

        self.cv2_fonts = [
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_PLAIN,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_COMPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
            cv2.FONT_HERSHEY_COMPLEX_SMALL,
        ]

    def __len__(self) -> int:
        return self.total_samples

    def _render_char(self, char: str) -> np.ndarray:
        canvas = np.zeros((self.img_size * 2, self.img_size * 2), dtype=np.uint8)

        font = random.choice(self.cv2_fonts)
        scale = random.uniform(0.7, 1.2)
        thickness = random.choice([1, 2])

        text_size, _ = cv2.getTextSize(char, font, scale, thickness)
        tw, th = text_size
        cx = (canvas.shape[1] - tw) // 2
        cy = (canvas.shape[0] + th) // 2

        # Draw WHITE character (255) on BLACK background (0)
        cv2.putText(canvas, char, (cx, cy), font, scale, 255, thickness, cv2.LINE_AA)

        angle = random.uniform(-10, 10)
        tx = random.uniform(-2, 2)
        ty = random.uniform(-2, 2)
        shear = random.uniform(-0.10, 0.10)

        center = (canvas.shape[1] // 2, canvas.shape[0] // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot_mat[0, 2] += tx
        rot_mat[1, 2] += ty
        rot_mat[0, 1] += shear

        transformed = cv2.warpAffine(canvas, rot_mat, (canvas.shape[1], canvas.shape[0]))

        # Fit tight bounding box to strokes, pad into square, and resize to img_size
        pts = np.argwhere(transformed > 30)
        if len(pts) > 5:
            y0, x0 = pts.min(axis=0)
            y1, x1 = pts.max(axis=0) + 1
            tight = transformed[y0:y1, x0:x1]
            h, w = tight.shape
            margin = 4
            max_d = max(h, w)
            square = np.zeros((max_d + 2 * margin, max_d + 2 * margin), dtype=np.uint8)
            oy = (square.shape[0] - h) // 2
            ox = (square.shape[1] - w) // 2
            square[oy:oy + h, ox:ox + w] = tight
            crop = cv2.resize(square, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        else:
            crop = cv2.resize(transformed, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        if random.random() > 0.5:
            noise = np.random.normal(0, random.uniform(3, 12), crop.shape)
            crop = np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if random.random() > 0.7:
            crop = cv2.GaussianBlur(crop, (3, 3), 0)

        return crop

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        class_idx = idx % len(CHARS)
        char = CHARS[class_idx]
        img = self._render_char(char)
        tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        return tensor, class_idx


class PlateCharDataset(Dataset):
    """
    Dataset loader for segmented character image files stored in class subdirectories.
    Structure: root_dir/class_name/xxx.png
    Binarizes images so character strokes are WHITE (1.0) and background is BLACK (0.0).
    """

    def __init__(self, root_dir: str, img_size: int = 32):
        self.root_dir = root_dir
        self.img_size = img_size
        self.samples = []

        if os.path.isdir(root_dir):
            for class_name in sorted(os.listdir(root_dir)):
                class_path = os.path.join(root_dir, class_name)
                # Map 'O' to '0' if 'O' is not in classes
                mapped_name = class_name.upper()
                if mapped_name == "O":
                    mapped_name = "0"

                if os.path.isdir(class_path) and mapped_name in CHAR2IDX:
                    class_idx = CHAR2IDX[mapped_name]
                    for fname in os.listdir(class_path):
                        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                            self.samples.append((os.path.join(class_path, fname), class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        fpath, label = self.samples[idx]
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            patch = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
            # Otsu thresholding with inversion (Kaggle dataset has dark characters on bright background)
            _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Fit tight bounding box to character and pad into square
            pts = np.argwhere(thresh > 0)
            if len(pts) > 10:
                y0, x0 = pts.min(axis=0)
                y1, x1 = pts.max(axis=0) + 1
                char_patch = thresh[y0:y1, x0:x1]
            else:
                char_patch = thresh

            h, w = char_patch.shape
            margin = 4
            max_dim = max(h, w)
            square = np.zeros((max_dim + 2 * margin, max_dim + 2 * margin), dtype=np.uint8)
            offset_y = (square.shape[0] - h) // 2
            offset_x = (square.shape[1] - w) // 2
            square[offset_y:offset_y + h, offset_x:offset_x + w] = char_patch
            patch = cv2.resize(square, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        tensor = torch.from_numpy(patch).float().unsqueeze(0) / 255.0
        return tensor, label
