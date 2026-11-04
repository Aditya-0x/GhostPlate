"""
Character Segmentation Module for License Plate Crops
Extracts, cleans, and standardizes individual alphanumeric characters.
Features Vertical Projection Profile (VPP) slicing and contour-based extraction.
"""

import cv2
import numpy as np


class CharacterSegmenter:
    """
    Segments characters from cropped license plate images.
    Produces normalized (1, 32, 32) tensors with WHITE text on BLACK background.
    """

    def __init__(
        self,
        target_plate_height: int = 64,
        char_output_size: tuple[int, int] = (32, 32)
    ):
        self.target_plate_height = target_plate_height
        self.char_output_size = char_output_size

    def preprocess_plate(self, plate_crop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Standardizes plate size and produces clean binary thresholded mask.
        Characters are WHITE (255) on BLACK (0) background.
        """
        h, w = plate_crop.shape[:2]
        if h == 0 or w == 0:
            return np.zeros((self.target_plate_height, 100), dtype=np.uint8), np.zeros((self.target_plate_height, 100), dtype=np.uint8)

        aspect = float(w) / float(h)
        target_w = int(self.target_plate_height * aspect)
        resized = cv2.resize(plate_crop, (target_w, self.target_plate_height), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Contrast normalization
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Otsu thresholding with inversion (license plates have dark text on light plate)
        _, thresh = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Polarity check: text strokes should be minority (white text on black background)
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)

        # Suppress outer plate borders and mounting frames
        thresh[:int(self.target_plate_height * 0.16), :] = 0
        thresh[int(self.target_plate_height * 0.84):, :] = 0
        thresh[:, :int(target_w * 0.12)] = 0  # Blue IND strip / EU badge
        thresh[:, int(target_w * 0.98):] = 0

        # Remove mounting screw at right edge if present
        screw_cx = int(target_w * 0.95)
        thresh[int(self.target_plate_height * 0.35):int(self.target_plate_height * 0.65), screw_cx - 8:screw_cx + 8] = 0

        # Clean tiny noise specks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return resized, thresh

    def segment(self, plate_crop: np.ndarray, num_chars: int = 10) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
        """
        Segments alphanumeric characters from the plate crop.

        Args:
            plate_crop: BGR uint8 cropped license plate image.
            num_chars: Expected character count for standard vehicle plates (default 10).

        Returns:
            List of tuples: (normalized_char_tensor [float32 (1, 32, 32)], (x, y, w, h))
        """
        _, thresh = self.preprocess_plate(plate_crop)
        plate_h, plate_w = thresh.shape

        # Vertical projection profile to locate active text region
        vpp = np.sum(thresh == 255, axis=0)
        active_cols = np.where(vpp > 0)[0]

        if len(active_cols) < 20:
            # Fallback if binary is too faint: use central text window
            start_x = int(plate_w * 0.12)
            end_x = int(plate_w * 0.96)
        else:
            start_x = active_cols[0]
            end_x = active_cols[-1]

        total_w = end_x - start_x
        char_w = float(total_w) / float(num_chars)

        results = []
        for i in range(num_chars):
            sx = int(start_x + i * char_w)
            ex = int(start_x + (i + 1) * char_w)
            patch = thresh[:, sx:ex]

            # Fit tight bounding box to character strokes
            pts = np.argwhere(patch > 0)
            if len(pts) > 5:
                py0, px0 = pts.min(axis=0)
                py1, px1 = pts.max(axis=0) + 1
                tight = patch[py0:py1, px0:px1]
                norm_tensor = self.normalize_char(tight)
                results.append((norm_tensor, (sx + px0, py0, px1 - px0, py1 - py0)))
            else:
                norm_tensor = self.normalize_char(patch)
                results.append((norm_tensor, (sx, 0, ex - sx, plate_h)))

        return results

    def normalize_char(self, char_patch: np.ndarray) -> np.ndarray:
        """
        Pads and normalizes character patch into square (1, 32, 32) tensor.
        WHITE text (1.0) on BLACK background (0.0).
        """
        h, w = char_patch.shape
        out_h, out_w = self.char_output_size

        if h == 0 or w == 0:
            return np.zeros((1, out_h, out_w), dtype=np.float32)

        margin = 4
        max_dim = max(h, w)
        square = np.zeros((max_dim + 2 * margin, max_dim + 2 * margin), dtype=np.uint8)

        offset_y = (square.shape[0] - h) // 2
        offset_x = (square.shape[1] - w) // 2
        square[offset_y:offset_y + h, offset_x:offset_x + w] = char_patch

        resized = cv2.resize(square, (out_w, out_h), interpolation=cv2.INTER_AREA)
        norm = (resized.astype(np.float32) / 255.0)[np.newaxis, :, :]  # (1, 32, 32)
        return norm
