"""
License Plate Localization & Character Segmentation Module (OpenCV)
===================================================================
Project: Automated License Plate Character Recognition under Rain, Fog, and Glare
Team: Group 6 — Aditya Verma (12408804), Varun Singh Rajpurohit (12409154),
      Damandeep Saini (12403195)

Implements:
1. License Plate Localization:
   - Morphological blackhat + Sobel-X vertical edge density detection
   - Morphological closing + contour geometry filtering (aspect ratio 2.0–6.2)
   - Automatic clean fallback for already pre-cropped license plate images.
2. Character Segmentation:
   - Adaptive Gaussian thresholding (handles uneven glare & fog gradients)
   - Connected-component analysis with aspect ratio, area, and vertical-span filters
   - Left-to-right bounding-box sorting and 32x32 centered canvas normalization
     matching the Lightweight PyTorch CNN input specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class CharacterSegment:
    """Single segmented character image and its bounding box on the plate crop."""

    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    crop_gray: np.ndarray            # Normalized 32x32 uint8 binary/grayscale crop


@dataclass
class SegmentationResult:
    """Full output of plate localization and character segmentation."""

    plate_bbox: Tuple[int, int, int, int]
    plate_crop_bgr: np.ndarray
    plate_binary: np.ndarray
    characters: List[CharacterSegment]
    used_precrop_fallback: bool
    annotated_plate_bgr: np.ndarray


def localize_license_plate(
    image_bgr: np.ndarray,
    min_aspect: float = 1.8,
    max_aspect: float = 6.5,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], bool]:
    """
    Detect and crop the license plate region from a vehicle image using
    morphological edge density and contour filtering.

    If the input image already has plate-like aspect ratio (>= 2.1) or no strong
    sub-contour is found, falls back cleanly to the full image (pre-cropped mode).
    """
    h, w = image_bgr.shape[:2]
    img_aspect = w / max(float(h), 1.0)

    # Fast check: if input is already a tight pre-cropped plate (wide aspect, low height)
    if img_aspect >= 2.35 and h <= 220:
        return image_bgr.copy(), (0, 0, w, h), True

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Emphasize dark characters on bright plate via morphological Blackhat
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)

    # Highlight dense vertical character strokes with Sobel-X
    grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
    grad_x = np.absolute(grad_x)
    max_val = float(np.max(grad_x))
    if max_val > 1e-5:
        grad_x = (255.0 * (grad_x / max_val)).astype(np.uint8)
    else:
        grad_x = np.zeros_like(gray)

    # Smooth & close horizontally to merge character edges into a single plate blob
    blurred = cv2.GaussianBlur(grad_x, (5, 5), 0)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (23, 7))
    closed = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, close_kernel)
    _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Clean up tiny isolated speckles
    thresh = cv2.erode(thresh, None, iterations=1)
    thresh = cv2.dilate(thresh, None, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = float(h * w)

    best_score = -1.0
    best_bbox: Tuple[int, int, int, int] | None = None

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if ch <= 0 or cw <= 0:
            continue
        aspect = cw / float(ch)
        area_ratio = (cw * ch) / img_area

        if min_aspect <= aspect <= max_aspect and 0.015 <= area_ratio <= 0.65:
            # Score candidate by edge density and proximity to canonical plate aspect (~3.5)
            roi_energy = float(np.mean(grad_x[y : y + ch, x : x + cw]))
            aspect_penalty = abs(aspect - 3.6) * 12.0
            score = roi_energy * np.sqrt(area_ratio) - aspect_penalty
            if score > best_score:
                best_score = score
                best_bbox = (x, y, cw, ch)

    if best_bbox is None:
        return image_bgr.copy(), (0, 0, w, h), True

    x, y, cw, ch = best_bbox
    pad_x = max(2, int(cw * 0.04))
    pad_y = max(2, int(ch * 0.08))
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w, x + cw + pad_x)
    y1 = min(h, y + ch + pad_y)

    return image_bgr[y0:y1, x0:x1].copy(), (x0, y0, x1 - x0, y1 - y0), False


def normalize_character_crop(binary_char: np.ndarray, output_size: int = 32) -> np.ndarray:
    """
    Center a segmented binary character crop onto a square 32x32 canvas
    preserving aspect ratio with clean padding (white glyph on black background).
    """
    ch, cw = binary_char.shape[:2]
    if ch == 0 or cw == 0:
        return np.zeros((output_size, output_size), dtype=np.uint8)

    inner = output_size - 6  # 3px margin on all sides
    scale = min(inner / float(cw), inner / float(ch))
    new_w = max(1, int(round(cw * scale)))
    new_h = max(1, int(round(ch * scale)))

    resized = cv2.resize(binary_char, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((output_size, output_size), dtype=np.uint8)
    offset_x = (output_size - new_w) // 2
    offset_y = (output_size - new_h) // 2
    canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized
    return canvas


def segment_characters(
    plate_bgr: np.ndarray,
    char_size: int = 32,
) -> Tuple[List[CharacterSegment], np.ndarray, np.ndarray]:
    """
    Segment alphanumeric characters from a localized license plate crop using:
    - Height normalization (fixed 96px height for scale-invariant morphology)
    - Adaptive Gaussian thresholding (robust to residual glare gradients)
    - 8-connected component analysis + geometry filtering
    - Left-to-right sorting by bounding-box x-coordinate.
    """
    target_h = 96
    orig_h, orig_w = plate_bgr.shape[:2]
    scale = target_h / max(float(orig_h), 1.0)
    target_w = max(160, int(round(orig_w * scale)))
    resized_bgr = cv2.resize(plate_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)

    # Mild Gaussian denoise + Adaptive Threshold (dark text -> foreground white 255)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21,
        C=9,
    )

    # Clear thin border frames (bolts, plate frame edges)
    margin_y = max(2, int(target_h * 0.05))
    margin_x = max(2, int(target_w * 0.02))
    binary[:margin_y, :] = 0
    binary[-margin_y:, :] = 0
    binary[:, :margin_x] = 0
    binary[:, -margin_x:] = 0

    # Slight morphological opening to remove stray raindrop speckles
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    candidates: List[Tuple[int, int, int, int, int]] = []
    for label_idx in range(1, num_labels):
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        cw = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        ch = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_idx, cv2.CC_STAT_AREA])

        height_ratio = ch / float(target_h)
        aspect = cw / max(float(ch), 1.0)
        fill_ratio = area / max(float(cw * ch), 1.0)

        # Geometric filters for standard alphanumeric plate characters (including narrow '1'/'I')
        if (
            0.34 <= height_ratio <= 0.88
            and 0.14 <= aspect <= 1.05
            and area >= 65
            and 0.16 <= fill_ratio <= 0.90
        ):
            candidates.append((x, y, cw, ch, label_idx))

    # Suppress nested boxes (e.g., inner hole artifacts) & sort left-to-right
    candidates.sort(key=lambda item: item[0])
    filtered: List[Tuple[int, int, int, int, int]] = []
    for cand in candidates:
        cx, cy, cw, ch, _ = cand
        is_nested = False
        for other in candidates:
            if cand == other:
                continue
            ox, oy, ow, oh, _ = other
            if cx >= ox and cy >= oy and (cx + cw) <= (ox + ow) and (cy + ch) <= (oy + oh):
                is_nested = True
                break
        if not is_nested:
            filtered.append(cand)

    # Cap extreme false positives to max 12 characters sorted left-to-right
    if len(filtered) > 12:
        filtered.sort(key=lambda item: item[2] * item[3], reverse=True)
        filtered = sorted(filtered[:12], key=lambda item: item[0])

    segments: List[CharacterSegment] = []
    annotated = resized_bgr.copy()

    for x, y, cw, ch, label_idx in filtered:
        mask = np.where(labels[y : y + ch, x : x + cw] == label_idx, 255, 0).astype(np.uint8)
        norm_crop = normalize_character_crop(mask, output_size=char_size)
        segments.append(CharacterSegment(bbox=(x, y, cw, ch), crop_gray=norm_crop))
        cv2.rectangle(annotated, (x, y), (x + cw, y + ch), (212, 182, 6), 2)  # Cyan BGR box

    return segments, binary, annotated


def localize_and_segment(
    image_bgr: np.ndarray,
    char_size: int = 32,
) -> SegmentationResult:
    """Run complete plate localization followed by character segmentation."""
    plate_crop, plate_bbox, used_fallback = localize_license_plate(image_bgr)
    segments, binary, annotated = segment_characters(plate_crop, char_size=char_size)
    return SegmentationResult(
        plate_bbox=plate_bbox,
        plate_crop_bgr=plate_crop,
        plate_binary=binary,
        characters=segments,
        used_precrop_fallback=used_fallback,
        annotated_plate_bgr=annotated,
    )


if __name__ == "__main__":
    test_plate = np.full((110, 420, 3), 235, dtype=np.uint8)
    cv2.putText(test_plate, "KA03MG4821", (16, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.85, (20, 20, 20), 5)
    result = localize_and_segment(test_plate)
    print(f"Segmentation self-test OK | Detected {len(result.characters)} characters")
