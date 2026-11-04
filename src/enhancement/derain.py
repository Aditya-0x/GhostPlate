"""
Rain Streak Attenuation and Deraining Pre-Filter
Mitigates high-frequency rain streaks and rain drop blur.
"""

import cv2
import numpy as np


def remove_rain_streaks(
    image: np.ndarray,
    streak_angle: float = 75.0,
    streak_length: int = 15,
    blend_ratio: float = 0.65
) -> np.ndarray:
    """
    Directional morphological filtering and edge-preserving bilateral filtering
    to suppress rain streaks while preserving alphanumeric plate edges.

    Args:
        image: BGR uint8 input image.
        streak_angle: Predominant falling angle of rain in degrees (default ~75 deg).
        streak_length: Length of rain streak structuring element.
        blend_ratio: Blend ratio between filtered base and bilateral detail.

    Returns:
        De-rained BGR uint8 image.
    """
    # 1. Edge-preserving bilateral filter separates base illumination from high-frequency streaks
    bilateral = cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)

    # 2. Directional morphological line filter for rain streak detection
    rad = np.deg2rad(streak_angle)
    dx = int(np.round(np.cos(rad) * streak_length))
    dy = int(np.round(np.sin(rad) * streak_length))

    kernel = np.zeros((abs(dy) * 2 + 1, abs(dx) * 2 + 1), dtype=np.uint8)
    center_y = abs(dy)
    center_x = abs(dx)
    cv2.line(kernel, (center_x - dx, center_y - dy), (center_x + dx, center_y + dy), 1, thickness=1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    streak_mask = cv2.subtract(gray, opened)
    _, streak_bin = cv2.threshold(streak_mask, 25, 255, cv2.THRESH_BINARY)

    # Dilate streak mask slightly
    streak_dilated = cv2.dilate(streak_bin, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    streak_weight = cv2.GaussianBlur(streak_dilated.astype(np.float32) / 255.0, (5, 5), 0)
    streak_weight_3ch = np.repeat(streak_weight[:, :, np.newaxis], 3, axis=2)

    # Where streaks are detected, replace with bilateral smoothed base
    derained = (1.0 - streak_weight_3ch * blend_ratio) * image.astype(np.float32) + (streak_weight_3ch * blend_ratio) * bilateral.astype(np.float32)

    return np.clip(derained, 0, 255).astype(np.uint8)

