"""
Headlight Glare Suppression and Illumination Balancing Module
Mitigates specular reflection, bloom, and overexposure from vehicle headlights.
"""

import cv2
import numpy as np


def detect_glare_mask(image: np.ndarray, threshold: int = 225, min_area: int = 50) -> np.ndarray:
    """
    Detects specular glare / headlight bloom regions using luminance & saturation criteria.

    Args:
        image: BGR uint8 image.
        threshold: Intensity threshold for specular bloom.
        min_area: Minimum connected pixel area to qualify as glare flare.

    Returns:
        Binary mask (uint8 [0, 255]) marking severe glare zones.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]

    # Glare usually has high value (bright) and relatively low saturation (white wash)
    glare_candidates = (v_channel >= threshold) & (s_channel <= 110)
    glare_mask = (glare_candidates * 255).astype(np.uint8)

    # Morphological dilation to encompass blooming halos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated_mask = cv2.dilate(glare_mask, kernel, iterations=2)

    # Filter out tiny noise specks
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(dilated_mask)
    clean_mask = np.zeros_like(dilated_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean_mask[labels == i] = 255

    return clean_mask


def adaptive_clahe(image: np.ndarray, clip_limit: float = 2.5, tile_grid_size: tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) on the LAB L-channel.
    Enhances contrast in low-light and washed-out plate characters.

    Args:
        image: BGR uint8 image.
        clip_limit: Threshold for contrast limiting.
        tile_grid_size: Grid size for histogram equalization.

    Returns:
        Contrast-enhanced BGR uint8 image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l)

    merged = cv2.merge((l_enhanced, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def multi_scale_retinex(image: np.ndarray, sigma_list: list[int] = [15, 80, 250]) -> np.ndarray:
    """
    Multi-Scale Retinex (MSR) for dynamic range compression and shadow detail recovery.

    Args:
        image: BGR uint8 image.
        sigma_list: Gaussian blur kernel scale factors.

    Returns:
        Illumination-balanced BGR uint8 image.
    """
    img_f = image.astype(np.float64) + 1.0  # Avoid log(0)
    retinex = np.zeros_like(img_f)
    weight = 1.0 / len(sigma_list)

    for sigma in sigma_list:
        blur = cv2.GaussianBlur(img_f, (0, 0), sigma)
        retinex += weight * (np.log(img_f) - np.log(blur + 1.0))

    # Normalize back to 0-255 using min-max scaling per channel
    msr_img = np.zeros_like(image, dtype=np.uint8)
    for c in range(3):
        ch = retinex[:, :, c]
        min_val, max_val = np.percentile(ch, 1), np.percentile(ch, 99)
        if max_val - min_val > 1e-5:
            scaled = np.clip((ch - min_val) / (max_val - min_val) * 255.0, 0, 255)
        else:
            scaled = np.zeros_like(ch)
        msr_img[:, :, c] = scaled.astype(np.uint8)

    return msr_img


def suppress_glare(
    image: np.ndarray,
    glare_mask: np.ndarray | None = None,
    gamma_glare: float = 1.8,
    blend_feather: int = 15
) -> np.ndarray:
    """
    Suppresses severe headlight glare by applying localized tone mapping / gamma correction.

    Args:
        image: BGR uint8 image.
        glare_mask: Optional binary mask of glare regions. If None, automatically detected.
        gamma_glare: Gamma attenuation for overexposed regions (gamma > 1 pulls down specular highlights).
        blend_feather: Feathering radius for smooth spatial blending.

    Returns:
        Glare-attenuated BGR image.
    """
    if glare_mask is None:
        glare_mask = detect_glare_mask(image)

    if np.count_nonzero(glare_mask) == 0:
        return image.copy()

    # Create feather weight map in [0, 1]
    feather_ksize = (2 * blend_feather + 1, 2 * blend_feather + 1)
    weight_map = cv2.GaussianBlur(glare_mask.astype(np.float32) / 255.0, feather_ksize, 0)
    weight_3ch = np.repeat(weight_map[:, :, np.newaxis], 3, axis=2)

    # Tone-mapped darkened image for saturated regions
    inv_gamma = 1.0 / gamma_glare
    table = np.array([((i / 255.0) ** gamma_glare) * 255 for i in range(256)]).astype(np.uint8)
    attenuated = cv2.LUT(image, table)

    # Smooth spatial alpha blend
    blended = (1.0 - weight_3ch) * image.astype(np.float32) + weight_3ch * attenuated.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)

