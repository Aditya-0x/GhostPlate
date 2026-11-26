"""
Dark-Channel Prior (DCP) De-hazing & Glare/Rain Mitigation Module
=================================================================
Project: Automated License Plate Character Recognition under Rain, Fog, and Glare
Team: Group 6 — Aditya Verma (12408804), Varun Singh Rajpurohit (12409154),
      Damandeep Saini (12403195)

Implements:
1. Dark Channel Prior (He et al., CVPR 2009) from scratch using OpenCV & NumPy:
   - Local patch minimum dark channel extraction
   - Top-0.1% atmospheric light (A) estimation
   - Initial transmission map estimation
   - Fast O(N) Guided Filter transmission refinement (He et al., ECCV 2010)
   - Scene radiance recovery with lower transmission bound t0
2. Glare & Rain Mitigation Stage:
   - Adaptive luminance-driven gamma correction to tame blown-out headlight glare
   - CLAHE on the L-channel in CIE-LAB color space for local contrast recovery
   - Edge-preserving bilateral filtering to suppress rain streaks without blurring
     alphanumeric character edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass
class DehazeConfig:
    """Configuration parameters for Dark-Channel Prior and glare mitigation."""

    patch_size: int = 15
    omega: float = 0.92
    t_min: float = 0.12
    guided_radius: int = 24
    guided_eps: float = 1e-3
    clahe_clip_limit: float = 2.4
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    bilateral_d: int = 5
    bilateral_sigma_color: float = 35.0
    bilateral_sigma_space: float = 35.0
    enable_glare_suppression: bool = True


def compute_dark_channel(image_rgb_float: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """
    Compute the dark channel of an RGB float image in [0, 1].

    For every pixel x:
        J_dark(x) = min_{y in Omega(x)} ( min_{c in {R,G,B}} I^c(y) )
    """
    min_channel = np.min(image_rgb_float, axis=2)
    kernel_size = max(3, patch_size | 1)  # Ensure odd kernel >= 3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel


def estimate_atmospheric_light(
    image_rgb_float: np.ndarray,
    dark_channel: np.ndarray,
    top_percent: float = 0.001,
) -> np.ndarray:
    """
    Estimate global atmospheric light vector A (shape: (3,)) by selecting the
    brightest 0.1% pixels in the dark channel and choosing the highest-intensity
    RGB pixel(s) among them (capped to avoid blown-out specular headlight bias).
    """
    h, w = dark_channel.shape
    num_pixels = h * w
    num_search = max(1, int(num_pixels * top_percent))

    flat_dark = dark_channel.reshape(num_pixels)
    flat_img = image_rgb_float.reshape(num_pixels, 3)

    # Indices of top 0.1% brightest dark-channel pixels
    top_indices = np.argpartition(flat_dark, -num_search)[-num_search:]
    candidates = flat_img[top_indices]

    # Pick mean of top candidates weighted by luminance to resist isolated headlight spikes
    luminance = 0.299 * candidates[:, 0] + 0.587 * candidates[:, 1] + 0.114 * candidates[:, 2]
    best_k = max(1, min(20, len(candidates)))
    strongest_idx = np.argpartition(luminance, -best_k)[-best_k:]
    atmospheric_light = np.mean(candidates[strongest_idx], axis=0)

    # Clamp to prevent division by zero or extreme highlight clipping
    return np.clip(atmospheric_light, 0.15, 0.96).astype(np.float32)


def guided_filter(
    guide_gray: np.ndarray,
    src_map: np.ndarray,
    radius: int = 24,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Fast O(N) Guided Filter (He et al., ECCV 2010) using OpenCV boxFilter.
    Refines the coarse block-wise transmission map along crisp image edges.
    """
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_I = cv2.boxFilter(guide_gray, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(src_map, cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(guide_gray * src_map, cv2.CV_32F, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide_gray * guide_gray, cv2.CV_32F, ksize)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

    refined = mean_a * guide_gray + mean_b
    return np.clip(refined, 0.0, 1.0)


def estimate_transmission(
    image_rgb_float: np.ndarray,
    atmospheric_light: np.ndarray,
    omega: float = 0.92,
    patch_size: int = 15,
) -> np.ndarray:
    """
    Estimate initial coarse transmission map:
        t_tilde(x) = 1 - omega * DarkChannel(I(x) / A)
    """
    normalized = image_rgb_float / np.maximum(atmospheric_light[None, None, :], 1e-4)
    dark_norm = compute_dark_channel(normalized, patch_size=patch_size)
    transmission = 1.0 - omega * dark_norm
    return np.clip(transmission, 0.05, 1.0).astype(np.float32)


def recover_scene_radiance(
    image_rgb_float: np.ndarray,
    transmission: np.ndarray,
    atmospheric_light: np.ndarray,
    t_min: float = 0.12,
) -> np.ndarray:
    """
    Recover haze-free scene radiance:
        J(x) = (I(x) - A) / max(t(x), t_min) + A
    """
    t_clamped = np.maximum(transmission, t_min)[:, :, None]
    A = atmospheric_light[None, None, :]
    radiance = (image_rgb_float - A) / t_clamped + A
    return np.clip(radiance, 0.0, 1.0).astype(np.float32)


def suppress_glare_and_rain(
    image_bgr: np.ndarray,
    config: DehazeConfig,
) -> np.ndarray:
    """
    Lightweight glare & rain mitigation stage:
    1. Adaptive gamma correction: compresses blown-out headlight glare when
       mean luminance is high and gently lifts dark shadowed plates.
    2. LAB L-channel CLAHE: boosts local contrast of plate characters without
       distorting chromaticity.
    3. Edge-preserving bilateral filter: attenuates fine rain streaks and sensor
       noise while preserving crisp alphanumeric boundaries.
    """
    # Step 1: Adaptive Gamma based on image luminance
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean_lum = float(np.mean(gray)) / 255.0
    highlight_ratio = float(np.mean(gray > 220))

    # If glare/high brightness dominates, gamma > 1 darkens highlights;
    # if underexposed, gamma < 1 lifts shadows.
    if highlight_ratio > 0.12 or mean_lum > 0.62:
        gamma = np.clip(1.0 + 1.45 * (mean_lum - 0.48) + 0.8 * highlight_ratio, 1.05, 2.15)
    elif mean_lum < 0.35:
        gamma = np.clip(0.65 + 0.7 * mean_lum, 0.60, 0.95)
    else:
        gamma = 1.0

    if abs(gamma - 1.0) > 0.03:
        lut = np.array(
            [np.clip(((i / 255.0) ** gamma) * 255.0, 0, 255) for i in range(256)],
            dtype=np.uint8,
        )
        gamma_corrected = cv2.LUT(image_bgr, lut)
    else:
        gamma_corrected = image_bgr.copy()

    # Step 2: CLAHE on L-channel in CIE-LAB color space
    lab = cv2.cvtColor(gamma_corrected, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_grid,
    )
    l_enhanced = clahe.apply(l_chan)
    enhanced_bgr = cv2.cvtColor(cv2.merge([l_enhanced, a_chan, b_chan]), cv2.COLOR_LAB2BGR)

    # Step 3: Edge-preserving bilateral filtering to suppress rain streaks
    filtered_bgr = cv2.bilateralFilter(
        enhanced_bgr,
        d=config.bilateral_d,
        sigmaColor=config.bilateral_sigma_color,
        sigmaSpace=config.bilateral_sigma_space,
    )
    return filtered_bgr


def dehaze_and_enhance(
    image_bgr: np.ndarray,
    config: DehazeConfig | None = None,
    return_intermediates: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Full Dark-Channel Prior De-hazing + Glare/Rain Mitigation pipeline.

    Args:
        image_bgr: Input uint8 BGR image (OpenCV convention).
        config: Optional DehazeConfig overriding defaults.
        return_intermediates: If True, also returns dark channel, raw/refined
            transmission maps, and atmospheric light estimate for visualization.

    Returns:
        Restored uint8 BGR image, or (restored_bgr, intermediates_dict).
    """
    if config is None:
        config = DehazeConfig()

    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty.")

    # Convert BGR uint8 -> RGB float32 in [0, 1]
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    guide_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # 1. Dark Channel Extraction
    dark_channel = compute_dark_channel(img_rgb, patch_size=config.patch_size)

    # 2. Atmospheric Light Estimation
    atmospheric_light = estimate_atmospheric_light(img_rgb, dark_channel)

    # 3. Coarse Transmission Map Estimation
    raw_transmission = estimate_transmission(
        img_rgb,
        atmospheric_light,
        omega=config.omega,
        patch_size=config.patch_size,
    )

    # 4. Guided Filter Refinement
    refined_transmission = guided_filter(
        guide_gray,
        raw_transmission,
        radius=config.guided_radius,
        eps=config.guided_eps,
    )

    # 5. Scene Radiance Recovery
    radiance_rgb = recover_scene_radiance(
        img_rgb,
        refined_transmission,
        atmospheric_light,
        t_min=config.t_min,
    )
    dehazed_bgr = cv2.cvtColor((radiance_rgb * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # 6. Glare & Rain Mitigation Stage
    if config.enable_glare_suppression:
        restored_bgr = suppress_glare_and_rain(dehazed_bgr, config)
    else:
        restored_bgr = dehazed_bgr

    if not return_intermediates:
        return restored_bgr

    intermediates = {
        "dark_channel": (np.clip(dark_channel, 0.0, 1.0) * 255.0).astype(np.uint8),
        "raw_transmission": (np.clip(raw_transmission, 0.0, 1.0) * 255.0).astype(np.uint8),
        "refined_transmission": (np.clip(refined_transmission, 0.0, 1.0) * 255.0).astype(np.uint8),
        "dcp_only_bgr": dehazed_bgr,
        "atmospheric_light": atmospheric_light,
    }
    return restored_bgr, intermediates


if __name__ == "__main__":
    # Quick self-test on a synthetic foggy gradient
    dummy = np.full((180, 480, 3), 210, dtype=np.uint8)
    cv2.putText(dummy, "DL01AB1234", (35, 115), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (115, 115, 115), 6)
    restored, maps = dehaze_and_enhance(dummy, return_intermediates=True)
    print(
        f"DCP self-test OK | Input std: {dummy.std():.2f} -> Restored std: {restored.std():.2f} "
        f"| A={maps['atmospheric_light'].round(3)}"
    )
