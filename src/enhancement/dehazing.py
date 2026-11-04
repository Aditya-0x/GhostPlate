"""
Dark Channel Prior (DCP) De-hazing Pre-Filter with Guided Filter Refinement
Based on:
- He, K., Sun, J., & Tang, X. "Single Image Haze Removal Using Dark Channel Prior." TPAMI 2011.
- He, K., Sun, J., & Tang, X. "Guided Image Filtering." ECCV 2010 / TPAMI 2013.
"""

import cv2
import numpy as np


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int = 40, eps: float = 1e-3) -> np.ndarray:
    """
    Guided filter for edge-preserving transmission map refinement.
    O(N) linear time filter using box filters.

    Args:
        guide: Grayscale guidance image normalized to [0, 1] (float32/float64).
        src: Filtering input image (e.g. coarse transmission map) in [0, 1].
        radius: Local filter patch radius.
        eps: Regularization parameter penalizing large variance.

    Returns:
        Refined transmission map of same shape as src.
    """
    ksize = (2 * radius + 1, 2 * radius + 1)

    mean_I = cv2.boxFilter(guide, cv2.CV_64F, ksize)
    mean_p = cv2.boxFilter(src, cv2.CV_64F, ksize)
    mean_Ip = cv2.boxFilter(guide * src, cv2.CV_64F, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(guide * guide, cv2.CV_64F, ksize)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_64F, ksize)

    q = mean_a * guide + mean_b
    return q.astype(np.float32)


def get_dark_channel(image: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """
    Computes the dark channel of an image.
    J^{dark}(x) = min_{c in {r,g,b}} ( min_{y in Omega(x)} (I^c(y)) )

    Args:
        image: Color image in BGR format, float32 [0, 1].
        patch_size: Local window size Omega(x).

    Returns:
        Dark channel 2D matrix (float32 [0, 1]).
    """
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel


def estimate_atmospheric_light(image: np.ndarray, dark_channel: np.ndarray, top_percent: float = 0.001) -> np.ndarray:
    """
    Estimates the global atmospheric light A from the brightest pixels in the dark channel.

    Args:
        image: Color image in BGR format, float32 [0, 1].
        dark_channel: Dark channel of the image.
        top_percent: Fraction of brightest pixels to sample (default 0.1%).

    Returns:
        Atmospheric light vector A: shape (1, 1, 3).
    """
    h, w = dark_channel.shape
    num_pixels = h * w
    num_brightest = max(int(num_pixels * top_percent), 1)

    dark_flat = dark_channel.ravel()
    img_flat = image.reshape((-1, 3))

    # Indices of top brightest pixels in dark channel
    indices = np.argpartition(dark_flat, -num_brightest)[-num_brightest:]

    # Pick the pixel in image with highest intensity among these candidates
    candidates = img_flat[indices]
    intensity = np.mean(candidates, axis=1)
    max_idx = np.argmax(intensity)
    A = candidates[max_idx].reshape((1, 1, 3))

    # Clamp A to avoid division by zero or extreme blowing
    return np.clip(A, 0.05, 0.98)


def estimate_transmission(image: np.ndarray, A: np.ndarray, patch_size: int = 15, omega: float = 0.95) -> np.ndarray:
    """
    Estimates the coarse transmission map:
    t~(x) = 1 - omega * min_c ( min_y ( I^c(y) / A^c ) )

    Args:
        image: Color image in BGR format, float32 [0, 1].
        A: Atmospheric light vector (1, 1, 3).
        patch_size: Local patch size for morphological erosion.
        omega: Aerial perspective constant (keeps a slight natural haze depth cue).

    Returns:
        Coarse transmission map (float32 [0, 1]).
    """
    norm_img = image / A
    raw_transmission = 1.0 - omega * get_dark_channel(norm_img, patch_size=patch_size)
    return raw_transmission.astype(np.float32)


def recover_radiance(
    image: np.ndarray,
    A: np.ndarray,
    transmission: np.ndarray,
    t0: float = 0.1
) -> np.ndarray:
    """
    Recovers the dehazed scene radiance:
    J(x) = (I(x) - A) / max(t(x), t0) + A

    Args:
        image: Color image in BGR format, float32 [0, 1].
        A: Atmospheric light vector (1, 1, 3).
        transmission: Refined transmission map.
        t0: Minimum transmission lower bound to prevent extreme noise amplification.

    Returns:
        Dehazed image in BGR format, float32 [0, 1].
    """
    t_clamped = np.clip(transmission, t0, 1.0)
    t_3ch = np.repeat(t_clamped[:, :, np.newaxis], 3, axis=2)
    J = (image - A) / t_3ch + A
    return np.clip(J, 0.0, 1.0)


def dark_channel_dehaze(
    image: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
    t0: float = 0.1,
    guided_radius: int = 40,
    guided_eps: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full single-image Dark Channel Prior dehazing pipeline.

    Args:
        image: Input BGR image (uint8 [0, 255] or float32 [0, 1]).
        patch_size: Patch size for dark channel calculation.
        omega: Haze preservation factor (0.95).
        t0: Transmission lower bound threshold.
        guided_radius: Radius for guided filter refinement.
        guided_eps: Regularization term for guided filter.

    Returns:
        tuple of:
          - dehazed_img: uint8 BGR dehazed image [0, 255]
          - refined_t: float32 transmission map [0, 1]
          - dark_ch: float32 dark channel [0, 1]
    """
    if image.dtype == np.uint8:
        img_f = image.astype(np.float32) / 255.0
    else:
        img_f = image.astype(np.float32)

    # 1. Dark Channel
    dark_ch = get_dark_channel(img_f, patch_size=patch_size)

    # 2. Atmospheric Light
    A = estimate_atmospheric_light(img_f, dark_ch)

    # 3. Coarse Transmission Map
    coarse_t = estimate_transmission(img_f, A, patch_size=patch_size, omega=omega)

    # 4. Edge-preserving Guided Filter Refinement
    gray_guide = cv2.cvtColor((img_f * 255.0).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    refined_t = guided_filter(gray_guide, coarse_t, radius=guided_radius, eps=guided_eps)
    refined_t = np.clip(refined_t, 0.0, 1.0)

    # 5. Scene Radiance Recovery
    radiance = recover_radiance(img_f, A, refined_t, t0=t0)

    dehazed_uint8 = (radiance * 255.0).round().astype(np.uint8)
    return dehazed_uint8, refined_t, dark_ch

