"""
Physical Adverse Weather Degradation Simulator
Simulates realistic Fog (Koschmieder scattering), Rain streaks, and Headlight Glare.
"""

import random
import cv2
import numpy as np


def add_fog(
    image: np.ndarray,
    beta: float = 1.2,
    atmospheric_light: tuple[int, int, int] = (225, 225, 230)
) -> np.ndarray:
    """
    Simulates optical fog using Koschmieder's physical scattering model:
    I(x) = J(x) * t(x) + A * (1 - t(x))
    where t(x) = exp(-beta * d(x))

    Args:
        image: BGR uint8 input image.
        beta: Extinction coefficient (higher = denser fog).
        atmospheric_light: BGR atmospheric light color vector.

    Returns:
        Foggy BGR uint8 image.
    """
    h, w = image.shape[:2]

    # Synthesize smooth depth gradient d(x): ground is closer, top/horizon is farther
    y_coords = np.linspace(0.2, 1.2, h, dtype=np.float32)[:, np.newaxis]
    depth = np.repeat(y_coords, w, axis=1)

    # Transmission map
    transmission = np.exp(-beta * depth)
    transmission_3ch = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)

    A = np.array(atmospheric_light, dtype=np.float32).reshape((1, 1, 3))
    img_f = image.astype(np.float32)

    foggy = img_f * transmission_3ch + A * (1.0 - transmission_3ch)
    return np.clip(foggy, 0, 255).astype(np.uint8)


def add_rain(
    image: np.ndarray,
    slant_angle: float = 75.0,
    density: float = 0.04,
    streak_length: int = 25,
    streak_thickness: int = 1,
    mist_intensity: float = 0.25
) -> np.ndarray:
    """
    Simulates falling rain streaks and splashing mist.

    Args:
        image: BGR uint8 image.
        slant_angle: Angle of falling rain in degrees.
        density: Rain streak occurrence probability.
        streak_length: Pixel length of streaks.
        streak_thickness: Line thickness.
        mist_intensity: Blurring mist opacity.

    Returns:
        Rain-degraded BGR uint8 image.
    """
    h, w = image.shape[:2]
    rain_layer = np.zeros((h, w), dtype=np.uint8)

    rad = np.deg2rad(slant_angle)
    dx = int(np.round(np.cos(rad) * streak_length))
    dy = int(np.round(np.sin(rad) * streak_length))

    num_drops = int(h * w * density / streak_length)
    for _ in range(num_drops):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        x2 = min(w - 1, max(0, x + dx))
        y2 = min(h - 1, max(0, y + dy))
        intensity = random.randint(180, 255)
        cv2.line(rain_layer, (x, y), (x2, y2), intensity, thickness=streak_thickness)

    # Motion blur rain layer
    rain_blur = cv2.GaussianBlur(rain_layer, (3, 3), 0)
    rain_3ch = np.repeat(rain_blur[:, :, np.newaxis], 3, axis=2)

    # Add diffuse mist
    mist = cv2.GaussianBlur(image, (9, 9), 0)

    # Composite rain streaks onto image
    base = (1.0 - mist_intensity) * image.astype(np.float32) + mist_intensity * mist.astype(np.float32)
    rainy = base + rain_3ch.astype(np.float32) * 0.45
    return np.clip(rainy, 0, 255).astype(np.uint8)


def add_headlight_glare(
    image: np.ndarray,
    glare_centers: list[tuple[int, int]] | None = None,
    glare_radius: int = 90,
    intensity: float = 1.0
) -> np.ndarray:
    """
    Simulates vehicle headlight glare and specular lens bloom.

    Args:
        image: BGR uint8 image.
        glare_centers: List of (x, y) coordinates of glare sources. If None, places 2 headlights.
        glare_radius: Radius sigma of the radial flare.
        intensity: Glare brightness factor.

    Returns:
        BGR uint8 image with headlight glare flares.
    """
    h, w = image.shape[:2]
    if glare_centers is None:
        # Default: two headlights flanking the lower center
        glare_centers = [
            (int(w * 0.22), int(h * 0.65)),
            (int(w * 0.78), int(h * 0.65))
        ]

    glare_map = np.zeros((h, w), dtype=np.float32)
    y_grid, x_grid = np.ogrid[:h, :w]

    for cx, cy in glare_centers:
        dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
        # Core intense burst
        core = np.exp(-dist_sq / (2.0 * (glare_radius * 0.35) ** 2))
        # Outer blooming halo
        halo = 0.4 * np.exp(-dist_sq / (2.0 * glare_radius ** 2))
        glare_map += (core + halo)

    glare_map = np.clip(glare_map * intensity, 0.0, 1.5)
    glare_3ch = np.repeat(glare_map[:, :, np.newaxis], 3, axis=2)

    # Specular white wash
    glare_color = np.array([245, 250, 255], dtype=np.float32).reshape((1, 1, 3))
    corrupted = image.astype(np.float32) + glare_3ch * glare_color * 0.7
    return np.clip(corrupted, 0, 255).astype(np.uint8)


def apply_combined_adverse_weather(
    image: np.ndarray,
    fog_beta: float = 0.8,
    rain_density: float = 0.03,
    with_glare: bool = True
) -> np.ndarray:
    """
    Combines Fog, Rain, and Headlight Glare simultaneously for severe adverse weather testing.
    """
    out = image.copy()
    if fog_beta > 0:
        out = add_fog(out, beta=fog_beta)
    if rain_density > 0:
        out = add_rain(out, density=rain_density)
    if with_glare:
        out = add_headlight_glare(out)
    return out

