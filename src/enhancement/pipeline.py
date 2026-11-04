"""
Unified Adverse Weather Enhancement Pipeline
Integrates Dark Channel Prior De-hazing, Headlight Glare Suppression, and Rain Deraining.
"""

import cv2
import numpy as np
from .dehazing import dark_channel_dehaze
from .glare_reduction import detect_glare_mask, suppress_glare, adaptive_clahe
from .derain import remove_rain_streaks


class AdverseWeatherEnhancer:
    """
    Unified pre-filter pipeline for license plate recognition under fog, rain, and glare.
    """

    def __init__(
        self,
        enable_dehaze: bool = True,
        enable_glare_suppression: bool = True,
        enable_derain: bool = True,
        enable_clahe: bool = True,
        patch_size: int = 15,
        omega: float = 0.95,
        t0: float = 0.1
    ):
        self.enable_dehaze = enable_dehaze
        self.enable_glare_suppression = enable_glare_suppression
        self.enable_derain = enable_derain
        self.enable_clahe = enable_clahe
        self.patch_size = patch_size
        self.omega = omega
        self.t0 = t0

    def enhance(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """
        Processes a weather-degraded image through the multi-stage restoration pipeline.

        Args:
            image: BGR uint8 input image.

        Returns:
            Dictionary containing:
              - 'enhanced': Final pre-filtered image
              - 'dehazed': Dehazed intermediate
              - 'transmission_map': Refined transmission map
              - 'dark_channel': Dark channel prior
              - 'glare_mask': Glare/flare binary mask
              - 'derained': Intermediate after rain attenuation
        """
        result = {
            "original": image.copy(),
            "enhanced": image.copy(),
            "dehazed": None,
            "transmission_map": None,
            "dark_channel": None,
            "glare_mask": None,
            "derained": None
        }

        current = image.copy()

        # Step 1: Headlight Glare Reduction (before dehazing to prevent blown lights from corrupting atmospheric light A)
        if self.enable_glare_suppression:
            glare_mask = detect_glare_mask(current, threshold=225)
            result["glare_mask"] = glare_mask
            current = suppress_glare(current, glare_mask=glare_mask, gamma_glare=1.7)

        # Step 2: Rain Streak Attenuation
        if self.enable_derain:
            current = remove_rain_streaks(current)
            result["derained"] = current.copy()

        # Step 3: Dark Channel Prior De-hazing with Guided Filter
        if self.enable_dehaze:
            dehazed, transmission, dark_channel = dark_channel_dehaze(
                current,
                patch_size=self.patch_size,
                omega=self.omega,
                t0=self.t0
            )
            current = dehazed
            result["dehazed"] = dehazed
            result["transmission_map"] = transmission
            result["dark_channel"] = dark_channel

        # Step 4: Adaptive Contrast Enhancement (CLAHE) on LAB Luminance
        if self.enable_clahe:
            current = adaptive_clahe(current, clip_limit=2.0, tile_grid_size=(8, 8))

        result["enhanced"] = current
        return result

