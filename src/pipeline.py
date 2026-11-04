"""
End-to-End ALPR Pipeline for Adverse Weather
Connects: Enhancer -> Plate Detector -> Character Segmenter -> Lightweight CNN Recognizer
"""

import os
import cv2
import numpy as np
import torch

from .enhancement import AdverseWeatherEnhancer
from .detector import LicensePlateDetector
from .segmenter import CharacterSegmenter
from .model import LightweightCharCNN, CHARS


class AdverseWeatherALPRPipeline:
    """
    Complete pipeline integrating weather restoration and automatic license plate recognition.
    """

    def __init__(
        self,
        weights_path: str | None = "models/char_cnn_lightweight.pth",
        device: str | None = None,
        enable_enhancement: bool = True
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.enable_enhancement = enable_enhancement

        self.enhancer = AdverseWeatherEnhancer()
        self.detector = LicensePlateDetector()
        self.segmenter = CharacterSegmenter()

        # Initialize CNN model
        self.model = LightweightCharCNN(num_classes=len(CHARS))
        if weights_path and os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def process(self, image: np.ndarray, with_enhancement: bool | None = None) -> dict:
        """
        Executes end-to-end restoration and recognition.

        Args:
            image: BGR uint8 input image.
            with_enhancement: Override enhancement toggle if specified.

        Returns:
            Dictionary containing recognition, detection, and enhancement diagnostics.
        """
        # Auto-downscale ultra-high-resolution images for fast inference
        max_dim = 1280
        ih, iw = image.shape[:2]
        if max(ih, iw) > max_dim:
            scale = max_dim / float(max(ih, iw))
            image = cv2.resize(image, (int(iw * scale), int(ih * scale)), interpolation=cv2.INTER_AREA)

        use_enhancement = self.enable_enhancement if with_enhancement is None else with_enhancement

        enhancement_data = None
        if use_enhancement:
            enhancement_data = self.enhancer.enhance(image)
            work_image = enhancement_data["enhanced"]
        else:
            work_image = image.copy()

        # 1. License Plate Detection
        plate_crop, plate_bbox = self.detector.detect(work_image)

        # 2. Character Segmentation
        char_samples = self.segmenter.segment(plate_crop)

        # 3. Character Recognition with Lightweight CNN
        char_preds = []
        for char_tensor, (cx, cy, cw, ch) in char_samples:
            # char_tensor is (1, 32, 32)
            tensor_batch = torch.from_numpy(char_tensor).unsqueeze(0).to(self.device)  # (1, 1, 32, 32)
            char, conf = self.model.predict_char(tensor_batch, device=self.device)
            char_preds.append((char, conf))

        plate_text = "".join([c for c, _ in char_preds])

        return {
            "plate_text": plate_text,
            "char_preds": char_preds,
            "plate_bbox": plate_bbox,
            "plate_crop": plate_crop,
            "work_image": work_image,
            "enhancement_data": enhancement_data
        }

