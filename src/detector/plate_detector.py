"""
Accurate License Plate Detection & Localization Module
Combines morphological top-hat/black-hat filtering and central geometric priors.
"""

import cv2
import numpy as np


class LicensePlateDetector:
    """
    Detects and localizes vehicle license plates.
    Rejects whole-vehicle false positives and accurately locks onto the plate rectangle.
    """

    def __init__(
        self,
        min_aspect_ratio: float = 2.0,
        max_aspect_ratio: float = 5.5,
        target_aspect_ratio: float = 3.6
    ):
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.target_aspect_ratio = target_aspect_ratio

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """
        Detects and extracts the license plate region.

        Args:
            image: BGR uint8 vehicle image.

        Returns:
            tuple of (plate_crop, (x, y, w, h))
        """
        img_h, img_w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # License plate search region in central lower vehicle bumper
        roi_y1 = int(img_h * 0.40)
        roi_y2 = int(img_h * 0.90)
        roi_x1 = int(img_w * 0.10)
        roi_x2 = int(img_w * 0.90)

        roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
        roi_h, roi_w = roi.shape

        # Morphological Top-Hat and Black-Hat to isolate plate text and boundaries
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        tophat = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, rect_kernel)
        blackhat = cv2.morphologyEx(roi, cv2.MORPH_BLACKHAT, rect_kernel)
        high_freq = cv2.add(tophat, blackhat)

        # Threshold high-frequency character region
        _, thresh = cv2.threshold(high_freq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Close characters horizontally into a coherent plate box
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_bbox = None
        best_score = -1.0

        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            ar = float(bw) / float(bh) if bh > 0 else 0

            # Strictly reject borders touching the outer ROI boundary
            if bx <= 2 or by <= 2 or (bx + bw) >= (roi_w - 2) or (by + bh) >= (roi_h - 2):
                continue

            # Plate dimensions must strictly be within expected vehicle plate proportions
            if (0.18 * img_w <= bw <= 0.60 * img_w) and (0.05 * img_h <= bh <= 0.25 * img_h) and (self.min_aspect_ratio <= ar <= self.max_aspect_ratio):
                cand = roi[by:by+bh, bx:bx+bw]
                std = float(np.std(cand))
                ar_score = 1.0 - abs(ar - self.target_aspect_ratio) / 3.0
                cx_dist = abs((bx + bw / 2.0) - (roi_w / 2.0)) / (roi_w / 2.0)
                center_score = 1.0 - cx_dist

                score = std * 0.4 + ar_score * 0.35 + center_score * 0.25

                if score > best_score:
                    best_score = score
                    # Capture full plate width including badge
                    pad_left = int(bw * 0.08)
                    fx = max(0, bx + roi_x1 - pad_left)
                    fy = by + roi_y1
                    fw = min(img_w - fx, bw + pad_left)
                    fh = bh
                    best_bbox = (fx, fy, fw, fh)

        if best_bbox is not None:
            x, y, w, h = best_bbox
            return image[y:y+h, x:x+w], best_bbox

        # Fallback: Central license plate mount geometry
        cw = int(img_w * 0.38)
        ch = int(cw / self.target_aspect_ratio)
        fx = (img_w - cw) // 2
        fy = int(img_h * 0.62)
        return image[fy:fy+ch, fx:fx+cw], (fx, fy, cw, ch)
