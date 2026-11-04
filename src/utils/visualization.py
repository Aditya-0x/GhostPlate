"""
Visualization Utilities for Adverse Weather ALPR Pipeline
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_pipeline_comparison(
    raw_img: np.ndarray,
    enhanced_img: np.ndarray,
    dark_channel: np.ndarray | None,
    transmission_map: np.ndarray | None,
    plate_bbox: tuple[int, int, int, int] | None,
    plate_crop: np.ndarray | None,
    char_predictions: list[tuple[str, float]],
    ground_truth: str = "",
    save_path: str | None = None
):
    """
    Plots a multi-panel visual breakdown of the ALPR restoration and recognition pipeline.
    """
    fig = plt.figure(figsize=(15, 10))

    # 1. Degraded input
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
    ax1.set_title("1. Adverse Weather Input (Fog/Rain/Glare)")
    ax1.axis("off")

    # 2. Dark Channel Prior
    ax2 = fig.add_subplot(2, 3, 2)
    if dark_channel is not None:
        ax2.imshow(dark_channel, cmap="gray")
        ax2.set_title("2. Dark Channel Prior $J^{dark}$")
    else:
        ax2.text(0.5, 0.5, "N/A", ha="center")
    ax2.axis("off")

    # 3. Transmission Map
    ax3 = fig.add_subplot(2, 3, 3)
    if transmission_map is not None:
        im3 = ax3.imshow(transmission_map, cmap="jet")
        ax3.set_title("3. Guided-Filtered Transmission $t(x)$")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    else:
        ax3.text(0.5, 0.5, "N/A", ha="center")
    ax3.axis("off")

    # 4. Enhanced / Restored Image with Plate Detection Box
    ax4 = fig.add_subplot(2, 3, 4)
    disp_enh = enhanced_img.copy()
    if plate_bbox is not None:
        x, y, w, h = plate_bbox
        cv2.rectangle(disp_enh, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(disp_enh, "Plate", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    ax4.imshow(cv2.cvtColor(disp_enh, cv2.COLOR_BGR2RGB))
    ax4.set_title("4. Dehazed & Glare-Attenuated + Detection")
    ax4.axis("off")

    # 5. Cropped License Plate
    ax5 = fig.add_subplot(2, 3, 5)
    if plate_crop is not None:
        ax5.imshow(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB))
        ax5.set_title("5. Cropped License Plate ROI")
    else:
        ax5.text(0.5, 0.5, "No Plate Detected", ha="center")
    ax5.axis("off")

    # 6. Character Recognition Results
    ax6 = fig.add_subplot(2, 3, 6)
    pred_str = "".join([c for c, _ in char_predictions])
    avg_conf = np.mean([conf for _, conf in char_predictions]) if char_predictions else 0.0

    ax6.text(0.1, 0.7, f"Ground Truth:  {ground_truth if ground_truth else 'N/A'}", fontsize=14, weight="bold")
    ax6.text(0.1, 0.45, f"Predicted:     {pred_str}", fontsize=16, color="green" if pred_str == ground_truth else "blue", weight="bold")
    ax6.text(0.1, 0.2, f"Avg Confidence: {avg_conf * 100:.1f}%", fontsize=12)
    ax6.set_title("6. Lightweight CNN Character OCR")
    ax6.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

