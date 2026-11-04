"""
Interactive Demo Script for Adverse Weather ALPR
Group 6: Aditya Verma, Varun Singh Rajpurohit, Appan
"""

import os
import argparse
import cv2

from src.pipeline import AdverseWeatherALPRPipeline
from src.utils.visualization import plot_pipeline_comparison


def run_demo(
    image_path: str = "data/samples/sample_01_combined.jpg",
    weights_path: str = "models/char_cnn_lightweight.pth",
    ground_truth: str = "KA05MC2024",
    output_plot: str = "outputs/output_demo.png",
    device: str | None = None
):
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(output_plot) or ".", exist_ok=True)
    if not os.path.exists(image_path):
        print(f"[!] Target image '{image_path}' not found. Generating sample data first...")
        from generate_sample_data import generate_benchmark_dataset
        generate_benchmark_dataset()

    print(f"[*] Loading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at {image_path}")

    print(f"[*] Initializing AdverseWeatherALPRPipeline on {device}...")
    pipeline = AdverseWeatherALPRPipeline(
        weights_path=weights_path if os.path.exists(weights_path) else None,
        device=device,
        enable_enhancement=True
    )

    print("[*] Processing image through De-hazing, Glare Attenuation, and ALPR...")
    res = pipeline.process(image)

    plate_text = res["plate_text"]
    char_preds = res["char_preds"]
    bbox = res["plate_bbox"]
    crop = res["plate_crop"]
    enh_data = res["enhancement_data"]

    print("\n" + "=" * 50)
    print("           ALPR RECOGNITION SUMMARY")
    print("=" * 50)
    print(f" Ground Truth:          {ground_truth}")
    print(f" Recognized Plate:      {plate_text}")
    print(f" Detected BBox (x,y,w,h): {bbox}")
    print("-" * 50)
    print(" Character Breakdown:")
    for idx, (ch, conf) in enumerate(char_preds, 1):
        print(f"   [{idx:02d}] Character '{ch}' -> Confidence: {conf * 100:.2f}%")
    print("=" * 50)

    # Plot pipeline comparison
    dark_ch = enh_data.get("dark_channel") if enh_data else None
    t_map = enh_data.get("transmission_map") if enh_data else None
    enhanced = res["work_image"]

    plot_pipeline_comparison(
        raw_img=image,
        enhanced_img=enhanced,
        dark_channel=dark_ch,
        transmission_map=t_map,
        plate_bbox=bbox,
        plate_crop=crop,
        char_predictions=char_preds,
        ground_truth=ground_truth,
        save_path=output_plot
    )
    print(f"[+] Visualization saved to '{output_plot}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Adverse Weather ALPR Demo")
    parser.add_argument("--image_path", type=str, default="data/samples/sample_01_combined.jpg")
    parser.add_argument("--weights_path", type=str, default="models/char_cnn_lightweight.pth")
    parser.add_argument("--ground_truth", type=str, default="KA05MC2024")
    parser.add_argument("--output_plot", type=str, default="outputs/output_demo.png")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    run_demo(
        image_path=args.image_path,
        weights_path=args.weights_path,
        ground_truth=args.ground_truth,
        output_plot=args.output_plot,
        device=args.device
    )

