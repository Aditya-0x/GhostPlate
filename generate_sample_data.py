"""
Sample Data and Adverse Weather Benchmark Generator
Renders synthetic realistic license plates on vehicle bumper backdrops and applies
physical Fog, Rain, and Headlight Glare degradations.
"""

import os
import json
import random
import cv2
import numpy as np

from src.utils.weather_effects import (
    add_fog,
    add_rain,
    add_headlight_glare,
    apply_combined_adverse_weather
)


def render_synthetic_vehicle_plate(
    plate_text: str = "KA05MC2024",
    img_size: tuple[int, int] = (400, 600)
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """
    Renders a realistic vehicle rear/front view with a mounted license plate.

    Args:
        plate_text: Alphanumeric plate string (e.g. 'DL04AB1234').
        img_size: (H, W) canvas dimension.

    Returns:
        tuple of (BGR vehicle image, (x, y, w, h) plate bbox).
    """
    h, w = img_size
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. Vehicle body gradient (dark metallic gray / navy)
    for y in range(h):
        shade = int(35 + 25 * (y / h))
        img[y, :] = [shade, shade + 5, shade + 10]

    # Bumper curvature / horizontal styling lines
    bumper_y = int(h * 0.5)
    cv2.line(img, (0, bumper_y), (w, bumper_y), (55, 60, 65), 3)
    cv2.line(img, (0, bumper_y + 120), (w, bumper_y + 120), (25, 25, 25), 4)

    # Headlights / Taillights
    light_y = int(h * 0.42)
    left_light_x = int(w * 0.12)
    right_light_x = int(w * 0.88)
    cv2.ellipse(img, (left_light_x, light_y), (45, 20), 0, 0, 360, (220, 240, 255), -1)
    cv2.ellipse(img, (right_light_x, light_y), (45, 20), 0, 0, 360, (220, 240, 255), -1)

    # 2. License Plate Mount Area
    plate_w = int(w * 0.38)
    plate_h = int(plate_w * 0.28)
    px = (w - plate_w) // 2
    py = int(h * 0.62)

    # Black mounting recess
    cv2.rectangle(img, (px - 8, py - 6), (px + plate_w + 8, py + plate_h + 6), (15, 15, 15), -1)

    # White / Yellow Plate Canvas
    plate_canvas = np.ones((plate_h, plate_w, 3), dtype=np.uint8) * 245
    # Thin black border
    cv2.rectangle(plate_canvas, (2, 2), (plate_w - 3, plate_h - 3), (10, 10, 10), 2)

    # Blue IND badge strip on the left (standard HSRP license plate design)
    badge_w = int(plate_w * 0.08)
    plate_canvas[:, :badge_w] = [180, 50, 20]  # Blue strip BGR

    # Draw Text Characters
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.68
    thickness = 2
    text_w = plate_w - badge_w - 18
    char_step = text_w / len(plate_text)

    for idx, ch in enumerate(plate_text):
        sz, _ = cv2.getTextSize(ch, font, scale, thickness)
        cx = int(badge_w + 10 + idx * char_step + (char_step - sz[0]) / 2)
        cy = int(plate_h * 0.70)
        cv2.putText(plate_canvas, ch, (cx, cy), font, scale, (15, 15, 15), thickness, cv2.LINE_AA)

    # Mounting screws
    cv2.circle(plate_canvas, (badge_w + 8, plate_h // 2), 3, (80, 80, 80), -1)
    cv2.circle(plate_canvas, (plate_w - 12, plate_h // 2), 3, (80, 80, 80), -1)

    # Paste plate into vehicle canvas
    img[py:py + plate_h, px:px + plate_w] = plate_canvas

    return img, (px, py, plate_w, plate_h)


def generate_benchmark_dataset(
    output_dir: str = "data/samples",
    num_samples: int = 6
):
    """
    Generates a benchmark suite covering:
      - clean
      - fog
      - rain
      - glare
      - combined (fog + rain + glare)
    """
    os.makedirs(output_dir, exist_ok=True)

    plate_strings = [
        "KA05MC2024",
        "DL01AB8899",
        "MH12DE4567",
        "HR26BR1122",
        "TN09AK7733",
        "UP32CV9090",
        "GJ01AZ5544",
        "WB02XY3311"
    ]

    manifest = []

    print(f"[*] Generating {num_samples} adverse weather test cases into '{output_dir}'...")

    for i in range(min(num_samples, len(plate_strings))):
        text = plate_strings[i]
        clean_img, bbox = render_synthetic_vehicle_plate(text)

        # Generate degradations
        fog_img = add_fog(clean_img, beta=1.3)
        rain_img = add_rain(clean_img, density=0.04, streak_length=22)
        glare_img = add_headlight_glare(clean_img, intensity=1.1)
        combined_img = apply_combined_adverse_weather(clean_img, fog_beta=1.0, rain_density=0.03, with_glare=True)

        sample_info = {
            "id": i + 1,
            "ground_truth": text,
            "bbox": bbox,
            "files": {}
        }

        modes = {
            "clean": clean_img,
            "fog": fog_img,
            "rain": rain_img,
            "glare": glare_img,
            "combined": combined_img
        }

        for mode, img in modes.items():
            fname = f"sample_{i+1:02d}_{mode}.jpg"
            fpath = os.path.join(output_dir, fname)
            cv2.imwrite(fpath, img)
            sample_info["files"][mode] = fname

        manifest.append(sample_info)

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[+] Benchmark dataset successfully generated at '{output_dir}'. Manifest: {manifest_path}")


if __name__ == "__main__":
    generate_benchmark_dataset()

