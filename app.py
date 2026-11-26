"""
Interactive Dark-Mode Gradio Demo UI — Adverse-Weather ALPR System
==================================================================
Project: Automated License Plate Character Recognition under Rain, Fog, and Glare
Team: Group 6 — Aditya Verma (12408804), Varun Singh Rajpurohit (12409154),
      Damandeep Saini (12403195)

Features:
- Upload any degraded vehicle/plate image or pick built-in weather-degraded samples
  (Dense Fog, Heavy Rain, Headlight Glare, Mixed Storm).
- Toggle the Dark-Channel Prior de-hazing & glare suppression pre-filter ON / OFF
  to observe its direct impact on character segmentation and CNN recognition.
- Inspect intermediate stages: De-hazed Image, Refined Transmission Map,
  Localized Plate with Character Bounding Boxes, and Normalized 32x32 Character Strip.
- Displays final predicted license plate string, mean confidence score, and latency (ms).

Run locally or on Google Colab:
    python app.py
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import cv2
import gradio as gr
import numpy as np

from src.dehaze import DehazeConfig, dehaze_and_enhance
from src.model import load_model, predict_character_batch
from src.segment import localize_and_segment
from src.train import (
    apply_weather_degradation,
    generate_clean_plate,
    train_character_cnn,
)

CHECKPOINT_PATH = "checkpoints/plate_cnn.pth"
SAMPLES_DIR = "sample_plates"


def ensure_model_and_samples() -> Tuple[object, object, List[Tuple[str, str]]]:
    """Ensure a trained checkpoint and sample degraded plates exist for instant demo use."""
    if not os.path.isfile(CHECKPOINT_PATH):
        print("[App] No checkpoint found — running fast 6-epoch bootstrap training...")
        train_character_cnn(
            epochs=6,
            batch_size=128,
            samples_per_class=90,
            checkpoint_path=CHECKPOINT_PATH,
        )

    model, device = load_model(CHECKPOINT_PATH)

    os.makedirs(SAMPLES_DIR, exist_ok=True)
    presets = [
        ("Dense Fog — DL01AB4829", "DL01AB4829", "fog", 0.78, 101),
        ("Heavy Rain — MH12DE7391", "MH12DE7391", "rain", 0.76, 202),
        ("Headlight Glare — KA03MG5042", "KA03MG5042", "glare", 0.80, 303),
        ("Mixed Storm — PB10CX9184", "PB10CX9184", "mixed", 0.74, 404),
    ]
    sample_entries: List[Tuple[str, str]] = []
    for label, plate_text, weather, severity, seed in presets:
        path = os.path.join(SAMPLES_DIR, f"{weather}_{plate_text}.png")
        if not os.path.isfile(path):
            rng = np.random.default_rng(seed)
            clean = generate_clean_plate(plate_text, embed_in_scene=False, rng=rng)
            degraded = apply_weather_degradation(clean, weather, severity=severity, rng=rng)
            cv2.imwrite(path, degraded)
        sample_entries.append((label, path))

    return model, device, sample_entries


MODEL, DEVICE, SAMPLE_ENTRIES = ensure_model_and_samples()


def build_character_strip(char_crops: List[np.ndarray]) -> np.ndarray:
    """Arrange normalized 32x32 segmented character crops into a clean horizontal strip."""
    if not char_crops:
        blank = np.full((72, 360, 3), 18, dtype=np.uint8)
        cv2.putText(blank, "No characters segmented", (44, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (148, 163, 184), 1)
        return blank

    tiles = []
    for crop in char_crops:
        up = cv2.resize(crop, (56, 56), interpolation=cv2.INTER_NEAREST)
        tile = cv2.cvtColor(up, cv2.COLOR_GRAY2RGB)
        tile = cv2.copyMakeBorder(tile, 6, 6, 6, 6, cv2.BORDER_CONSTANT, value=(15, 20, 29))
        cv2.rectangle(tile, (2, 2), (65, 65), (6, 182, 212), 1)
        tiles.append(tile)
    return np.hstack(tiles)


def run_alpr_inference(
    input_image_rgb: np.ndarray | None,
    enable_dcp: bool,
    omega: float,
    patch_size: int,
    enable_glare_stage: bool,
    inject_weather: str,
    weather_severity: float,
):
    """Execute end-to-end ALPR pipeline and return all visual stages and metrics."""
    if input_image_rgb is None:
        return None, None, None, None, "No image loaded", "0.0%", "0.0 ms"

    t_start = time.perf_counter()
    image_bgr = cv2.cvtColor(input_image_rgb, cv2.COLOR_RGB2BGR)

    # Optional synthetic weather stress-test injection
    if inject_weather != "None":
        image_bgr = apply_weather_degradation(
            image_bgr,
            weather_type=inject_weather.lower(),
            severity=float(weather_severity),
            rng=np.random.default_rng(42),
        )

    # Stage 1: Dark-Channel Prior De-hazing & Glare Mitigation
    if enable_dcp:
        cfg = DehazeConfig(
            patch_size=int(patch_size),
            omega=float(omega),
            enable_glare_suppression=bool(enable_glare_stage),
        )
        processed_bgr, maps = dehaze_and_enhance(image_bgr, config=cfg, return_intermediates=True)
        transmission_vis = cv2.applyColorMap(maps["refined_transmission"], cv2.COLORMAP_OCEAN)
        transmission_rgb = cv2.cvtColor(transmission_vis, cv2.COLOR_BGR2RGB)
    else:
        processed_bgr = image_bgr.copy()
        transmission_rgb = np.zeros_like(input_image_rgb)
        cv2.putText(
            transmission_rgb,
            "DCP Pre-Filter Disabled",
            (20, max(36, transmission_rgb.shape[0] // 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (148, 163, 184),
            2,
        )

    # Stage 2: Plate Localization & Character Segmentation
    seg = localize_and_segment(processed_bgr)
    char_crops = [c.crop_gray for c in seg.characters]

    # Stage 3: Lightweight PyTorch CNN Character Recognition
    plate_str, mean_conf, per_char = predict_character_batch(MODEL, char_crops, DEVICE)
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    dehazed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    annotated_rgb = cv2.cvtColor(seg.annotated_plate_bgr, cv2.COLOR_BGR2RGB)
    strip_rgb = build_character_strip(char_crops)

    breakdown = "  ".join(f"{ch}({conf * 100:.0f}%)" for ch, conf in per_char) if per_char else "No characters detected"
    predicted_display = f"{plate_str or 'UNREADABLE'}   |   {breakdown}"
    conf_display = f"{mean_conf * 100:.1f}%"
    latency_display = f"{elapsed_ms:.1f} ms"

    return (
        dehazed_rgb,
        transmission_rgb,
        annotated_rgb,
        strip_rgb,
        predicted_display,
        conf_display,
        latency_display,
    )


def build_demo() -> gr.Blocks:
    """Create a dark-mode Gradio Blocks UI."""
    theme = gr.themes.Base(
        primary_hue="cyan",
        neutral_hue="slate",
    ).set(
        body_background_fill="#080B10",
        body_text_color="#E6EDF5",
        block_background_fill="#0F141D",
        block_border_color="#222C3D",
        input_background_fill="#151C28",
    )

    with gr.Blocks(theme=theme, title="ClearPlate ALPR — Adverse Weather License Plate Recognition") as demo:
        gr.Markdown(
            """
            # Automated License Plate Character Recognition under Rain, Fog, and Glare
            **Group 6** — Aditya Verma (12408804) · Varun Singh Rajpurohit (12409154) · Damandeep Saini (12403195)
            """
        )

        with gr.Row():
            with gr.Column(scale=4):
                input_img = gr.Image(label="Input Vehicle / License Plate Image", type="numpy")
                enable_dcp = gr.Checkbox(value=True, label="Enable Dark-Channel Prior De-Hazing Pre-Filter")
                enable_glare = gr.Checkbox(value=True, label="Enable Glare & Rain Mitigation (Gamma + LAB CLAHE + Bilateral)")
                omega_slider = gr.Slider(0.50, 0.98, value=0.92, step=0.02, label="Haze Removal Strength (omega)")
                patch_slider = gr.Slider(5, 25, value=15, step=2, label="Dark Channel Patch Size (px)")

                with gr.Accordion("Synthetic Weather Stress-Test Overlay (Optional)", open=False):
                    inject_weather = gr.Radio(
                        choices=["None", "Fog", "Rain", "Glare", "Mixed"],
                        value="None",
                        label="Apply Synthetic Degradation to Input",
                    )
                    weather_severity = gr.Slider(0.20, 0.95, value=0.70, step=0.05, label="Degradation Severity")

                run_btn = gr.Button("Run ALPR Pipeline", variant="primary")

                gr.Examples(
                    examples=[[path] for _, path in SAMPLE_ENTRIES],
                    inputs=[input_img],
                    label="Built-in Adverse Weather Samples",
                )

            with gr.Column(scale=8):
                with gr.Row():
                    pred_box = gr.Textbox(label="Predicted License Plate & Per-Character Confidence", interactive=False)
                with gr.Row():
                    conf_box = gr.Textbox(label="Mean Confidence", interactive=False)
                    latency_box = gr.Textbox(label="End-to-End Latency", interactive=False)

                with gr.Row():
                    dehazed_out = gr.Image(label="Stage 1: De-Hazed & Glare-Suppressed Output")
                    trans_out = gr.Image(label="Stage 1b: Guided-Filter Refined Transmission Map")

                with gr.Row():
                    seg_out = gr.Image(label="Stage 2: Localized Plate & Bounding Boxes")
                strip_out = gr.Image(label="Stage 2b: Segmented 32x32 Character Inputs to PyTorch CNN")

        inputs = [input_img, enable_dcp, omega_slider, patch_slider, enable_glare, inject_weather, weather_severity]
        outputs = [dehazed_out, trans_out, seg_out, strip_out, pred_box, conf_box, latency_box]

        run_btn.click(fn=run_alpr_inference, inputs=inputs, outputs=outputs)
        enable_dcp.change(fn=run_alpr_inference, inputs=inputs, outputs=outputs)

    return demo


if __name__ == "__main__":
    app = build_demo()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
