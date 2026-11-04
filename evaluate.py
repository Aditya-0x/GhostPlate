"""
Ablation and Benchmark Evaluation Script
Evaluates ALPR accuracy and image restoration metrics (PSNR, SSIM, CER, Exact Match)
across both End-to-End detection and Isolated Plate Crop recognition.
"""

import os
import json
import argparse
import cv2
import numpy as np
import torch

from src.pipeline import AdverseWeatherALPRPipeline


def compute_levenshtein(s1: str, s2: str) -> int:
    """Computes minimum edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Peak Signal-to-Noise Ratio (PSNR) in dB."""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Computes Structural Similarity Index (SSIM) on luminance channel."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    img1_f = gray1.astype(np.float64)
    img2_f = gray2.astype(np.float64)

    mu1 = cv2.GaussianBlur(img1_f, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2_f, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(img1_f * img1_f, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2_f * img2_f, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1_f * img2_f, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(np.mean(ssim_map))


def evaluate_benchmark(
    data_dir: str = "data/samples",
    weights_path: str = "models/char_cnn_lightweight.pth",
    device: str | None = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest_path = os.path.join(data_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[!] Manifest not found at {manifest_path}. Run generate_sample_data.py first.")
        return

    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    pipeline = AdverseWeatherALPRPipeline(weights_path=weights_path, device=device)

    conditions = ["clean", "fog", "rain", "glare", "combined"]

    print("\n" + "=" * 80)
    print("  AUTOMATED LICENSE PLATE CHARACTER RECOGNITION BENCHMARK & ABLATION")
    print("  Group 6: Aditya Verma, Varun Singh Rajpurohit, Appan")
    print("=" * 80)

    results_table = []

    for cond in conditions:
        for mode_name, use_enhancement in [("Without Pre-filter (Raw)", False), ("With DCP + Glare Pre-filter", True)]:
            total_chars = 0
            total_edit_dist = 0
            exact_matches = 0
            psnr_list = []
            ssim_list = []

            for sample in samples:
                gt = sample["ground_truth"]
                clean_path = os.path.join(data_dir, sample["files"]["clean"])
                clean_img = cv2.imread(clean_path)

                test_path = os.path.join(data_dir, sample["files"][cond])
                test_img = cv2.imread(test_path)

                # Process
                res = pipeline.process(test_img, with_enhancement=use_enhancement)
                pred_text = res["plate_text"]

                # Image metrics (against ground truth clean image)
                work_img = res["work_image"]
                psnr_val = compute_psnr(clean_img, work_img)
                ssim_val = compute_ssim(clean_img, work_img)
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)

                # Text metrics
                edit_dist = compute_levenshtein(pred_text, gt)
                total_edit_dist += edit_dist
                total_chars += len(gt)
                if pred_text == gt:
                    exact_matches += 1

            cer = (total_edit_dist / max(1, total_chars)) * 100.0
            acc = (exact_matches / len(samples)) * 100.0
            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)

            results_table.append({
                "condition": cond.capitalize(),
                "mode": mode_name,
                "cer": cer,
                "exact_acc": acc,
                "psnr": avg_psnr,
                "ssim": avg_ssim
            })

    # Print Table
    header = f"{'Condition':<12} | {'Pipeline Mode':<30} | {'CER (%)':<8} | {'Exact Acc (%)':<14} | {'PSNR (dB)':<10} | {'SSIM':<6}"
    print(header)
    print("-" * len(header))

    for row in results_table:
        print(f"{row['condition']:<12} | {row['mode']:<30} | {row['cer']:<8.2f} | {row['exact_acc']:<14.1f} | {row['psnr']:<10.2f} | {row['ssim']:<6.4f}")

    print("=" * 80)
    print("[+] Evaluation completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ALPR pipeline under adverse weather")
    parser.add_argument("--data_dir", type=str, default="data/samples")
    parser.add_argument("--weights_path", type=str, default="models/char_cnn_lightweight.pth")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    evaluate_benchmark(data_dir=args.data_dir, weights_path=args.weights_path, device=args.device)
