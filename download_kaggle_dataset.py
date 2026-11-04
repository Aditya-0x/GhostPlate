"""
Kaggle Adverse Weather Vehicle Plates Dataset Downloader & Preparer
Handles automated downloading, extraction, and validation of the Kaggle dataset.
"""

import os
import sys
import zipfile
import argparse


def check_kaggle_auth() -> bool:
    """Checks if Kaggle API credentials exist in default locations or environment."""
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    return os.path.exists(kaggle_json) or bool(has_env)


def download_dataset(
    dataset_name: str = "aditya2803/adverse-weather-vehicle-plates",
    output_dir: str = "data/kaggle_plates"
):
    """
    Downloads and extracts the dataset using the Kaggle Python API.
    """
    os.makedirs(output_dir, exist_ok=True)

    if not check_kaggle_auth():
        print("\n" + "!" * 70)
        print("  KAGGLE API TOKEN NOT DETECTED")
        print("!" * 70)
        print("To download directly from Kaggle CLI/API:")
        print("1. Go to https://www.kaggle.com/settings -> Account -> 'Create New Token'.")
        print("2. Save the downloaded 'kaggle.json' to:")
        print(f"   {os.path.expanduser('~/.kaggle/kaggle.json')}")
        print("\nAlternatively, you can manually download the dataset zip from Kaggle,")
        print(f"place it into '{output_dir}', and run this script again with --extract.")
        print("!" * 70 + "\n")
        return False

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"[*] Authenticated with Kaggle. Downloading '{dataset_name}' to '{output_dir}'...")
        api.dataset_download_files(dataset_name, path=output_dir, unzip=True)
        print(f"[+] Dataset successfully downloaded and extracted to '{output_dir}'.")
        return True
    except Exception as e:
        print(f"[!] Failed to download via Kaggle API: {e}")
        print("Please verify the exact dataset slug or download manually into the directory.")
        return False


def extract_existing_zip(zip_path: str, extract_dir: str = "data/kaggle_plates"):
    """Extracts a manually downloaded Kaggle ZIP archive."""
    if not os.path.exists(zip_path):
        print(f"[!] Zip file '{zip_path}' not found.")
        return False
    print(f"[*] Extracting '{zip_path}' into '{extract_dir}'...")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"[+] Extraction complete.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kaggle Adverse Weather Plate Dataset Downloader")
    parser.add_argument("--dataset", type=str, default="aditya2803/adverse-weather-vehicle-plates", help="Kaggle dataset slug (owner/dataset-name)")
    parser.add_argument("--output_dir", type=str, default="data/kaggle_plates", help="Output directory")
    parser.add_argument("--zip_path", type=str, default=None, help="Path to local zip file to extract")
    args = parser.parse_args()

    if args.zip_path:
        extract_existing_zip(args.zip_path, args.output_dir)
    else:
        download_dataset(args.dataset, args.output_dir)

