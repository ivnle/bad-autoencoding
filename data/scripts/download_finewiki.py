#!/usr/bin/env python3
"""
Download FineWiki English subset using huggingface_hub.

This script downloads only the English Wikipedia data (~37.5 GB, 15 parquet files)
from the HuggingFaceFW/finewiki dataset.
"""

from huggingface_hub import snapshot_download
import os

def main():
    print("Starting FineWiki English subset download...")
    print("Expected size: ~37.5 GB (15 parquet files)")
    print("Download location: ./data/finewiki/data/enwiki/")
    print()

    folder = snapshot_download(
        "HuggingFaceFW/finewiki",
        repo_type="dataset",
        local_dir="./data/finewiki/",
        allow_patterns=["data/enwiki/*"]  # English only
    )

    print()
    print(f"Download complete! Data stored in: {folder}")
    print()
    print("Verifying download...")

    # Check if the directory exists and count files
    enwiki_path = os.path.join(folder, "data", "enwiki")
    if os.path.exists(enwiki_path):
        files = [f for f in os.listdir(enwiki_path) if f.endswith('.parquet')]
        print(f"✅ Found {len(files)} parquet files in {enwiki_path}")

        # Calculate total size
        total_size = sum(os.path.getsize(os.path.join(enwiki_path, f)) for f in files)
        total_size_gb = total_size / (1024**3)
        print(f"✅ Total size: {total_size_gb:.2f} GB")
    else:
        print(f"❌ Directory not found: {enwiki_path}")

    print()
    print("Next step:")
    print("  uv run python data/scripts/filter_finewiki.py")

if __name__ == "__main__":
    main()
