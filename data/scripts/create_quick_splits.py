#!/usr/bin/env python3
"""
Quick Train/Val Split Generator

Generates reproducible train/validation splits from a subset of data that's still being generated.
Designed for fast iteration without waiting for full dataset completion.

Features:
- Early-exit loading (reads only first N samples)
- Reproducible shuffling with fixed seed
- Configurable validation size
- Manifest file for traceability
- No image copying (references existing image paths)

Usage:
    uv run python scripts/create_quick_splits.py \
        --input data/training/full/samples.jsonl \
        --output_dir data/training/splits_110k \
        --max_samples 110000 \
        --val_size 10000 \
        --seed 42
"""

import argparse
import json
import random
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm


def load_samples(
    input_path: Path,
    max_samples: int,
    desc: str = "Loading samples"
) -> List[Dict[str, Any]]:
    """
    Load first N samples from JSONL with early-exit for efficiency.

    Args:
        input_path: Path to input JSONL file
        max_samples: Maximum number of samples to load
        desc: Description for progress bar

    Returns:
        List of sample dictionaries

    Memory usage: ~1.65 GB for 110K samples (verified)
    """
    samples = []

    with open(input_path, 'r') as f:
        with tqdm(desc=desc, unit=' samples', total=max_samples) as pbar:
            for idx, line in enumerate(f):
                samples.append(json.loads(line))
                pbar.update(1)

                # Early exit to avoid reading entire file
                if idx + 1 >= max_samples:
                    break

    return samples


def write_jsonl(samples: List[Dict[str, Any]], output_path: Path, desc: str = "Writing"):
    """Write samples to JSONL file with progress bar."""
    with open(output_path, 'w') as f:
        for sample in tqdm(samples, desc=desc, unit=' samples'):
            f.write(json.dumps(sample) + '\n')


def compute_manifest_hash(input_path: Path, max_samples: int, seed: int) -> str:
    """Compute hash for manifest to detect if splits need regeneration."""
    hash_input = f"{input_path}:{max_samples}:{seed}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16]


def create_manifest(
    output_dir: Path,
    input_path: Path,
    max_samples: int,
    val_size: int,
    train_size: int,
    seed: int
):
    """
    Create manifest file with metadata for traceability.

    Manifest allows you to know exactly what went into each split,
    which is critical for reproducibility and debugging.
    """
    manifest = {
        'created_at': datetime.now().isoformat(),
        'source_file': str(input_path),
        'max_samples': max_samples,
        'validation_size': val_size,
        'training_size': train_size,
        'total_size': val_size + train_size,
        'seed': seed,
        'config_hash': compute_manifest_hash(input_path, max_samples, seed),
        'split_ratio': f"{val_size}/{train_size} (val/train)",
        'val_percentage': round(100 * val_size / (val_size + train_size), 2),
    }

    manifest_path = output_dir / 'split_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Manifest written to: {manifest_path}")
    print(f"  Config hash: {manifest['config_hash']}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate reproducible train/val splits from data subset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard usage (110K total, 10K val, 100K train)
  uv run python scripts/create_quick_splits.py \\
      --input data/training/full/samples.jsonl \\
      --output_dir data/training/splits_110k \\
      --max_samples 110000 \\
      --val_size 10000 \\
      --seed 42

  # Smaller test split (1K total, 100 val, 900 train)
  uv run python scripts/create_quick_splits.py \\
      --input data/training/full/samples.jsonl \\
      --output_dir data/training/splits_1k \\
      --max_samples 1000 \\
      --val_size 100 \\
      --seed 42

Note: Images are NOT copied. The split JSONL files reference existing image paths.
        """
    )

    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Path to input JSONL file (e.g., data/training/full/samples.jsonl)'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        required=True,
        help='Output directory for train.jsonl and val.jsonl'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=110000,
        help='Maximum samples to load from input file (default: 110000)'
    )
    parser.add_argument(
        '--val_size',
        type=int,
        default=10000,
        help='Number of samples for validation split (default: 10000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducible shuffling (default: 42)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        return 1

    if args.val_size >= args.max_samples:
        print(f"Error: val_size ({args.val_size}) must be less than max_samples ({args.max_samples})")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Quick Train/Val Split Generator")
    print("=" * 70)
    print(f"Input:       {args.input}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Max samples: {args.max_samples:,}")
    print(f"Val size:    {args.val_size:,}")
    print(f"Train size:  {args.max_samples - args.val_size:,}")
    print(f"Seed:        {args.seed}")
    print(f"Val ratio:   {100 * args.val_size / args.max_samples:.1f}%")
    print("=" * 70)
    print()

    # Load samples (early-exit reading)
    print(f"Loading first {args.max_samples:,} samples...")
    samples = load_samples(args.input, args.max_samples)

    if len(samples) < args.max_samples:
        print(f"\nWarning: File only contains {len(samples):,} samples (requested {args.max_samples:,})")
        print(f"         Adjusting val_size proportionally...")
        # Adjust val_size proportionally
        actual_val_size = int(args.val_size * len(samples) / args.max_samples)
    else:
        actual_val_size = args.val_size

    # Shuffle with fixed seed for reproducibility
    print(f"\nShuffling with seed={args.seed}...")
    random.seed(args.seed)
    random.shuffle(samples)

    # Split: first N samples → validation, rest → training
    print(f"\nSplitting into validation ({actual_val_size:,}) and training ({len(samples) - actual_val_size:,})...")
    val_samples = samples[:actual_val_size]
    train_samples = samples[actual_val_size:]

    # Write splits
    val_path = args.output_dir / 'val.jsonl'
    train_path = args.output_dir / 'train.jsonl'

    write_jsonl(val_samples, val_path, desc="Writing val.jsonl")
    write_jsonl(train_samples, train_path, desc="Writing train.jsonl")

    # Create symlink to images directory for vision regime compatibility
    images_symlink = args.output_dir / 'images'
    images_target = Path('../full/images')

    if images_symlink.is_symlink():
        # It's a symlink - check if it's valid or broken
        if images_symlink.exists():
            # Valid symlink (target exists)
            print(f"\n✓ Images symlink already exists: {images_symlink}")
        else:
            # Broken symlink (target missing) - recreate it
            print(f"\n⚠ Broken symlink detected, recreating...")
            images_symlink.unlink()
            images_symlink.symlink_to(images_target)
            print(f"✓ Recreated symlink: {images_symlink} -> {images_target}")
    elif images_symlink.is_dir():
        # Images directory already exists (e.g., when outputting to same dir as prepare_training_data.py)
        print(f"\n✓ Images directory already exists: {images_symlink}")
    elif images_symlink.exists():
        # Something else exists with that name
        print(f"\n⚠ Warning: {images_symlink} exists but is not a directory or symlink, skipping...")
    else:
        # Nothing exists - create new symlink
        print(f"\n✓ Creating symlink: {images_symlink} -> {images_target}")
        images_symlink.symlink_to(images_target)

    # Create manifest for traceability
    create_manifest(
        args.output_dir,
        args.input,
        args.max_samples,
        len(val_samples),
        len(train_samples),
        args.seed
    )

    # Summary
    print("\n" + "=" * 70)
    print("✓ Split generation complete!")
    print("=" * 70)
    print(f"Validation: {val_path} ({len(val_samples):,} samples)")
    print(f"Training:   {train_path} ({len(train_samples):,} samples)")
    print(f"\nNote: Images are referenced from existing paths (no copying performed)")
    print("=" * 70)

    return 0


if __name__ == '__main__':
    exit(main())
