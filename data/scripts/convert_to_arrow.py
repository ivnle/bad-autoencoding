#!/usr/bin/env python3
"""
Convert JSONL training data to Arrow format for memory-efficient loading.

Arrow files are memory-mapped, reducing per-experiment memory from ~40-45GB
to ~100MB for the 510k dataset while preserving deterministic ordering.

Usage:
    uv run python data/scripts/convert_to_arrow.py \
        --input data/training/splits_510k/train.jsonl \
        --output data/training/splits_510k/train_arrow

    # Convert both train and val:
    for split in train val; do
        uv run python data/scripts/convert_to_arrow.py \
            --input data/training/splits_510k/${split}.jsonl \
            --output data/training/splits_510k/${split}_arrow
    done
"""

import argparse
import json
import logging
from pathlib import Path

from datasets import Dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def count_lines(filepath: Path) -> int:
    """Count lines in a file efficiently."""
    count = 0
    with open(filepath, 'rb') as f:
        for _ in f:
            count += 1
    return count


def jsonl_generator(input_path: Path, total: int = None):
    """
    Generator that yields samples from JSONL file.

    Uses streaming to avoid loading entire file into memory during conversion.
    """
    with open(input_path, 'r') as f:
        for line in tqdm(f, desc="Reading JSONL", total=total, unit=" samples"):
            yield json.loads(line)


def convert_jsonl_to_arrow(input_path: Path, output_path: Path):
    """
    Convert JSONL file to Arrow format.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output Arrow directory
    """
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")

    # Count lines for progress bar
    logger.info("Counting samples...")
    num_samples = count_lines(input_path)
    logger.info(f"Found {num_samples:,} samples")

    # Get input file size for reference
    input_size_gb = input_path.stat().st_size / (1024**3)
    logger.info(f"Input file size: {input_size_gb:.2f} GB")

    # Handle empty input file
    if num_samples == 0:
        logger.warning("Input file is empty - creating empty Arrow dataset")
        # Create empty dataset with minimal schema
        dataset = Dataset.from_dict({
            'id': [],
            'context_tokens': [],
            'continuation_tokens': []
        })
        dataset.save_to_disk(str(output_path))
        logger.info(f"Created empty Arrow dataset at {output_path}")
        return

    # Create dataset from generator (streams, doesn't load all into memory)
    logger.info("Converting to Arrow format...")
    dataset = Dataset.from_generator(
        lambda: jsonl_generator(input_path, total=num_samples)
    )

    logger.info(f"Dataset created with {len(dataset):,} samples")
    logger.info(f"Features: {list(dataset.features.keys())}")

    # Save to disk (Arrow format)
    logger.info(f"Saving to {output_path}...")
    dataset.save_to_disk(str(output_path))

    # Report output size
    output_size = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
    output_size_gb = output_size / (1024**3)
    if input_size_gb > 0:
        logger.info(f"Output size: {output_size_gb:.2f} GB ({output_size_gb/input_size_gb:.1f}x input)")
    else:
        logger.info(f"Output size: {output_size_gb:.2f} GB")

    # Verify by loading and checking first/last samples
    logger.info("Verifying conversion...")
    loaded = Dataset.load_from_disk(str(output_path))

    # Check first sample
    with open(input_path, 'r') as f:
        first_jsonl = json.loads(f.readline())
    first_arrow = loaded[0]

    assert first_jsonl['id'] == first_arrow['id'], \
        f"First sample ID mismatch: {first_jsonl['id']} vs {first_arrow['id']}"

    # Check sample count
    assert len(loaded) == num_samples, \
        f"Sample count mismatch: {len(loaded)} vs {num_samples}"

    logger.info(f"Verification passed: {len(loaded):,} samples, IDs match")
    logger.info("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL training data to Arrow format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        required=True,
        help='Input JSONL file path'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        required=True,
        help='Output Arrow directory path'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite output directory if it exists'
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")

    if not args.input.suffix == '.jsonl':
        logger.warning(f"Input file does not have .jsonl extension: {args.input}")

    # Check output
    if args.output.exists():
        if args.overwrite:
            logger.warning(f"Overwriting existing output: {args.output}")
            import shutil
            shutil.rmtree(args.output)
        else:
            parser.error(f"Output directory already exists: {args.output} (use --overwrite to replace)")

    convert_jsonl_to_arrow(args.input, args.output)


if __name__ == '__main__':
    main()
