"""
Arrow-backed datasets for memory-efficient training.

These are drop-in replacements for VisionCompressionDataset and TextBaselineDataset
that use HuggingFace Datasets (Arrow format) for memory-mapped loading.

Memory usage: ~50-100MB regardless of dataset size (vs ~40-45GB for 500K samples with JSONL).

Usage:
    # Convert JSONL to Arrow first:
    uv run python data/scripts/convert_to_arrow.py \
        --input data/training/splits_510k/train.jsonl \
        --output data/training/splits_510k/train_arrow

    # Then use in training (auto-detected by train.py if path is a directory):
    python train.py --data_path data/training/splits_510k/train_arrow ...
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torchvision.io
from datasets import Dataset as HFDataset
from datasets import load_from_disk
from torch.utils.data import Dataset

from trainers.config import VISION_MODES
from trainers.utils.image import BasicImageTransform


logger = logging.getLogger(__name__)


class ArrowTextDataset(Dataset):
    """
    Memory-efficient text dataset backed by Arrow files.

    Drop-in replacement for TextBaselineDataset.
    Uses memory-mapped Arrow files, reducing memory from ~40GB to ~50MB for 500K samples.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_samples: Optional[int] = None,
        context_length: Optional[int] = None,
        hybrid_text_tokens: int = 0
    ):
        """
        Initialize Arrow-backed text dataset.

        Args:
            data_path: Path to Arrow dataset directory (created by convert_to_arrow.py)
            tokenizer: Tokenizer (kept for interface compatibility, not used directly)
            max_samples: Optional limit on number of samples to use
            context_length: Optional truncation length for context tokens
            hybrid_text_tokens: Number of tokens from context end to use as hybrid text
        """
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.hybrid_text_tokens = hybrid_text_tokens
        self.data_path = Path(data_path)

        # Load Arrow dataset (memory-mapped, NOT loaded into RAM)
        logger.info(f"Loading Arrow dataset from {data_path} (memory-mapped)")
        self.dataset: HFDataset = load_from_disk(str(data_path))

        # Apply max_samples limit if specified
        if max_samples is not None and max_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(max_samples))
            logger.info(f"Limited to {max_samples} samples")

        logger.info(f"Loaded {len(self.dataset):,} samples from {data_path} (memory-mapped)")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Returns dict with same structure as TextBaselineDataset:
            - context: torch.Tensor of context token IDs
            - continuation: torch.Tensor of continuation token IDs
            - hybrid_text: list of token IDs (empty if hybrid_text_tokens=0)
            - sample_id: string identifier
        """
        # This loads ONLY the requested row from disk via memory mapping
        sample = self.dataset[idx]

        context_tokens = list(sample['context_tokens'])

        # Apply context truncation if specified (same logic as TextBaselineDataset)
        if self.context_length is not None:
            if self.context_length < 0:
                raise ValueError("context_length must be non-negative")
            if self.context_length == 0:
                context_tokens = []
            elif len(context_tokens) > self.context_length:
                context_tokens = context_tokens[-self.context_length:]

        # Extract hybrid text if enabled (last K tokens from context)
        hybrid_text = []
        if self.hybrid_text_tokens > 0:
            if self.hybrid_text_tokens > len(context_tokens):
                raise ValueError(
                    f"hybrid_text_tokens ({self.hybrid_text_tokens}) exceeds context length "
                    f"({len(context_tokens)}) for sample {idx}"
                )
            hybrid_text = context_tokens[-self.hybrid_text_tokens:]

        # Convert to tensors (same as TextBaselineDataset)
        return {
            'context': torch.tensor(context_tokens, dtype=torch.long),
            'continuation': torch.tensor(list(sample['continuation_tokens']), dtype=torch.long),
            'hybrid_text': hybrid_text,
            'sample_id': sample.get('id', f'sample_{idx}')
        }


class ArrowVisionDataset(Dataset):
    """
    Memory-efficient vision dataset backed by Arrow files.

    Drop-in replacement for VisionCompressionDataset.
    Arrow provides memory-efficient metadata/token storage.
    Images are still loaded from disk per-sample (same as original).
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        vision_mode: str = 'small',
        transform=None,
        max_samples: Optional[int] = None,
        hybrid_text_tokens: int = 0
    ):
        """
        Initialize Arrow-backed vision dataset.

        Args:
            data_path: Path to Arrow dataset directory (created by convert_to_arrow.py)
            tokenizer: Tokenizer (kept for interface compatibility)
            vision_mode: One of 'tiny', 'small', 'base', 'large'
            transform: Image transform (defaults to BasicImageTransform)
            max_samples: Optional limit on number of samples to use
            hybrid_text_tokens: Number of tokens from context end to use as hybrid text
        """
        self.tokenizer = tokenizer
        self.vision_mode = vision_mode
        self.mode_config = VISION_MODES[vision_mode]
        self.transform = transform or BasicImageTransform()
        self.hybrid_text_tokens = hybrid_text_tokens
        self.data_path = Path(data_path)

        # Determine image directory from data_path
        # Arrow path is like "data/training/splits_510k/train_arrow"
        # Images are in "data/training/splits_510k/images/{mode}/"
        self.data_dir = self.data_path.parent
        self.image_dir = self.data_dir / 'images' / vision_mode

        # Load Arrow dataset (memory-mapped)
        logger.info(f"Loading Arrow dataset from {data_path} (memory-mapped)")
        self.dataset: HFDataset = load_from_disk(str(data_path))

        if max_samples is not None and max_samples < len(self.dataset):
            self.dataset = self.dataset.select(range(max_samples))
            logger.info(f"Limited to {max_samples} samples")

        logger.info(f"Loaded {len(self.dataset):,} samples from {data_path} (memory-mapped)")
        logger.info(f"Vision mode: {vision_mode} ({self.mode_config['tokens']} tokens, "
                   f"{self.mode_config['image_size']}x{self.mode_config['image_size']})")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.

        Returns dict with same structure as VisionCompressionDataset:
            - image: torch.Tensor of shape (3, H, W) in uint8
            - continuation: torch.Tensor of continuation token IDs
            - context: torch.Tensor of context token IDs
            - hybrid_text: list of token IDs (empty if hybrid_text_tokens=0)
            - sample_id: string identifier
            - image_path: string path to image file
        """
        # Load metadata from Arrow (memory-mapped)
        sample = self.dataset[idx]
        sample_id = sample['id']

        # Load image from disk (same as VisionCompressionDataset)
        image_path = self.image_dir / f"{sample_id}.png"

        if not image_path.exists():
            raise FileNotFoundError(f"Pre-rendered image not found: {image_path}")

        # Load image using torchvision.io (1.33x faster than PIL)
        image_tensor = torchvision.io.decode_image(
            str(image_path),
            mode=torchvision.io.ImageReadMode.RGB
        )  # uint8 [C, H, W]

        # Pad to base_size (should already be correct size, but ensures consistency)
        base_size = self.mode_config['base_size']
        _, h, w = image_tensor.shape
        if h != base_size or w != base_size:
            pad_value = int(self.transform.mean[0] * 255)
            padded = torch.full((3, base_size, base_size), pad_value, dtype=torch.uint8)
            y_offset = (base_size - h) // 2
            x_offset = (base_size - w) // 2
            padded[:, y_offset:y_offset+h, x_offset:x_offset+w] = image_tensor
            image_tensor = padded

        # Return uint8 tensor - normalization happens on GPU in trainer (32.9x faster)
        # image_tensor is already uint8 [C, H, W]

        # Get tokens from Arrow
        continuation_tokens = list(sample['continuation_tokens'])
        context_tokens = list(sample['context_tokens'])

        # Get hybrid text tokens if enabled
        hybrid_text = []
        if self.hybrid_text_tokens > 0:
            hybrid_text = context_tokens[-self.hybrid_text_tokens:]

        return {
            'image': image_tensor,
            'continuation': torch.tensor(continuation_tokens, dtype=torch.long),
            'context': torch.tensor(context_tokens, dtype=torch.long),
            'hybrid_text': hybrid_text,
            'sample_id': sample_id,
            'image_path': str(image_path)
        }
