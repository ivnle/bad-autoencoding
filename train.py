"""
Production Training Script for Vision Compression Experiment

Trains DeepSeek-OCR model with two regimes:
- Regime 1 (Vision Compression): Context as image, predict continuation
- Regime 2 (Text Baseline): Context as text (optionally truncated), predict continuation

Based on Phase 5.3 learnings with all fixes applied.
"""

import json
import logging
import math
import os
import random
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from PIL import Image, ImageDraw, ImageFont, ImageOps
import torchvision.io
from torchvision import transforms
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Import from trainers module
from trainers import (
    VisionCompressionTrainer,
    TextBaselineTrainer,
    MeanPoolCompressionTrainer,
    Conv1dResidualCompressionTrainer,
    BasicImageTransform,
    get_gpu_memory_mb,
    load_model_and_tokenizer,
    get_encoder_params,
    get_decoder_params,
    count_parameters
)

# Import training utilities
from training.validation import validate_conv1d_params
from training.arguments import create_argument_parser
from training.naming import generate_from_args
from training.checkpoints import (
    save_checkpoint,
    load_checkpoint,
    peek_checkpoint_metadata,
    validate_and_load_init_checkpoint,
    load_checkpoint_args,
    merge_args
)

# Import DeepSeek-OCR constants from trainers config
from trainers.config import (
    VISION_MODES,
    VISION_PROMPT_PRESETS,
    GUNDAM_PRESET,
    VISION_TOKEN_COUNT,
)

# Import text similarity metrics for qualitative evaluation
from training.metrics import cal_per_metrics

# Import Arrow-backed datasets for memory-efficient loading
from training.arrow_datasets import ArrowTextDataset, ArrowVisionDataset

# Module-level constant for BOS token ID (set after loading tokenizer)
BOS_TOKEN_ID = 0  # Default value, will be set from tokenizer

# Flag to track if effective context tokens have been logged
_effective_context_logged = False

# Track target tokens per sample for throughput calculation
_target_tokens_per_sample = 0

# Global reference to train sampler (for checkpoint saving)
_train_sampler = None

# Graceful shutdown flags (for SIGTERM/SIGINT handling)
_shutdown_requested = False
_shutdown_in_progress = False

# Enable TF32 for faster float32 operations
# Despite bf16 training, vendor code uses explicit float32 for numerical stability:
# - RMSNorm (120-200× per forward): explicit .to(torch.float32) for variance computation
# - Attention softmax (60-100× per forward): dtype=torch.float32 parameter
# - Final logits (1× per forward): .float() upcast before loss computation
# - MoE routing: float32 softmax for expert selection
# TF32 provides 2-5% speedup with zero accuracy impact (maintains float32 dynamic range)
torch.set_float32_matmul_precision('high')


# ============================================================================
# Graceful Shutdown Signal Handling
# ============================================================================

def _handle_shutdown_signal(signum, frame):
    """
    Handle SIGTERM/SIGINT for graceful shutdown with checkpoint save.

    When stop-gpu script sends SIGTERM, this sets a flag that the training
    loop checks. Training saves a checkpoint and exits cleanly.

    Second signal during shutdown = force exit (for stuck processes).
    """
    global _shutdown_requested, _shutdown_in_progress
    signal_name = 'SIGTERM' if signum == signal.SIGTERM else 'SIGINT'

    if _shutdown_in_progress:
        # Second signal during save = force exit
        print(f"\n[SHUTDOWN] Second {signal_name} received during checkpoint save, forcing exit!")
        sys.exit(1)

    print(f"\n[SHUTDOWN] Received {signal_name}, will save checkpoint and exit at next safe point...")
    _shutdown_requested = True


def _register_shutdown_handlers():
    """Register signal handlers for graceful shutdown."""
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    signal.signal(signal.SIGINT, _handle_shutdown_signal)


# ============================================================================
# Reproducibility
# ============================================================================

def seed_worker(worker_id):
    """
    Seed worker processes for reproducible data loading

    This function is called once per DataLoader worker process to ensure
    reproducible shuffling and any randomness in transforms/augmentations.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================================
# Stateful Random Sampler (for mid-epoch resume)
# ============================================================================

class StatefulRandomSampler:
    """
    Random sampler that saves permutation state for mid-epoch checkpoint resume.

    Unlike PyTorch's default RandomSampler, this preserves the shuffle order
    across checkpoint save/resume, preventing duplicate data exposure.

    IMPORTANT: This sampler generates a single permutation that is reused for
    all epochs. For multi-epoch training, you should implement a set_epoch()
    method that regenerates the permutation at each epoch boundary while
    preserving the ability to resume mid-epoch. For single-epoch training,
    this limitation does not affect correctness.

    Usage:
        sampler = StatefulRandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler)

        # Save state in checkpoint
        state = sampler.get_state()

        # Restore state on resume
        sampler = StatefulRandomSampler(dataset, permutation=state)
    """
    def __init__(self, data_source, permutation=None, skip_first=0):
        """
        Args:
            data_source: Dataset to sample from
            permutation: Optional pre-computed permutation (for resume)
            skip_first: Number of samples to skip on FIRST iteration only (for mid-epoch resume)
        """
        self.data_source = data_source
        self.num_samples = len(data_source)
        self.skip_first = skip_first
        self._first_iteration = True  # Flag: have we started iterating yet?

        if permutation is not None:
            # Resume: use provided permutation
            # Validate that permutation size matches current dataset size
            if len(permutation) != self.num_samples:
                raise ValueError(
                    f"Dataset size mismatch: checkpoint has {len(permutation)} samples, "
                    f"but current dataset has {self.num_samples} samples. "
                    f"Cannot resume with different dataset size."
                )

            # Validate permutation content (detect corrupted checkpoints)
            perm_set = set(permutation)
            expected_set = set(range(self.num_samples))
            if perm_set != expected_set:
                missing = sorted(expected_set - perm_set)
                extra = sorted(perm_set - expected_set)

                error_parts = [
                    f"Corrupted permutation in checkpoint: contains "
                    f"{len(perm_set)} unique values but expected {self.num_samples}."
                ]

                if missing:
                    missing_str = str(missing[:10]) + (f"... ({len(missing)} total)" if len(missing) > 10 else "")
                    error_parts.append(f"Missing indices: {missing_str}.")

                if extra:
                    extra_str = str(extra[:10]) + (f"... ({len(extra)} total)" if len(extra) > 10 else "")
                    error_parts.append(f"Unexpected indices: {extra_str}.")

                if len(permutation) != len(perm_set):
                    num_dupes = len(permutation) - len(perm_set)
                    error_parts.append(f"Permutation has {num_dupes} duplicate entries.")

                error_parts.append("Checkpoint may be corrupted.")
                raise ValueError(" ".join(error_parts))

            self.permutation = permutation
        else:
            # Fresh start: generate new permutation using global RNG
            # (global RNG is seeded via torch.manual_seed in main())
            self.permutation = torch.randperm(self.num_samples).tolist()

    def __iter__(self):
        # Apply skip only on very first iteration (mid-epoch resume)
        if self._first_iteration and self.skip_first > 0:
            self._first_iteration = False  # Clear flag immediately
            return iter(self.permutation[self.skip_first:])
        else:
            # All subsequent epochs: full iteration
            return iter(self.permutation)

    def __len__(self):
        return self.num_samples

    def get_state(self):
        """
        Get sampler state for checkpointing.

        Returns a copy to prevent accidental mutation of internal state.
        """
        return list(self.permutation)


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(output_dir: Path, rank: int = 0):
    """Setup logging to both file and console"""
    log_file = output_dir / "train.log"

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def compute_perplexity(loss: float) -> float:
    """
    Compute perplexity from cross-entropy loss

    Perplexity = exp(loss)

    Args:
        loss: Cross-entropy loss value

    Returns:
        Perplexity value. Returns inf if loss is too large (overflow protection).
    """
    try:
        return math.exp(loss)
    except OverflowError:
        # Loss too large, return infinity
        return float('inf')


# ============================================================================
# Text Rendering (for Regime 1)
# ============================================================================

def render_text_to_image(text: str, width: int = 1024, height: int = 1024) -> Image.Image:
    """
    Render text to image (simplified version)

    Args:
        text: Text to render
        width: Image width
        height: Image height

    Returns:
        PIL Image with rendered text
    """
    # Create white background
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)

    # Use default font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Simple text wrapping
    margin = 40
    y_offset = margin
    line_height = 25
    max_width = width - 2 * margin

    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        if len(test_line) * 10 < max_width:  # Approximate
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    # Draw lines
    for line in lines:
        if y_offset + line_height > height - margin:
            break
        draw.text((margin, y_offset), line, fill='black', font=font)
        y_offset += line_height

    return image


# ============================================================================
# Dataset Classes
# ============================================================================

class VisionCompressionDataset(Dataset):
    """
    Dataset for Regime 1: Vision Compression

    Loads from JSONL with pre-rendered images at fixed resolutions.
    Uses FIXED 1000-token context (pre-rendered) + 1000-token continuation.
    """

    def __init__(self, data_path: str, tokenizer, vision_mode: str = 'small', transform=None, max_samples: Optional[int] = None, hybrid_text_tokens: int = 0):
        self.tokenizer = tokenizer
        self.vision_mode = vision_mode
        self.mode_config = VISION_MODES[vision_mode]
        self.transform = transform or BasicImageTransform()
        self.hybrid_text_tokens = hybrid_text_tokens

        # Determine image directory
        # Assume data_path is like "data/training/train.jsonl"
        # Images are in "data/training/images/{mode}/sample_XXXXXX.png"
        self.data_dir = Path(data_path).parent
        self.image_dir = self.data_dir / 'images' / vision_mode

        # Load data from JSONL (with early-exit for max_samples)
        self.samples = []
        desc = f"Loading samples (max={max_samples})" if max_samples else "Loading samples"
        with open(data_path, 'r') as f:
            with tqdm(desc=desc, unit='sample', total=max_samples) as pbar:
                for idx, line in enumerate(f):
                    self.samples.append(json.loads(line))
                    pbar.update(1)
                    # Early exit if max_samples is specified (avoid loading entire file)
                    if max_samples is not None and idx + 1 >= max_samples:
                        break

        logging.info(f"Loaded {len(self.samples)} samples from {data_path}" +
                    (f" (limited to max_samples={max_samples})" if max_samples else ""))
        logging.info(f"Vision mode: {vision_mode} ({self.mode_config['tokens']} tokens, "
                    f"{self.mode_config['image_size']}x{self.mode_config['image_size']})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Load pre-rendered image
        sample_id = sample['id']
        image_path = self.image_dir / f"{sample_id}.png"

        if not image_path.exists():
            raise FileNotFoundError(f"Pre-rendered image not found: {image_path}")

        # Load image using torchvision.io (1.33x faster than PIL)
        image_tensor = torchvision.io.decode_image(
            str(image_path),
            mode=torchvision.io.ImageReadMode.RGB
        )  # uint8 [C, H, W]

        # 2. Pad to base_size (should already be correct size, but ensures consistency)
        base_size = self.mode_config['base_size']
        _, h, w = image_tensor.shape
        if h != base_size or w != base_size:
            pad_value = int(self.transform.mean[0] * 255)
            padded = torch.full((3, base_size, base_size), pad_value, dtype=torch.uint8)
            y_offset = (base_size - h) // 2
            x_offset = (base_size - w) // 2
            padded[:, y_offset:y_offset+h, x_offset:x_offset+w] = image_tensor
            image_tensor = padded

        # 3. Return uint8 tensor - normalization happens on GPU in trainer (32.9x faster)
        # image_tensor is already uint8 [C, H, W]

        # 4. Get continuation tokens (pre-tokenized in dataset)
        continuation_tokens = sample['continuation_tokens']

        # 5. Convert continuation to tensor
        continuation_tokens_tensor = torch.tensor(continuation_tokens, dtype=torch.long)

        # 6. Get context tokens (needed for reconstruction objective)
        context_tokens = sample['context_tokens']
        context_tokens_tensor = torch.tensor(context_tokens, dtype=torch.long)

        # 7. Get hybrid text tokens if enabled (last K tokens from context)
        hybrid_text = []
        if self.hybrid_text_tokens > 0:
            hybrid_text = context_tokens[-self.hybrid_text_tokens:]

        return {
            'image': image_tensor,
            'continuation': continuation_tokens_tensor,
            'context': context_tokens_tensor,  # For reconstruction objective
            'hybrid_text': hybrid_text,
            'sample_id': sample_id,
            'image_path': str(image_path)
        }


class TextBaselineDataset(Dataset):
    """
    Dataset for Regime 2: Text Baseline

    Loads from JSONL with context + continuation token sequences.
    Optionally truncates the context to the most recent `context_length` tokens.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_samples: Optional[int] = None,
        context_length: Optional[int] = None,
        hybrid_text_tokens: int = 0
    ):
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.hybrid_text_tokens = hybrid_text_tokens

        # Load data from JSONL (with early-exit for max_samples)
        self.samples = []
        desc = f"Loading samples (max={max_samples})" if max_samples else "Loading samples"
        with open(data_path, 'r') as f:
            with tqdm(desc=desc, unit='sample', total=max_samples) as pbar:
                for idx, line in enumerate(f):
                    self.samples.append(json.loads(line))
                    pbar.update(1)
                    # Early exit if max_samples is specified (avoid loading entire file)
                    if max_samples is not None and idx + 1 >= max_samples:
                        break

        logging.info(f"Loaded {len(self.samples)} samples from {data_path}" +
                    (f" (limited to max_samples={max_samples})" if max_samples else ""))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        context_tokens = sample['context_tokens']
        if self.context_length is not None:
            if self.context_length < 0:
                raise ValueError("context_length must be non-negative")
            if self.context_length == 0:
                context_tokens = []
            elif len(context_tokens) > self.context_length:
                context_tokens = context_tokens[-self.context_length:]

        # Extract hybrid text if enabled (last K tokens from context)
        # Note: For compression regimes (meanpool, conv1d_residual), these tokens are ALSO
        # included in the compression - this redundancy is intentional for better quality
        hybrid_text = []
        if self.hybrid_text_tokens > 0:
            if self.hybrid_text_tokens > len(context_tokens):
                raise ValueError(
                    f"hybrid_text_tokens ({self.hybrid_text_tokens}) exceeds context length "
                    f"({len(context_tokens)}) for sample {idx}"
                )
            hybrid_text = context_tokens[-self.hybrid_text_tokens:]

        # Convert to tensors in worker process (parallel across workers)
        # This is much faster than converting in main process (collate_fn)
        # because workers run in parallel (8x speedup for 8 workers)
        return {
            'context': torch.tensor(context_tokens, dtype=torch.long),
            'continuation': torch.tensor(sample['continuation_tokens'], dtype=torch.long),
            'hybrid_text': hybrid_text,
            'sample_id': sample.get('id', f'sample_{idx}')
        }


# ============================================================================
# Dataset Format Detection Helper
# ============================================================================

def is_arrow_format(data_path: str) -> bool:
    """
    Detect if data_path points to Arrow format (directory with metadata) or JSONL (file).

    Arrow datasets are stored as directories created by convert_to_arrow.py,
    containing dataset_info.json and other Arrow metadata files.
    JSONL datasets are single files with .jsonl extension.

    Args:
        data_path: Path to dataset (either Arrow directory or JSONL file)

    Returns:
        True if Arrow format (directory with dataset_info.json), False otherwise
    """
    path = Path(data_path)
    # Arrow datasets have dataset_info.json inside the directory
    return path.is_dir() and (path / "dataset_info.json").exists()


def create_vision_dataset(
    data_path: str,
    tokenizer,
    vision_mode: str,
    max_samples: Optional[int] = None,
    hybrid_text_tokens: int = 0
):
    """
    Create vision dataset with auto-detection of format (Arrow or JSONL).

    Args:
        data_path: Path to Arrow directory or JSONL file
        tokenizer: Tokenizer instance
        vision_mode: Vision mode ('tiny', 'small', 'base', 'large')
        max_samples: Optional limit on samples
        hybrid_text_tokens: Number of hybrid text tokens

    Returns:
        ArrowVisionDataset or VisionCompressionDataset
    """
    if is_arrow_format(data_path):
        logging.info(f"Detected Arrow format: {data_path}")
        return ArrowVisionDataset(
            data_path,
            tokenizer,
            vision_mode=vision_mode,
            max_samples=max_samples,
            hybrid_text_tokens=hybrid_text_tokens
        )
    else:
        logging.info(f"Detected JSONL format: {data_path}")
        return VisionCompressionDataset(
            data_path,
            tokenizer,
            vision_mode=vision_mode,
            max_samples=max_samples,
            hybrid_text_tokens=hybrid_text_tokens
        )


def create_text_dataset(
    data_path: str,
    tokenizer,
    max_samples: Optional[int] = None,
    context_length: Optional[int] = None,
    hybrid_text_tokens: int = 0
):
    """
    Create text dataset with auto-detection of format (Arrow or JSONL).

    Args:
        data_path: Path to Arrow directory or JSONL file
        tokenizer: Tokenizer instance
        max_samples: Optional limit on samples
        context_length: Optional context truncation length
        hybrid_text_tokens: Number of hybrid text tokens

    Returns:
        ArrowTextDataset or TextBaselineDataset
    """
    if is_arrow_format(data_path):
        logging.info(f"Detected Arrow format: {data_path}")
        return ArrowTextDataset(
            data_path,
            tokenizer,
            max_samples=max_samples,
            context_length=context_length,
            hybrid_text_tokens=hybrid_text_tokens
        )
    else:
        logging.info(f"Detected JSONL format: {data_path}")
        return TextBaselineDataset(
            data_path,
            tokenizer,
            max_samples=max_samples,
            context_length=context_length,
            hybrid_text_tokens=hybrid_text_tokens
        )


# ============================================================================
# Batch Preparation Helpers
# ============================================================================

def prepare_text_batch(context_tokens: torch.Tensor, continuation_tokens: torch.Tensor):
    """
    Prepare text batch by concatenating BOS, context and continuation, creating labels.

    This helper is used by both the collate function (during data loading) and
    the trainer's prepare_batch method (for backward compatibility with tests).

    Args:
        context_tokens: Token IDs for text context (shape: [batch_size, seq_len])
        continuation_tokens: Token IDs for text continuation (shape: [batch_size, seq_len])

    Returns:
        Tuple of (input_ids, labels)
        - input_ids: [BOS] + context + continuation tokens
        - labels: -100 for BOS and context positions, actual tokens for continuation
    """
    batch_size = context_tokens.shape[0]
    context_len = context_tokens.shape[1]

    # Add BOS token at the beginning (match official DeepSeek-OCR implementation)
    bos_id = BOS_TOKEN_ID
    bos_tokens = context_tokens.new_full((batch_size, 1), bos_id)

    # Concatenate: [BOS] + [CONTEXT_TOKENS] + [CONTINUATION_TOKENS]
    input_ids = torch.cat([bos_tokens, context_tokens, continuation_tokens], dim=1)

    # Create labels: -100 for BOS and context (ignored in loss), actual tokens for continuation
    # Use context_tokens.new_full() to preserve device and dtype
    labels = torch.cat([
        context_tokens.new_full((batch_size, 1), -100),  # BOS position masked
        context_tokens.new_full((batch_size, context_len), -100),  # Context masked
        continuation_tokens  # Only continuation contributes to loss
    ], dim=1)

    return input_ids, labels


# ============================================================================
# Collate Functions (for batching)
# ============================================================================

def vision_collate_fn(batch):
    """Collate function for vision compression dataset

    All images are pre-rendered to fixed sizes and all continuation tokens
    are exactly 1000 tokens, so no padding is needed - just stack tensors.

    For hybrid mode, also handles stacking of text tokens (if present).
    """
    # Check if hybrid mode is enabled (first item has non-empty hybrid_text)
    has_hybrid_text = len(batch[0]['hybrid_text']) > 0

    result = {
        'image': torch.stack([item['image'] for item in batch]),
        'continuation': torch.stack([item['continuation'] for item in batch]),
        'context': torch.stack([item['context'] for item in batch]),
        'sample_id': [item['sample_id'] for item in batch],
        'image_path': [item['image_path'] for item in batch]
    }

    # Add hybrid text tokens if present
    if has_hybrid_text:
        hybrid_text_tensors = [torch.tensor(item['hybrid_text'], dtype=torch.long) for item in batch]
        result['hybrid_text'] = torch.stack(hybrid_text_tensors)
    else:
        result['hybrid_text'] = None

    return result


def text_collate_fn(batch):
    """Collate function for text baseline dataset

    Context length can vary (depending on truncation), but all samples share the same
    length per batch so stacking works without padding.

    Note: Tensors are created in parallel by worker processes, so this function
    just stacks them (very fast). This is the standard PyTorch pattern and
    avoids blocking the main process with tensor conversion.
    """
    # Stack pre-tensorized batches (workers already converted lists to tensors)
    context_tokens = torch.stack([item['context'] for item in batch])
    continuation_tokens = torch.stack([item['continuation'] for item in batch])

    # Prepare input_ids and labels (concatenation + label masking)
    input_ids, labels = prepare_text_batch(context_tokens, continuation_tokens)

    # Return prepared batch (don't include context/continuation to avoid memory duplication)
    return {
        'input_ids': input_ids,
        'labels': labels,
        'sample_id': [item['sample_id'] for item in batch]
    }


def compression_collate_fn(batch):
    """Collate function for compression regimes (meanpool, conv1d_residual, etc.)

    Keeps context and continuation separate (not concatenated) so the trainer
    can choose which to use based on objective (lm vs reconstruction).

    Note: Tensors are created in parallel by worker processes, so this function
    just stacks them (very fast). This is the standard PyTorch pattern and
    avoids blocking the main process with tensor conversion.
    """
    # Stack pre-tensorized batches (workers already converted lists to tensors)
    context_tokens = torch.stack([item['context'] for item in batch])
    continuation_tokens = torch.stack([item['continuation'] for item in batch])

    # Check if hybrid mode is enabled (first item has non-empty hybrid_text)
    has_hybrid_text = len(batch[0]['hybrid_text']) > 0

    result = {
        'context': context_tokens,
        'continuation': continuation_tokens,
        'sample_id': [item['sample_id'] for item in batch]
    }

    # Add hybrid text tokens if present
    # Note: hybrid_text is returned as a list from __getitem__ (not pre-tensorized like context/continuation)
    # so it needs conversion to tensor here before stacking
    if has_hybrid_text:
        hybrid_text_tensors = [torch.tensor(item['hybrid_text'], dtype=torch.long) for item in batch]
        result['hybrid_text'] = torch.stack(hybrid_text_tensors)
    else:
        result['hybrid_text'] = None

    return result


# ============================================================================
# Training Functions
# ============================================================================

def maybe_save_periodic_checkpoint(
    trainer, optimizer, scheduler,
    args, global_step, epoch, batch_idx,
    loss, best_val_loss,
    output_dir, logger, wandb_run_id=None,
    sampler_state=None
):
    """
    Save periodic checkpoint if save_steps is configured and step matches

    Returns:
        True if checkpoint was saved, False otherwise
    """
    if args.save_steps > 0 and not args.no_checkpoints and global_step % args.save_steps == 0:
        checkpoint_path = output_dir / f'checkpoint_step_{global_step}.pt'
        save_checkpoint(
            trainer.model, optimizer, scheduler,
            epoch, batch_idx, global_step,
            loss, None,
            best_val_loss,
            args, checkpoint_path, logger,
            train_perplexity=compute_perplexity(loss),
            val_perplexity=None,
            wandb_run_id=wandb_run_id,
            sampler_state=sampler_state
        )
        return True
    return False


def maybe_run_periodic_eval(
    trainer, val_dataloader, regime,
    args, global_step, epoch, batch_idx,
    train_loss, best_val_loss,
    output_dir, optimizer, scheduler, logger, wandb_run_id=None,
    sampler_state=None
):
    """
    Run periodic evaluation if eval_steps is configured and step matches

    Returns:
        Tuple of (updated_best_val_loss, checkpoint_saved)
    """
    if val_dataloader and args.eval_steps > 0 and global_step % args.eval_steps == 0:
        logger.info(f"\nRunning validation at step {global_step}...")
        val_metrics = evaluate(
            trainer, val_dataloader, regime, logger, args, global_step,
            num_qualitative_samples=args.num_qualitative_samples,
            max_generation_tokens=args.max_generation_tokens,
            output_dir=output_dir,
            eval_seed=args.eval_seed
        )

        # Save best model if validation improved
        checkpoint_saved = False
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            if not args.no_checkpoints:
                best_checkpoint_path = output_dir / 'best_checkpoint.pt'
                save_checkpoint(
                    trainer.model, optimizer, scheduler,
                    epoch, batch_idx, global_step,
                    train_loss, val_metrics['loss'],
                    best_val_loss,
                    args, best_checkpoint_path, logger,
                    train_perplexity=compute_perplexity(train_loss),
                    val_perplexity=val_metrics['perplexity'],
                    wandb_run_id=wandb_run_id,
                    sampler_state=sampler_state
                )
                logger.info(f"New best validation loss: {best_val_loss:.4f}, perplexity: {val_metrics['perplexity']:.2f}")
                checkpoint_saved = True
            else:
                logger.info(f"New best validation loss: {best_val_loss:.4f}, perplexity: {val_metrics['perplexity']:.2f} (checkpoint saving disabled)")

            # Log best model update to wandb
            if args.use_wandb:
                wandb.log({
                    'val/best_loss': best_val_loss,
                    'val/best_perplexity': val_metrics['perplexity']
                }, step=global_step)

        # Return to training mode
        trainer.model.train()

        return best_val_loss, checkpoint_saved

    return best_val_loss, False


def train_epoch(
    trainer,
    dataloader,
    val_dataloader,
    optimizer,
    scheduler,
    regime: str,
    epoch: int,
    args,
    logger,
    global_step: int,
    best_val_loss: float,
    output_dir: Path,
    wandb_run_id: Optional[str] = None,
    initial_skip_batches: int = 0
):
    """
    Train for one epoch with step-based validation

    Args:
        wandb_run_id: Weights & Biases run ID (optional)

    Returns:
        Dict with training metrics, updated global_step, and best_val_loss
    """
    trainer.model.train()

    # Compute sampler state once for all checkpoint saves in this epoch
    # (permutation is immutable, so this is safe and efficient)
    sampler_state = _train_sampler.get_state() if _train_sampler else None

    # Use GPU tensors for loss tracking to avoid CPU-GPU sync every batch
    # Keep in float32 for precision (model outputs float32 loss from CrossEntropyLoss)
    device = torch.device(args.device)
    total_loss = torch.zeros((), device=device)
    accumulated_loss = torch.zeros((), device=device)

    total_grad_norm = 0.0
    # Initialize num_steps to account for already-completed gradient steps on resume
    # This ensures log output shows correct epoch-level step count (e.g., "Step 2082" not "Step 10")
    num_steps = initial_skip_batches // args.gradient_accumulation_steps
    num_batches = 0

    # Track accumulated batches for gradient accumulation
    accumulated_batches = 0

    # Track epoch timing for throughput calculation
    epoch_start_time = time.time()
    step_timer_start = time.time()  # For step-level throughput
    tokens_since_last_log = 0  # Track tokens in current logging window

    progress_bar = tqdm(
        dataloader,
        desc=f"Epoch {epoch+1}/{args.num_epochs}",
        initial=initial_skip_batches  # Start progress bar at skip position for absolute count
    )

    # Open debug log file if requested
    debug_log_file = None
    if args.debug_log_sample_ids:
        debug_log_path = output_dir / 'sample_ids_log.jsonl'
        debug_log_file = open(debug_log_path, 'a')  # Append mode for resume support

    for step, batch in enumerate(progress_bar):
        # Check for graceful shutdown request (from SIGTERM/SIGINT)
        if _shutdown_requested:
            global _shutdown_in_progress
            _shutdown_in_progress = True

            logger.info("[SHUTDOWN] Saving interrupted checkpoint...")
            interrupted_checkpoint_path = output_dir / 'interrupted_checkpoint.pt'
            # Compute current train loss for checkpoint
            current_train_loss = (accumulated_loss.item() / max(accumulated_batches, 1)) if accumulated_batches > 0 else 0.0

            # Compute (epoch, batch_idx) for checkpoint
            # step is the batch we're ABOUT to process, so last completed is step-1
            if step == 0:
                if initial_skip_batches > 0:
                    # Resumed mid-epoch but no new batches completed yet
                    # Save pointing to batch before resume point (will redo that batch)
                    checkpoint_epoch = epoch
                    completed_batch_idx = initial_skip_batches - 1
                else:
                    # Fresh epoch, no batches completed yet
                    # Save as "end of previous epoch" so resume starts at current epoch
                    # Note: epoch-1 may be -1 if epoch=0, which is fine (resumes at epoch 0)
                    checkpoint_epoch = epoch - 1
                    completed_batch_idx = -1
            else:
                # Normal case: round down to last complete optimizer step boundary
                # This ensures no data loss on resume - partially accumulated batches
                # will be re-processed rather than skipped
                checkpoint_epoch = epoch
                processed = (step - 1) + initial_skip_batches
                applied = ((processed + 1) // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
                completed_batch_idx = max(applied - 1, -1)

            logger.info(f"[SHUTDOWN] Saving checkpoint: epoch={checkpoint_epoch}, batch_idx={completed_batch_idx}, global_step={global_step}")
            save_checkpoint(
                trainer.model, optimizer, scheduler,
                checkpoint_epoch, completed_batch_idx, global_step,
                current_train_loss, None,  # train_loss, val_loss (no val during interrupt)
                best_val_loss,
                args, interrupted_checkpoint_path, logger,
                train_perplexity=compute_perplexity(current_train_loss) if current_train_loss > 0 else None,
                val_perplexity=None,
                wandb_run_id=wandb_run_id,
                sampler_state=sampler_state
            )
            logger.info(f"[SHUTDOWN] Checkpoint saved to {interrupted_checkpoint_path}")
            logger.info("[SHUTDOWN] Exiting gracefully.")

            # Close debug log file if open
            if debug_log_file is not None:
                debug_log_file.close()

            sys.exit(0)

        # [DEBUG] Log sample IDs for this batch
        if debug_log_file is not None and 'sample_id' in batch:
            import json
            sample_ids = batch['sample_id'].tolist() if hasattr(batch['sample_id'], 'tolist') else batch['sample_id']
            log_entry = {
                'epoch': epoch,
                'step': step,
                'global_step': global_step + num_batches,
                'sample_ids': sample_ids
            }
            debug_log_file.write(json.dumps(log_entry) + '\n')
            debug_log_file.flush()

        # Forward pass
        objective = args.objective
        if objective not in ['lm', 'reconstruction']:
            raise ValueError(f"Invalid objective: {objective}. Must be 'lm' or 'reconstruction'")

        if regime == 'vision':
            image = batch['image']
            hybrid_text = batch.get('hybrid_text', None)

            # Determine target based on objective
            if objective == 'lm':
                target = batch['continuation']
            else:  # reconstruction
                target = batch['context']

            loss, labels = trainer.forward(image, target, hybrid_text, objective=objective)

        elif regime in ['text', 'meanpool', 'conv1d_residual']:
            # Text-based regimes: extract context and target from batch
            context = batch['context'] if 'context' in batch else None
            continuation = batch['continuation'] if 'continuation' in batch else None
            hybrid_text = batch.get('hybrid_text', None)

            # Determine target based on objective
            if objective == 'lm':
                target = continuation
            else:  # reconstruction
                target = context  # Reconstruct the context itself

            if context is None or target is None:
                raise ValueError(f"Batch missing required keys for {regime} regime")

            # Pass hybrid_text for regimes that support it
            if regime in ['meanpool', 'conv1d_residual']:
                loss, labels = trainer.forward(context, target, hybrid_text, objective=objective)
            else:
                loss, labels = trainer.forward(context, target, objective=objective)

        # Log effective context tokens once (count tokens masked from loss)
        global _effective_context_logged, _target_tokens_per_sample
        if not _effective_context_logged:
            # Count tokens where labels == -100 (per-sample, batch size irrelevant)
            effective_context_tokens = (labels[0] == -100).sum().item()

            # Count target tokens (tokens contributing to loss)
            target_tokens_per_sample = (labels[0] != -100).sum().item()

            # Calculate compression ratio vs baseline 1000-token context
            if effective_context_tokens > 0:
                compression_ratio = 1000 / effective_context_tokens
            else:
                compression_ratio = float('inf')

            # Log to W&B config for filtering/comparing runs by budget
            # Skip when resuming same W&B run - config was already set in original run
            if args.use_wandb and WANDB_AVAILABLE and not wandb.run.resumed:
                wandb.config.update({
                    'effective_context_tokens': effective_context_tokens,
                    'compression_ratio': compression_ratio
                })

            logger.info(f"Effective context tokens (per-sample): {effective_context_tokens} | Compression ratio: {compression_ratio:.2f}x")
            logger.info(f"Target tokens per sample: {target_tokens_per_sample}")
            _effective_context_logged = True
            _target_tokens_per_sample = target_tokens_per_sample

        # Track unscaled loss from ALL batches (not just last in accumulation window)
        # Use detach() to accumulate on GPU without sync (no .item() call)
        accumulated_loss += loss.detach()
        accumulated_batches += 1
        num_batches += 1

        # Track tokens for throughput calculation (batch size from labels)
        if _target_tokens_per_sample > 0:
            batch_size_actual = labels.shape[0]
            tokens_since_last_log += _target_tokens_per_sample * batch_size_actual

        # Normalize loss for gradient accumulation
        loss = loss / args.gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Gradient accumulation
        if accumulated_batches % args.gradient_accumulation_steps == 0:
            # Compute encoder/decoder gradient norms BEFORE clipping (for accurate monitoring)
            if args.train_encoder:
                # Use cached param lists (computed once at optimizer creation)
                encoder_params = trainer.model._cached_encoder_params
                decoder_params = trainer.model._cached_decoder_params

                # Compute norms without clipping (use inf threshold)
                encoder_grad_norm_value = float(torch.nn.utils.clip_grad_norm_(encoder_params, float('inf')))
                decoder_grad_norm_value = float(torch.nn.utils.clip_grad_norm_(decoder_params, float('inf')))
            else:
                encoder_grad_norm_value = None
                decoder_grad_norm_value = None

            # Gradient clipping (CRITICAL for MoE stability)
            # Applied AFTER computing per-group norms for logging
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in trainer.model.parameters() if p.requires_grad],
                max_norm=args.max_grad_norm
            )
            # Cache grad_norm value (single CPU-GPU sync, reused for logging)
            grad_norm_value = float(grad_norm)

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Scheduler step
            if scheduler is not None:
                scheduler.step()

            # Track metrics (average loss over accumulation window)
            # Compute average on GPU first, then sync once to CPU for logging
            avg_accumulated_loss_tensor = accumulated_loss / accumulated_batches
            avg_accumulated_loss = float(avg_accumulated_loss_tensor)  # Single CPU-GPU sync
            total_loss += avg_accumulated_loss_tensor
            total_grad_norm += grad_norm_value
            num_steps += 1
            global_step += 1

            # Reset accumulation trackers
            accumulated_loss.zero_()
            accumulated_batches = 0

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{avg_accumulated_loss:.4f}',
                'ppl': f'{compute_perplexity(avg_accumulated_loss):.2f}',
                'grad_norm': f'{grad_norm_value:.2f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
            })

            # Log periodically (both wandb and console)
            should_log = args.log_steps > 0 and num_steps % args.log_steps == 0

            # Calculate step-level throughput (since last log)
            if should_log:
                step_timer_end = time.time()
                step_duration = step_timer_end - step_timer_start
                tokens_per_sec = tokens_since_last_log / step_duration if step_duration > 0 else 0
            else:
                tokens_per_sec = 0  # Only calculated when logging

            if args.use_wandb and should_log:
                log_dict = {
                    'train/loss': avg_accumulated_loss,
                    'train/perplexity': compute_perplexity(avg_accumulated_loss),
                    'train/grad_norm': grad_norm_value,
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'train/tokens_per_sec': tokens_per_sec,
                    'train/epoch': epoch + ((step + initial_skip_batches + 1) / len(dataloader))
                }

                # Add separate encoder/decoder gradient norms if training encoder (use cached values)
                if args.train_encoder:
                    log_dict['train/encoder_grad_norm'] = encoder_grad_norm_value
                    log_dict['train/decoder_grad_norm'] = decoder_grad_norm_value

                    # Find learning rates by param group name (not positional index)
                    for group in optimizer.param_groups:
                        if group.get('name') == 'encoder':
                            log_dict['train/encoder_lr'] = group['lr']
                        elif group.get('name') == 'decoder':
                            log_dict['train/decoder_lr'] = group['lr']

                wandb.log(log_dict, step=global_step)

            if should_log:
                logger.info(
                    f"Epoch {epoch+1} Step {num_steps} (Global: {global_step}): "
                    f"loss={avg_accumulated_loss:.4f}, "
                    f"ppl={compute_perplexity(avg_accumulated_loss):.2f}, "
                    f"grad_norm={grad_norm_value:.2f}, "
                    f"lr={optimizer.param_groups[0]['lr']:.2e}, "
                    f"throughput={tokens_per_sec:.0f} tok/s"
                )

            # Periodic checkpoint saving (if save_steps > 0)
            maybe_save_periodic_checkpoint(
                trainer, optimizer, scheduler,
                args, global_step, epoch, step + initial_skip_batches,
                avg_accumulated_loss, best_val_loss,
                output_dir, logger, wandb_run_id,
                sampler_state=sampler_state
            )

            # Step-based validation (if eval_steps > 0 and val_dataloader provided)
            best_val_loss, _ = maybe_run_periodic_eval(
                trainer, val_dataloader, regime,
                args, global_step, epoch, step + initial_skip_batches,
                avg_accumulated_loss, best_val_loss,
                output_dir, optimizer, scheduler, logger, wandb_run_id,
                sampler_state=sampler_state
            )

            # Reset timer and token counter AFTER checkpoint/eval complete
            # This ensures only training time contributes to throughput metrics
            if should_log:
                step_timer_start = time.time()
                tokens_since_last_log = 0

    # Post-loop: Flush any remaining accumulated gradients
    if accumulated_batches > 0:
        logger.info(f"Flushing {accumulated_batches} remainder batches from gradient accumulation")

        # Rescale gradients to compensate for partial accumulation window
        # Each loss was divided by gradient_accumulation_steps, but we only accumulated
        # accumulated_batches of them, so we need to scale up by this factor
        scale_factor = args.gradient_accumulation_steps / accumulated_batches
        logger.info(f"  Rescaling gradients by {scale_factor:.2f}x (compensating for {accumulated_batches}/{args.gradient_accumulation_steps} batches)")

        for param in trainer.model.parameters():
            if param.grad is not None:
                param.grad.mul_(scale_factor)

        # Gradient clipping (after rescaling)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in trainer.model.parameters() if p.requires_grad],
            max_norm=args.max_grad_norm
        )
        # Cache grad_norm value (single CPU-GPU sync, reused for logging)
        grad_norm_value = float(grad_norm)

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Scheduler step
        if scheduler is not None:
            scheduler.step()

        # Track metrics (same GPU-side pattern as main loop)
        avg_accumulated_loss_tensor = accumulated_loss / accumulated_batches
        avg_accumulated_loss = float(avg_accumulated_loss_tensor)  # Single CPU-GPU sync
        total_loss += avg_accumulated_loss_tensor
        total_grad_norm += grad_norm_value
        num_steps += 1
        global_step += 1

        logger.info(f"Remainder batch: loss={avg_accumulated_loss:.4f}, ppl={compute_perplexity(avg_accumulated_loss):.2f}, grad_norm={grad_norm_value:.2f}")

        # Check for periodic checkpoint/eval on remainder step
        # Use step from last batch in the loop (step variable is still in scope)
        maybe_save_periodic_checkpoint(
            trainer, optimizer, scheduler,
            args, global_step, epoch, step + initial_skip_batches,
            avg_accumulated_loss, best_val_loss,
            output_dir, logger, wandb_run_id,
            sampler_state=sampler_state
        )

        best_val_loss, _ = maybe_run_periodic_eval(
            trainer, val_dataloader, regime,
            args, global_step, epoch, step + initial_skip_batches,
            avg_accumulated_loss, best_val_loss,
            output_dir, optimizer, scheduler, logger, wandb_run_id,
            sampler_state=sampler_state
        )

    # Final averaging - sync GPU tensor to CPU once at epoch end
    avg_loss = float(total_loss / num_steps) if num_steps > 0 else 0.0
    avg_grad_norm = total_grad_norm / num_steps if num_steps > 0 else 0

    # Calculate epoch-level throughput
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    total_target_tokens = _target_tokens_per_sample * num_batches * args.batch_size if _target_tokens_per_sample > 0 else 0
    epoch_tokens_per_sec = total_target_tokens / epoch_duration if epoch_duration > 0 else 0

    # Close debug log file if it was opened
    if debug_log_file is not None:
        debug_log_file.close()

    return {
        'loss': avg_loss,
        'perplexity': compute_perplexity(avg_loss),
        'grad_norm': avg_grad_norm,
        'num_steps': num_steps,
        'global_step': global_step,
        'best_val_loss': best_val_loss,
        'epoch_duration': epoch_duration,
        'epoch_tokens_per_sec': epoch_tokens_per_sec
    }


def evaluate(
    trainer,
    dataloader,
    regime: str,
    logger,
    args=None,
    global_step: Optional[int] = None,
    num_qualitative_samples: int = 0,
    max_generation_tokens: int = 200,
    output_dir: Optional[Path] = None,
    eval_seed: int = 42
):
    """
    Evaluate on validation set with quantitative and qualitative evaluation

    Args:
        trainer: VisionCompressionTrainer or TextBaselineTrainer
        dataloader: Validation dataloader
        regime: 'vision' or 'text'
        logger: Logger instance
        args: Training arguments (optional, for wandb logging)
        global_step: Global training step (optional, for wandb/file naming)
        num_qualitative_samples: Number of samples to generate text for (0 = skip)
        max_generation_tokens: Maximum tokens to generate per sample
        output_dir: Output directory for saving qualitative samples
        eval_seed: Seed for deterministic validation (ensures reproducible random subsampling)

    Returns:
        Dict with evaluation metrics and qualitative samples
    """
    trainer.model.eval()

    # Quantitative evaluation
    total_loss = 0.0
    num_samples = 0

    # Qualitative evaluation
    qualitative_samples = []

    # Create generator for deterministic random operations (validation only)
    eval_generator = torch.Generator(device='cpu')
    eval_generator.manual_seed(eval_seed)

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Forward pass for loss (quantitative)
            objective = args.objective if args else 'lm'

            if regime == 'vision':
                image = batch['image']
                hybrid_text = batch.get('hybrid_text', None)

                # Determine target based on objective
                if objective == 'lm':
                    target = batch['continuation']
                else:  # reconstruction
                    target = batch['context']

                loss, _ = trainer.forward(image, target, hybrid_text, objective=objective)

            elif regime in ['text', 'meanpool', 'conv1d_residual']:
                # Text-based regimes: extract context and target from batch
                context = batch['context']
                continuation = batch['continuation']
                hybrid_text = batch.get('hybrid_text', None)

                # Determine target based on objective
                if objective == 'lm':
                    target = continuation
                else:  # reconstruction
                    target = context

                # Pass hybrid_text for regimes that support it
                if regime in ['meanpool', 'conv1d_residual']:
                    loss, _ = trainer.forward(context, target, hybrid_text, objective=objective)
                else:  # text
                    loss, _ = trainer.forward(context, target, objective=objective)

            total_loss += loss.item()
            num_samples += 1

            # Qualitative evaluation: collect samples from batch until we have enough
            batch_size = len(batch['sample_id'])
            for sample_idx in range(batch_size):
                # Stop if we have collected enough samples
                if len(qualitative_samples) >= num_qualitative_samples:
                    break

                sample_id = batch['sample_id'][sample_idx]
                image_path = batch.get('image_path', [None] * batch_size)[sample_idx]

                try:
                    if regime == 'vision':
                        # Vision regime: extract context from image
                        # Try to get human-readable context (not always available)
                        if trainer.vision_prompt:
                            context_prompt = f"[Image: {sample_id}] + \"{trainer.vision_prompt}\""
                        else:
                            context_prompt = f"[Image: {sample_id}]"

                        # Extract hybrid text for this sample (if hybrid mode is enabled)
                        sample_hybrid = None
                        if hybrid_text is not None:
                            sample_hybrid = hybrid_text[sample_idx]

                        # Generate text from image (with hybrid text to match training distribution)
                        generated_text = trainer.generate_text(
                            image[sample_idx],  # Sample at index
                            hybrid_text_tokens=sample_hybrid,
                            max_new_tokens=max_generation_tokens,
                            temperature=0.0  # Greedy for consistency
                        )

                        # Ground truth: depends on objective
                        if objective == 'reconstruction':
                            # For reconstruction: show original context (what we're reconstructing from image)
                            ground_truth_tokens = batch['context'][sample_idx, :max_generation_tokens]
                        else:  # lm
                            # For LM: show continuation (what we're predicting)
                            ground_truth_tokens = batch['continuation'][sample_idx, :max_generation_tokens]

                        ground_truth = trainer.tokenizer.decode(
                            ground_truth_tokens,
                            skip_special_tokens=True
                        )

                    elif regime in ['meanpool', 'conv1d_residual']:
                        # Compression regimes: extract context from batch
                        context_tokens = batch['context'][sample_idx]

                        # Extract hybrid text if regime supports it
                        sample_hybrid = None
                        hybrid_text_batch = batch.get('hybrid_text', None)
                        if hybrid_text_batch is not None:
                            sample_hybrid = hybrid_text_batch[sample_idx]

                        # Decode context for logging (show compressed representation)
                        # Generate text (meanpool/conv1d_residual need hybrid_text)
                        if regime == 'meanpool':
                            context_prompt = f"[Mean pooled from {context_tokens.shape[0]} tokens]"
                        else:  # conv1d_residual
                            context_prompt = f"[Conv1D Residual compressed from {context_tokens.shape[0]} tokens]"

                        generated_text = trainer.generate_text(
                            context_tokens,
                            hybrid_text_tokens=sample_hybrid,
                            max_new_tokens=max_generation_tokens,
                            temperature=0.0  # Greedy for consistency
                        )

                        # Ground truth: depends on objective
                        if objective == 'reconstruction':
                            # For reconstruction: show original context (what we're trying to reconstruct)
                            ground_truth_tokens = batch['context'][sample_idx, :max_generation_tokens]
                        else:  # lm
                            # For LM: show continuation (what we're trying to predict)
                            ground_truth_tokens = batch['continuation'][sample_idx, :max_generation_tokens]

                        ground_truth = trainer.tokenizer.decode(
                            ground_truth_tokens,
                            skip_special_tokens=True
                        )

                    else:  # text regime
                        # Text regime now uses compression_collate_fn (context/continuation)
                        context_tokens = batch['context'][sample_idx]

                        # Decode context for logging
                        context_prompt = trainer.tokenizer.decode(
                            context_tokens,
                            skip_special_tokens=True
                        )

                        # Generate text from context
                        generated_text = trainer.generate_text(
                            context_tokens,
                            max_new_tokens=max_generation_tokens,
                            temperature=0.0  # Greedy for consistency
                        )

                        # Ground truth: decode continuation tokens (up to max_generation_tokens)
                        ground_truth_tokens = batch['continuation'][sample_idx, :max_generation_tokens]
                        ground_truth = trainer.tokenizer.decode(
                            ground_truth_tokens,
                            skip_special_tokens=True
                        )

                    # Check for empty generation (silent failure)
                    if not generated_text.strip():
                        logger.warning(f"Empty generation for sample {sample_id}")
                        generated_text = "[EMPTY GENERATION]"

                    # Store qualitative sample
                    qualitative_samples.append({
                        'sample_id': sample_id,
                        'image_path': image_path if regime == 'vision' else None,
                        'context_prompt': context_prompt,
                        'generated_text': generated_text,
                        'ground_truth': ground_truth,
                        'generation_tokens': max_generation_tokens
                    })

                except Exception as e:
                    logger.error(f"Failed to generate text for sample {sample_id}: {e}")
                    qualitative_samples.append({
                        'sample_id': sample_id,
                        'image_path': image_path if regime == 'vision' else None,
                        'context_prompt': f"[Error: {sample_id}]",
                        'generated_text': f"[GENERATION FAILED: {str(e)}]",
                        'ground_truth': "[N/A]",
                        'generation_tokens': max_generation_tokens
                    })

    # Compute quantitative metrics
    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    perplexity = compute_perplexity(avg_loss)

    # Compute reconstruction metrics on qualitative samples
    qualitative_metrics = {}
    if qualitative_samples:
        metrics_per_sample = []
        for sample in qualitative_samples:
            # Skip error samples
            if "[GENERATION FAILED" in sample['generated_text']:
                continue

            # Compute metrics for this sample
            try:
                sample_metrics = cal_per_metrics(
                    None,  # predict_root_ not used by the function
                    sample['generated_text'],   # pred = prediction/hypothesis
                    sample['ground_truth']      # gt = ground truth/reference
                )
                metrics_per_sample.append(sample_metrics)
            except LookupError as e:
                # NLTK data missing (wordnet for METEOR)
                if 'wordnet' in str(e).lower() and not hasattr(logger, '_wordnet_warning_shown'):
                    logger.warning(f"NLTK wordnet data missing - METEOR score unavailable. "
                                  f"Run: python -m nltk.downloader wordnet omw-1.4")
                    logger._wordnet_warning_shown = True
                # Skip this sample and continue
                continue

        # Average metrics across samples
        if metrics_per_sample:
            for key in metrics_per_sample[0].keys():
                # Filter out None values (can occur for precision/recall/f_measure with empty sets)
                values = [m[key] for m in metrics_per_sample if m[key] is not None]
                if values:
                    qualitative_metrics[key] = sum(values) / len(values)
                else:
                    # All values were None - set to 0.0
                    qualitative_metrics[key] = 0.0

    logger.info(f"Validation loss: {avg_loss:.4f}, perplexity: {perplexity:.2f}")
    if qualitative_metrics:
        logger.info(f"Qualitative metrics (n={len(metrics_per_sample)}):")
        logger.info(f"  BLEU: {qualitative_metrics['bleu']:.4f}")
        logger.info(f"  METEOR: {qualitative_metrics['meteor']:.4f}")
        logger.info(f"  Edit Distance: {qualitative_metrics['edit_dist']:.4f}")
        logger.info(f"  F-measure: {qualitative_metrics['f_measure']:.4f}")

    # Log qualitative samples to console
    if qualitative_samples:
        logger.info(f"\n{'='*70}")
        logger.info("Qualitative Evaluation Samples:")
        logger.info('='*70)
        for i, sample in enumerate(qualitative_samples, 1):
            logger.info(f"\nSample {i} (ID: {sample['sample_id']}):")
            # Truncate context/generated/ground_truth for console display (200 chars)
            context_display = sample['context_prompt'][:200] + ('...' if len(sample['context_prompt']) > 200 else '')
            generated_display = sample['generated_text'][:200] + ('...' if len(sample['generated_text']) > 200 else '')
            ground_truth_display = sample['ground_truth'][:200] + ('...' if len(sample['ground_truth']) > 200 else '')

            logger.info(f"Context:      {context_display}")
            logger.info(f"Generated:    {repr(generated_display)}")
            logger.info(f"Ground Truth: {repr(ground_truth_display)}")
            logger.info('-'*70)

    # Save qualitative samples to JSONL file
    if qualitative_samples and output_dir:
        output_dir = Path(output_dir)
        step_label = f"step_{global_step}" if global_step is not None else "final"
        qualitative_file = output_dir / f"qualitative_{step_label}.jsonl"

        with open(qualitative_file, 'w') as f:
            for sample in qualitative_samples:
                f.write(json.dumps(sample) + '\n')

        logger.info(f"\nQualitative samples saved to: {qualitative_file}")

    # Log to wandb if enabled
    if args and args.use_wandb and global_step is not None:
        wandb_data = {
            'val/loss': avg_loss,
            'val/perplexity': perplexity
        }

        # Add qualitative metrics to wandb
        if qualitative_metrics:
            wandb_data['val/qual_bleu'] = qualitative_metrics['bleu']
            wandb_data['val/qual_meteor'] = qualitative_metrics['meteor']
            wandb_data['val/qual_edit_dist'] = qualitative_metrics['edit_dist']
            wandb_data['val/qual_f_measure'] = qualitative_metrics['f_measure']
            wandb_data['val/qual_precision'] = qualitative_metrics['precision']
            wandb_data['val/qual_recall'] = qualitative_metrics['recall']

        # Add qualitative samples to wandb as a table
        if qualitative_samples:
            # Truncate for wandb display (full text in JSONL)
            wandb_table_data = []
            for sample in qualitative_samples:
                wandb_table_data.append([
                    sample['sample_id'],
                    sample['context_prompt'][:200],  # Truncate context
                    sample['generated_text'][:200],   # Truncate generated
                    sample['ground_truth'][:200]      # Truncate ground truth
                ])

            wandb_data['val/qualitative_samples'] = wandb.Table(
                columns=['sample_id', 'context', 'generated', 'ground_truth'],
                data=wandb_table_data
            )

        wandb.log(wandb_data, step=global_step)

    trainer.model.train()

    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'num_samples': num_samples,
        'qualitative_samples': qualitative_samples,
        'qualitative_metrics': qualitative_metrics
    }


# ============================================================================
# Two-Stage Training Support
# ============================================================================

# ============================================================================
# Main Training Loop
# ============================================================================

def main(args):
    # Register graceful shutdown handlers (SIGTERM/SIGINT)
    _register_shutdown_handlers()

    # Handle resume: always use checkpoint's directory
    if args.resume_from_checkpoint:
        checkpoint_path = Path(args.resume_from_checkpoint)
        original_output_dir = str(checkpoint_path.parent)

        # Extract timestamp from directory name for W&B consistency
        # Directory format: outputs/production_{regime}_{params}_{objective}_{timestamp}
        dir_name = checkpoint_path.parent.name
        timestamp_match = re.search(r'_(\d{8}_\d{6})$', dir_name)
        if timestamp_match:
            # Use original timestamp for W&B name (ensures perfect match)
            args.timestamp = timestamp_match.group(1)
            print(f"Extracted original timestamp from checkpoint directory: {args.timestamp}")

        # Warn if user provided different output_dir (will be overridden)
        if args.output_dir is not None and args.output_dir != original_output_dir:
            print(f"Warning: Overriding --output_dir with checkpoint's directory")
            print(f"  Provided: {args.output_dir}")
            print(f"  Using: {original_output_dir}")

        args.output_dir = original_output_dir

    # Auto-generate output_dir if not provided
    if args.output_dir is None:
        auto_output_dir, auto_wandb_name = generate_from_args(args)
        args.output_dir = str(auto_output_dir)
        # Also set wandb_run_name for consistency (will be used later)
        if args.wandb_run_name is None:
            args.wandb_run_name = auto_wandb_name
        print(f"Auto-generated output directory: {args.output_dir}")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always print OUTPUT_DIR for bash script to capture (regardless of how it was set)
    print(f"OUTPUT_DIR={output_dir}")

    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting training with args: {args}")

    # Backward compatibility: map deprecated --train_projection to --train_encoder
    if hasattr(args, 'train_projection') and args.train_projection:
        logger.warning(
            "--train_projection is deprecated. Use --train_encoder instead. "
            "Automatically setting --train_encoder=True."
        )
        args.train_encoder = True

    # Sync train_projection with train_encoder for any legacy code that might reference it
    if hasattr(args, 'train_projection'):
        args.train_projection = args.train_encoder

    if args.resume_from_checkpoint:
        logger.info(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        logger.info(f"Continuing outputs in directory: {output_dir}")

    # Validate data_path requirement
    if not args.validation_only and not args.data_path:
        raise ValueError(
            "--data_path is required for training. "
            "It is only optional when using --validation_only mode."
        )

    if args.regime == 'text' and args.text_context_tokens is not None and args.text_context_tokens < 0:
        raise ValueError("--text_context_tokens must be >= 0 when using the text regime.")
    if args.regime != 'text' and args.text_context_tokens is not None:
        logger.warning("--text_context_tokens is ignored when regime != 'text'.")

    if args.regime in ['vision', 'meanpool', 'conv1d_residual'] and args.hybrid_text_tokens < 0:
        raise ValueError("--hybrid_text_tokens must be >= 0")
    if args.regime not in ['vision', 'meanpool', 'conv1d_residual'] and args.hybrid_text_tokens > 0:
        raise ValueError(
            f"--hybrid_text_tokens is only supported for vision, meanpool, and conv1d_residual regimes. "
            f"Got regime='{args.regime}' with hybrid_text_tokens={args.hybrid_text_tokens}"
        )

    # Validate meanpool compression parameters
    if args.regime == 'meanpool':
        if args.compression_window_size < 1:
            raise ValueError(f"--compression_window_size must be >= 1, got {args.compression_window_size}")
        if args.compression_stride < 1:
            raise ValueError(f"--compression_stride must be >= 1, got {args.compression_stride}")

    # Validate conv1d_residual compression parameters
    if args.regime == 'conv1d_residual':
        validate_conv1d_params(
            args.compression_target,
            args.conv_kernel,
            regime='conv1d_residual'
        )

    # Validate init_from_checkpoint (mutually exclusive with resume)
    if args.init_from_checkpoint:
        if args.resume_from_checkpoint:
            raise ValueError(
                "Cannot use both --init_from_checkpoint (two-stage training) and "
                "--resume_from_checkpoint (resume training). Choose one."
            )

        # Validate file exists
        init_ckpt_path = Path(args.init_from_checkpoint)
        if not init_ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {init_ckpt_path}\n"
                f"Check that the path is correct."
            )

        logger.info(f"Will initialize model from checkpoint: {args.init_from_checkpoint}")

    # Validate allow_objective_switch (requires init_from_checkpoint)
    if args.allow_objective_switch and not args.init_from_checkpoint:
        raise ValueError(
            "--allow_objective_switch requires --init_from_checkpoint.\n"
            "This flag enables objective switching for two-stage training (reconstruction → LM)."
        )

    # Validate objective compatibility
    if args.regime == 'text' and args.objective == 'reconstruction':
        raise ValueError(
            "Text regime does not support reconstruction objective. "
            "Use a compression regime (meanpool, conv1d_residual) for reconstruction."
        )

    # Resolve vision prompt (preset key or custom string)
    if args.vision_prompt:
        if args.vision_prompt in VISION_PROMPT_PRESETS:
            resolved_prompt = VISION_PROMPT_PRESETS[args.vision_prompt]
            logger.info(f"Using preset vision prompt: '{args.vision_prompt}' → '{repr(resolved_prompt)}'")
            args.vision_prompt = resolved_prompt
        else:
            # Custom string - warn if doesn't start with \n
            if not args.vision_prompt.startswith('\n'):
                logger.warning(
                    f"Vision prompt doesn't start with \\n: '{args.vision_prompt}'. "
                    f"DeepSeek-OCR expects prompts to start with newline. "
                    f"Consider using a preset or adding \\n prefix."
                )
            logger.info(f"Using custom vision prompt: '{repr(args.vision_prompt)}'")

    if args.regime != 'vision' and args.vision_prompt:
        logger.warning("--vision_prompt is ignored when regime != 'vision'.")

    # Normalize train_encoder for regimes where it's not applicable
    # meanpool: only has separator embedding (always trained), no compression module
    # Setting to False ensures wandb logs reflect reality and optimizer uses single param group
    if args.regime == 'meanpool':
        if args.train_encoder:
            logger.info(
                f"Setting train_encoder=False for {args.regime} regime "
                f"(no trainable compression module, only separator embedding which is always trained)."
            )
        args.train_encoder = False

    # Set random seeds for reproducibility (data ordering only, not model ops)
    if args.seed is not None:
        logger.info(f"Setting random seed: {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        # Note: We do NOT enable torch.use_deterministic_algorithms() to avoid slowdown
        # This gives reproducible data ordering while keeping fast nondeterministic CUDA ops

    # Save args
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Peek checkpoint metadata early (if resuming)
    # This allows us to get batch_idx BEFORE creating the dataloader (for sampler skip logic)
    checkpoint_metadata = None
    resume_batch_idx = None  # For mid-epoch resume
    if args.resume_from_checkpoint:
        checkpoint_metadata = peek_checkpoint_metadata(
            Path(args.resume_from_checkpoint),
            logger
        )

        # Extract batch_idx early for dataloader creation (must happen before sampler is instantiated)
        loaded_batch_idx = checkpoint_metadata['batch_idx']
        if loaded_batch_idx == -1:
            # End-of-epoch checkpoint: will start from next epoch
            resume_batch_idx = None
        else:
            # Mid-epoch checkpoint: will skip batches 0..loaded_batch_idx (inclusive)
            resume_batch_idx = loaded_batch_idx + 1

    # Initialize Weights & Biases (if enabled)
    wandb_run_id = None
    if args.use_wandb:
        if not WANDB_AVAILABLE:
            logger.warning("wandb is not installed. Install with: uv add wandb")
            logger.warning("Continuing without wandb logging...")
            args.use_wandb = False
        else:
            # Generate run name if not provided (use unified naming logic)
            if args.wandb_run_name is None:
                _, args.wandb_run_name = generate_from_args(args)
                logger.info(f"Auto-generated W&B run name: {args.wandb_run_name}")

            # Build wandb.init() kwargs
            init_kwargs = {
                'project': args.wandb_project,
                'config': vars(args),
            }

            # Always create fresh WandB run when resuming to avoid stale data conflicts
            if checkpoint_metadata and checkpoint_metadata.get('wandb_run_id'):
                logger.info(f"Checkpoint has WandB run ID: {checkpoint_metadata['wandb_run_id']}")
                logger.info("Creating fresh WandB run (not resuming to avoid stale data)")

            # Let WandB create new run (don't set 'id' or force resume)
            init_kwargs['name'] = args.wandb_run_name
            init_kwargs['resume'] = 'allow'  # Allow resume if run exists

            # Initialize wandb
            wandb.init(**init_kwargs)
            wandb_run_id = wandb.run.id
            logger.info(f"Initialized W&B run: {args.wandb_project}/{wandb.run.name} (ID: {wandb_run_id})")

    # Load model and tokenizer
    logger.info("Loading model and tokenizer...")
    # Only load vision encoder for vision regime (saves ~1.5-2GB GPU memory for text regimes)
    load_vision_encoder = (args.regime == 'vision')
    model, tokenizer = load_model_and_tokenizer(
        device=args.device,
        use_optimized_model=args.use_optimized_model,
        use_encoder_checkpointing=args.use_encoder_checkpointing,
        load_vision_encoder=load_vision_encoder
    )

    # Enable decoder gradient checkpointing if requested
    if args.use_decoder_checkpointing:
        logger.info("Enabling decoder gradient checkpointing...")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        num_layers = len(model.model.layers)
        logger.info(f"  ✓ Decoder checkpointing enabled for {num_layers} transformer layers")
        logger.info(f"  Expected: ~30-50% activation memory reduction, ~15-20% compute overhead")

    # Compile model if requested
    if args.compile:
        logger.info(f"Compiling model with torch.compile (mode={args.compile_mode})...")
        if args.compile_mode == 'max-autotune':
            logger.info("  ⚠️  max-autotune mode: First compilation will take 10-30 minutes")
        else:
            logger.info("  Note: First forward pass will compile (may take several minutes)")
        model = torch.compile(model, mode=args.compile_mode)

    # Set global BOS token ID from tokenizer
    global BOS_TOKEN_ID
    BOS_TOKEN_ID = tokenizer.bos_token_id

    # Create trainer based on regime
    if args.regime == 'vision':
        trainer = VisionCompressionTrainer(
            model, tokenizer,
            vision_mode=args.vision_mode,
            device=args.device,
            hybrid_text_tokens=args.hybrid_text_tokens,
            vision_prompt=args.vision_prompt,
            train_encoder=args.train_encoder
        )
        logger.info(f"Created Vision Compression trainer (mode: {args.vision_mode})")
        if args.hybrid_text_tokens > 0:
            logger.info(f"  Hybrid mode enabled: {args.hybrid_text_tokens} additional text tokens")

    elif args.regime == 'text':
        trainer = TextBaselineTrainer(model, tokenizer, device=args.device)
        logger.info("Created Text Baseline trainer")

    elif args.regime == 'meanpool':
        trainer = MeanPoolCompressionTrainer(
            model, tokenizer,
            window_size=args.compression_window_size,
            stride=args.compression_stride,
            device=args.device,
            hybrid_text_tokens=args.hybrid_text_tokens
        )
        logger.info(f"Created Mean Pool Compression trainer")
        logger.info(f"  Compression: 1000 → {trainer.compressed_tokens} tokens")

    elif args.regime == 'conv1d_residual':
        trainer = Conv1dResidualCompressionTrainer(
            model, tokenizer,
            compression_target=args.compression_target,
            conv_kernel=args.conv_kernel,
            device=args.device,
            hybrid_text_tokens=args.hybrid_text_tokens,
            train_encoder=args.train_encoder
        )
        logger.info("Created Conv1D Residual Pyramid Compression trainer")
        logger.info(f"  Architecture: Residual blocks with skip connections")
        logger.info(f"  Kernel size: {args.conv_kernel}")
        logger.info(f"  Compression: 1000 → {trainer.compressed_tokens} tokens ({1000/args.compression_target:.2f}x)")

    logger.info(f"Training objective: {args.objective}")

    # Universal checkpoint loading for two-stage training (all regimes)
    # This replaces the old regime-specific init_from_checkpoint handling
    stage1_metadata = None
    if args.init_from_checkpoint:
        logger.info(f"\n{'='*80}")
        logger.info("TWO-STAGE TRAINING: Loading Stage 1 checkpoint")
        logger.info(f"{'='*80}")

        # STEP 1: Peek at checkpoint metadata (don't load weights yet)
        stage1_metadata = peek_checkpoint_metadata(
            Path(args.init_from_checkpoint),
            logger=logger
        )

        # STEP 2: Validate architecture compatibility BEFORE loading weights
        # This ensures users get helpful error messages instead of cryptic PyTorch errors
        validate_and_load_init_checkpoint(stage1_metadata, args, logger)

        # STEP 3: Now load model weights (architecture is validated, loading will succeed)
        load_checkpoint(
            Path(args.init_from_checkpoint),
            model=trainer.model,
            optimizer=None,
            scheduler=None,
            device=torch.device(args.device),
            logger=logger,
            args=args,
            weights_only=True
        )

        # Log transition details
        logger.info(f"\nStage 1 → Stage 2 Transition:")
        logger.info(f"  Stage 1 checkpoint: {args.init_from_checkpoint}")
        logger.info(f"  Stage 1 regime: {stage1_metadata['regime']}")
        logger.info(f"  Stage 1 objective: {stage1_metadata['objective']}")
        logger.info(f"  Stage 1 epoch: {stage1_metadata['epoch']}")
        logger.info(f"  Stage 1 best_val_loss: {stage1_metadata.get('best_val_loss', 'N/A')}")
        if stage1_metadata.get('wandb_run_id'):
            logger.info(f"  Stage 1 W&B run: {stage1_metadata['wandb_run_id']}")

        logger.info(f"\n  Stage 2 regime: {args.regime} ✓ MATCH")
        logger.info(f"  Stage 2 objective: {args.objective}" +
                   (f" (CHANGED from {stage1_metadata['objective']})" if stage1_metadata['objective'] != args.objective else " (SAME)"))

        logger.info(f"\n✓ Successfully loaded model weights from Stage 1")
        logger.info(f"✓ Fresh optimizer will be created for Stage 2")
        logger.info(f"✓ New W&B run will track Stage 2")
        logger.info(f"{'='*80}\n")

    # Log parameter counts to W&B
    # Skip when resuming same W&B run - config was already set in original run
    if args.use_wandb and WANDB_AVAILABLE and not wandb.run.resumed:
        params = count_parameters(trainer.model)

        # Get encoder/decoder breakdown
        encoder_params_list = get_encoder_params(trainer.model)
        decoder_params_list = get_decoder_params(trainer.model)
        encoder_param_count = sum(p.numel() for p in encoder_params_list)
        decoder_param_count = sum(p.numel() for p in decoder_params_list)

        config_update = {
            'total_params': params['total'],
            'trainable_params': params['trainable'],
            'frozen_params': params['frozen'],
            'trainable_params_pct': params['trainable_pct'],
            'encoder_params': encoder_param_count,
            'decoder_params': decoder_param_count,
            'compile': args.compile
        }

        # Add Stage 1 metadata if doing two-stage training
        if stage1_metadata:
            config_update['two_stage_training'] = True
            config_update['stage1_checkpoint'] = stage1_metadata['checkpoint_path']
            config_update['stage1_regime'] = stage1_metadata['regime']
            config_update['stage1_objective'] = stage1_metadata['objective']
            config_update['stage1_epoch'] = stage1_metadata['epoch']
            config_update['stage1_best_val_loss'] = stage1_metadata.get('best_val_loss')
            config_update['stage1_wandb_run_id'] = stage1_metadata.get('wandb_run_id')

        wandb.config.update(config_update)

        logger.info(f"Logged parameter counts to W&B: "
                    f"total={params['total']:,}, trainable={params['trainable']:,}, "
                    f"encoder={encoder_param_count:,}, decoder={decoder_param_count:,}")

        if stage1_metadata:
            logger.info(f"Logged Stage 1 metadata to W&B config for tracking")

    # Create datasets
    # Skip training data in validation-only mode (not needed, speeds up startup)
    if args.validation_only:
        logger.info("Validation-only mode: skipping training dataset (not needed)")
        train_dataset = None
        train_dataloader = None
        collate_fn = None  # Will be set when creating validation dataset
    else:
        logger.info(f"Loading training data from {args.data_path}")
        if args.regime == 'vision':
            train_dataset = create_vision_dataset(
                args.data_path,
                tokenizer,
                vision_mode=args.vision_mode,
                max_samples=args.max_samples,
                hybrid_text_tokens=args.hybrid_text_tokens
            )
            collate_fn = vision_collate_fn
        else:
            # Text and compression regimes use TextBaselineDataset with compression_collate_fn
            # (keeps context/continuation separate for objective-based selection)
            # Determine if hybrid mode applies to current regime
            hybrid_tokens = args.hybrid_text_tokens if args.regime in ['meanpool', 'conv1d_residual'] else 0

            train_dataset = create_text_dataset(
                args.data_path,
                tokenizer,
                max_samples=args.max_samples,
                context_length=args.text_context_tokens if args.regime == 'text' else None,
                hybrid_text_tokens=hybrid_tokens
            )
            collate_fn = compression_collate_fn

            if args.regime == 'text':
                kept_tokens = 'full' if train_dataset.context_length is None else train_dataset.context_length
                logger.info(f"Text baseline context tokens per sample: {kept_tokens}")
            else:
                logger.info(f"{args.regime.capitalize()} regime: using full 1000-token context")

        # Create dataloader
        # Note: Images are pre-rendered and loaded from disk, so multiprocessing is safe

        # Get sampler state from checkpoint metadata (if resuming)
        saved_sampler_state = checkpoint_metadata.get('sampler_state', None) if checkpoint_metadata else None

        # Calculate samples to skip for mid-epoch resume (optimization to avoid loading batches we don't need)
        skip_samples = 0
        if resume_batch_idx is not None and resume_batch_idx > 0:
            skip_samples = resume_batch_idx * args.batch_size
            logger.info(f"Mid-epoch resume: skipping first {skip_samples} samples at sampler level (batch {resume_batch_idx})")

        # Create stateful sampler for mid-epoch resume support
        global _train_sampler
        _train_sampler = StatefulRandomSampler(train_dataset, permutation=saved_sampler_state, skip_first=skip_samples)
        train_sampler = _train_sampler

        # Validate mid-epoch resume from legacy checkpoints
        if resume_batch_idx is not None and resume_batch_idx > 0 and saved_sampler_state is None:
            logger.error(
                f"\n{'='*70}\n"
                f"CRITICAL: Mid-epoch resume from legacy checkpoint without sampler state!\n"
                f"  Checkpoint batch_idx: {resume_batch_idx}\n"
                f"  Sampler state: MISSING\n"
                f"\n"
                f"Cannot guarantee correct data ordering for mid-epoch resume.\n"
                f"Samples may be duplicated or skipped, invalidating results.\n"
                f"\n"
                f"Solutions:\n"
                f"  1. Resume from end-of-epoch checkpoint (batch_idx=-1) instead\n"
                f"  2. Start fresh training run with this checkpoint as init\n"
                f"  3. Accept potentially corrupted sample order (NOT RECOMMENDED)\n"
                f"{'='*70}\n"
            )
            raise ValueError(
                f"Cannot resume mid-epoch (batch {resume_batch_idx}) from legacy checkpoint "
                f"without sampler_state. Use end-of-epoch checkpoint or start fresh."
            )

        # Build DataLoader kwargs
        dataloader_kwargs = {
            'batch_size': args.batch_size,
            'sampler': train_sampler,  # Use custom sampler instead of shuffle=True
            'collate_fn': collate_fn,
            'num_workers': args.num_workers,
            'pin_memory': True if args.device.startswith('cuda') else False,
        }

        # Add reproducibility options if seed is set
        if args.seed is not None:
            # Worker seeding for reproducible augmentation
            dataloader_kwargs['worker_init_fn'] = seed_worker

        # Add multiprocessing options only if num_workers > 0
        if args.num_workers > 0:
            dataloader_kwargs['prefetch_factor'] = args.prefetch_factor  # Increase prefetch buffer for better GPU utilization
            dataloader_kwargs['persistent_workers'] = True  # Keep workers alive between epochs

        train_dataloader = DataLoader(train_dataset, **dataloader_kwargs)

    # Validation dataloader (if provided)
    val_dataloader = None
    if args.val_data_path:
        logger.info(f"Loading validation data from {args.val_data_path}")
        # Note: Don't apply max_samples to validation - always use full validation set
        if args.regime == 'vision':
            val_dataset = create_vision_dataset(
                args.val_data_path,
                tokenizer,
                vision_mode=args.vision_mode,
                max_samples=None,
                hybrid_text_tokens=args.hybrid_text_tokens
            )
            # Set collate_fn if not already set (happens in validation-only mode)
            if collate_fn is None:
                collate_fn = vision_collate_fn
        elif args.regime == 'text':
            val_dataset = create_text_dataset(
                args.val_data_path,
                tokenizer,
                max_samples=None,
                context_length=args.text_context_tokens,
                hybrid_text_tokens=0  # Text regime doesn't use hybrid mode
            )
            val_kept_tokens = 'full' if val_dataset.context_length is None else val_dataset.context_length
            logger.info(f"Validation text context tokens per sample: {val_kept_tokens}")
            # Set collate_fn if not already set (happens in validation-only mode)
            if collate_fn is None:
                collate_fn = compression_collate_fn
        else:
            # Meanpool, conv1d_residual, and other compression regimes
            # Determine if hybrid mode applies to current regime
            val_hybrid_tokens = args.hybrid_text_tokens if args.regime in ['meanpool', 'conv1d_residual'] else 0

            val_dataset = create_text_dataset(
                args.val_data_path,
                tokenizer,
                max_samples=None,
                context_length=None,  # Full context
                hybrid_text_tokens=val_hybrid_tokens
            )
            logger.info(f"Validation {args.regime} regime: using full 1000-token context")
            # Set collate_fn if not already set (happens in validation-only mode)
            if collate_fn is None:
                collate_fn = compression_collate_fn

        val_dataloader_kwargs = {
            'batch_size': args.batch_size,
            'shuffle': False,
            'collate_fn': collate_fn,
            'num_workers': args.num_workers,
            'pin_memory': True if args.device.startswith('cuda') else False
        }

        # Add worker_init_fn for reproducibility (no generator needed since shuffle=False)
        if args.seed is not None:
            val_dataloader_kwargs['worker_init_fn'] = seed_worker

        val_dataloader = DataLoader(val_dataset, **val_dataloader_kwargs)

    # Validation-only mode: Run single validation pass and exit
    if args.validation_only:
        if args.resume_from_checkpoint:
            raise ValueError(
                "--validation_only is incompatible with --resume_from_checkpoint. "
                "Use --resume_from_checkpoint for continuing training, or --validation_only "
                "for a fresh validation run without any training."
            )

        if not val_dataloader:
            raise ValueError(
                "--validation_only requires --val_data_path to be specified. "
                "Cannot run validation without validation data."
            )

        logger.info(f"\n{'='*70}")
        logger.info("VALIDATION-ONLY MODE")
        logger.info(f"{'='*70}")
        logger.info("Running single validation pass (no training)")
        logger.info(f"Dataset: {args.val_data_path}")
        logger.info(f"Regime: {args.regime}")
        logger.info(f"Objective: {args.objective}")
        logger.info(f"{'='*70}\n")

        # Run validation
        val_metrics = evaluate(
            trainer, val_dataloader, args.regime, logger, args, global_step=0,
            num_qualitative_samples=args.num_qualitative_samples,
            max_generation_tokens=args.max_generation_tokens,
            output_dir=output_dir,
            eval_seed=args.eval_seed
        )

        logger.info(f"\n{'='*70}")
        logger.info("VALIDATION RESULTS")
        logger.info(f"{'='*70}")
        logger.info(f"Loss: {val_metrics['loss']:.4f}")
        logger.info(f"Perplexity: {val_metrics['perplexity']:.2f}")
        logger.info(f"{'='*70}\n")

        logger.info("Validation-only mode complete. Exiting.")

        # Finalize wandb run before exiting
        if args.use_wandb:
            wandb.finish()
            logger.info("W&B run finished")

        return  # Exit early without training

    # Create optimizer with differential learning rates if training encoder
    if args.train_encoder:
        encoder_params = get_encoder_params(model)
        decoder_params = get_decoder_params(model)

        # Validate parameter partitioning (critical for catching configuration errors)
        if not encoder_params:
            raise ValueError(
                "No trainable encoder parameters found! "
                "Check that --train_encoder matches the model architecture and that "
                "encoder components (SAM, ViT, Projector) have requires_grad=True."
            )

        if not decoder_params:
            raise ValueError(
                "No trainable decoder parameters found! "
                "Check model initialization and parameter freezing logic."
            )

        # Validate no parameter overlap between encoder and decoder groups
        encoder_ids = {id(p) for p in encoder_params}
        decoder_ids = {id(p) for p in decoder_params}
        overlap = encoder_ids & decoder_ids

        if overlap:
            raise ValueError(
                f"Found {len(overlap)} parameters in both encoder and decoder groups! "
                f"This would cause parameters to be updated with conflicting learning rates. "
                f"Check get_encoder_params() and get_decoder_params() logic in trainers/utils/freezing.py"
            )

        # Validate all trainable params are covered (no params left out)
        all_trainable = {id(p) for p in model.parameters() if p.requires_grad}
        covered = encoder_ids | decoder_ids
        missing = all_trainable - covered

        if missing:
            logger.warning(
                f"Found {len(missing)} trainable parameters not assigned to encoder or decoder groups. "
                f"These params will not be optimized. Total trainable: {len(all_trainable)}, "
                f"Encoder: {len(encoder_ids)}, Decoder: {len(decoder_ids)}"
            )

        # Cache param lists on model for efficient gradient norm computation during training
        # Avoids iterating all parameters on every training step
        model._cached_encoder_params = encoder_params
        model._cached_decoder_params = decoder_params

        # Create optimizer with optional 8-bit quantization
        if args.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit([
                    {'params': encoder_params, 'lr': args.encoder_lr, 'name': 'encoder'},
                    {'params': decoder_params, 'lr': args.learning_rate, 'name': 'decoder'}
                ], weight_decay=args.weight_decay)
                optimizer_type = 'adamw_8bit'
                logger.info(
                    f"Created 8-bit AdamW optimizer (bitsandbytes) with differential LR:\n"
                    f"  Encoder: {len(encoder_params)} param tensors @ lr={args.encoder_lr}\n"
                    f"  Decoder: {len(decoder_params)} param tensors @ lr={args.learning_rate}\n"
                    f"  Memory savings: ~75% optimizer state (16.8GB for 2.8B params)\n"
                    f"  Expected overhead: ~2-5%"
                )
            except Exception as e:
                logger.error(
                    f"8-bit optimizer unavailable: {e}\n"
                    f"Falling back to standard AdamW optimizer."
                )
                use_fused = args.device.startswith('cuda') and torch.cuda.is_available()
                optimizer = AdamW([
                    {'params': encoder_params, 'lr': args.encoder_lr, 'name': 'encoder'},
                    {'params': decoder_params, 'lr': args.learning_rate, 'name': 'decoder'}
                ], weight_decay=args.weight_decay, fused=use_fused)
                optimizer_type = 'adamw_fused' if use_fused else 'adamw_standard'
                logger.info(f"Fallback: Created standard AdamW optimizer (fused={use_fused})")
        else:
            # Use fused AdamW if CUDA is available (3-5% speedup)
            use_fused = args.device.startswith('cuda') and torch.cuda.is_available()
            optimizer = AdamW([
                {'params': encoder_params, 'lr': args.encoder_lr, 'name': 'encoder'},
                {'params': decoder_params, 'lr': args.learning_rate, 'name': 'decoder'}
            ], weight_decay=args.weight_decay, fused=use_fused)
            optimizer_type = 'adamw_fused' if use_fused else 'adamw_standard'
            logger.info(
                f"Created AdamW optimizer with differential LR:\n"
                f"  Encoder: {len(encoder_params)} param tensors @ lr={args.encoder_lr}\n"
                f"  Decoder: {len(decoder_params)} param tensors @ lr={args.learning_rate}\n"
                f"  Fused kernels: {use_fused}"
            )
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        # Create optimizer with optional 8-bit quantization
        if args.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(
                    trainable_params,
                    lr=args.learning_rate,
                    weight_decay=args.weight_decay
                )
                optimizer_type = 'adamw_8bit'
                logger.info(
                    f"Created 8-bit AdamW optimizer (bitsandbytes):\n"
                    f"  Learning rate: {args.learning_rate}\n"
                    f"  Memory savings: ~75% optimizer state (16.8GB for 2.8B params)\n"
                    f"  Expected overhead: ~2-5%"
                )
            except Exception as e:
                logger.error(
                    f"8-bit optimizer unavailable: {e}\n"
                    f"Falling back to standard AdamW optimizer."
                )
                use_fused = args.device.startswith('cuda') and torch.cuda.is_available()
                optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay, fused=use_fused)
                optimizer_type = 'adamw_fused' if use_fused else 'adamw_standard'
                logger.info(f"Fallback: Created standard AdamW optimizer (lr={args.learning_rate}, fused={use_fused})")
        else:
            # Use fused AdamW if CUDA is available (3-5% speedup)
            use_fused = args.device.startswith('cuda') and torch.cuda.is_available()
            optimizer = AdamW(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay, fused=use_fused)
            optimizer_type = 'adamw_fused' if use_fused else 'adamw_standard'
            logger.info(f"Created AdamW optimizer with lr={args.learning_rate}, fused={use_fused}")

    # Create scheduler with sanity checks
    num_training_steps = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps) * args.num_epochs
    num_warmup_steps = int(num_training_steps * args.warmup_ratio)

    # Sanity check: Ensure we have enough training steps
    if num_training_steps <= 0:
        raise ValueError(
            f"Dataset too small for gradient accumulation settings!\n"
            f"  Dataset size: {len(train_dataloader)}\n"
            f"  Gradient accumulation steps: {args.gradient_accumulation_steps}\n"
            f"  Effective steps per epoch: {num_training_steps // args.num_epochs}\n"
            f"  Suggestion: Reduce --gradient_accumulation_steps or increase dataset size"
        )

    # Guard warmup: Ensure warmup doesn't exceed total steps
    if num_warmup_steps >= num_training_steps:
        logger.warning(
            f"Warmup steps ({num_warmup_steps}) >= total training steps ({num_training_steps}). "
            f"Disabling warmup."
        )
        num_warmup_steps = 0

    # Create scheduler
    if num_warmup_steps > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=num_warmup_steps
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - num_warmup_steps
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[num_warmup_steps]
        )
        logger.info(f"Created scheduler with warmup_steps={num_warmup_steps}, total_steps={num_training_steps}")
    else:
        # No warmup, just cosine annealing
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps
        )
        logger.info(f"Created scheduler without warmup, total_steps={num_training_steps}")

    # Log optimizer configuration to W&B
    if args.use_wandb:
        # Calculate estimated optimizer memory
        trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if optimizer_type == 'adamw_8bit':
            # 8-bit: 2 bytes per param (momentum + variance in INT8)
            optimizer_memory_gb = (trainable_param_count * 2) / (1024**3)
        elif optimizer_type in ['adamw_fused', 'adamw_standard']:
            # Standard: 8 bytes per param (momentum + variance in FP32)
            optimizer_memory_gb = (trainable_param_count * 8) / (1024**3)
        else:
            optimizer_memory_gb = None

        optimizer_config = {
            'optimizer_type': optimizer_type,
            'optimizer_8bit': (optimizer_type == 'adamw_8bit'),  # Actual state, not user intent
        }
        if optimizer_memory_gb is not None:
            optimizer_config['optimizer_memory_gb'] = optimizer_memory_gb

        wandb.config.update(optimizer_config)
        logger.info(f"Logged optimizer config to W&B: type={optimizer_type}, "
                   f"memory={optimizer_memory_gb:.2f}GB" if optimizer_memory_gb else f"type={optimizer_type}")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    global_step = 0

    if args.resume_from_checkpoint:
        # Load model/optimizer state from checkpoint
        # (metadata was already loaded earlier via peek_checkpoint_metadata)
        device = torch.device(args.device)
        _, _, _, _, _ = load_checkpoint(
            Path(args.resume_from_checkpoint),
            model,
            optimizer,
            scheduler,
            device,
            logger,
            args  # Pass args to check for encoder_trainable mismatch
        )

        # Use metadata that was peeked earlier
        best_val_loss = checkpoint_metadata['best_val_loss']
        global_step = checkpoint_metadata['global_step']
        loaded_epoch = checkpoint_metadata['epoch']
        loaded_batch_idx = checkpoint_metadata['batch_idx']

        # Note: We always create fresh WandB runs when resuming (never reattach to old runs)
        # So we don't preserve the old wandb_run_id - it stays None if WandB is disabled
        # This ensures checkpoint_log.txt accurately reflects which runs created which checkpoints

        # wandb_run_id was set during wandb.init() earlier (if using wandb)
        logger.info(
            f"Restored training state: epoch={loaded_epoch}, "
            f"batch_idx={loaded_batch_idx}, global_step={global_step}, "
            f"best_val_loss={best_val_loss:.4f}"
        )

        # Log resume metadata to wandb config (crash-recovery only, not two-stage training)
        if args.use_wandb:
            wandb.config.update({
                'resume_from_epoch': loaded_epoch,
                'resume_from_step': global_step,
            }, allow_val_change=True)

        # Determine resume strategy based on batch_idx
        # Note: resume_batch_idx was already calculated early (before dataloader creation)
        if loaded_batch_idx == -1:
            # End-of-epoch checkpoint: start from next epoch
            start_epoch = loaded_epoch + 1
            logger.info(f"Resuming from next epoch (epoch {start_epoch})")
        else:
            # Mid-epoch checkpoint: resume from same epoch, skip processed batches

            # Validate that batch_size hasn't changed (critical for mid-epoch resume)
            checkpoint_batch_size = checkpoint_metadata.get('batch_size')
            if checkpoint_batch_size is not None and checkpoint_batch_size != args.batch_size:
                raise ValueError(
                    f"Batch size mismatch: checkpoint used batch_size={checkpoint_batch_size}, "
                    f"but current config has batch_size={args.batch_size}. "
                    f"Changing batch_size during resume is not supported for mid-epoch checkpoints."
                )

            start_epoch = loaded_epoch
            logger.info(f"Resuming mid-epoch: will skip first {resume_batch_idx} batches of epoch {start_epoch}")

    # Training loop
    logger.info("Starting training loop...")

    # Optional: Initial validation before any gradient updates (establish baseline)
    if args.initial_validation and val_dataloader:
        logger.info(f"\n{'='*70}")
        logger.info("Running initial validation (before any training)...")
        logger.info(f"{'='*70}")
        initial_val_metrics = evaluate(
            trainer, val_dataloader, args.regime, logger, args, global_step=0,
            num_qualitative_samples=args.num_qualitative_samples,
            max_generation_tokens=args.max_generation_tokens,
            output_dir=output_dir,
            eval_seed=args.eval_seed
        )
        logger.info(f"Initial validation - Loss: {initial_val_metrics['loss']:.4f}, "
                   f"Perplexity: {initial_val_metrics['perplexity']:.2f}")
        logger.info(f"{'='*70}\n")

        # Update best_val_loss if this initial validation is better
        # (unlikely, but possible if resuming from checkpoint)
        if initial_val_metrics['loss'] < best_val_loss:
            best_val_loss = initial_val_metrics['loss']

        # Clean up GPU memory after validation to avoid fragmentation before training
        # Validation allocates different tensor sizes than training, which can cause
        # fragmentation. Releasing cached memory allows fresh contiguous allocation.
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Cleared GPU memory cache after initial validation")

    # Initialize train_metrics in case loop is skipped (e.g., resume at final epoch)
    train_metrics = {
        'loss': 0.0,
        'grad_norm': 0.0,
        'num_steps': 0,
        'global_step': global_step,
        'best_val_loss': best_val_loss,
        'perplexity': 0.0,
        'epoch_tokens_per_sec': 0.0,
        'epoch_duration': 0.0
    }

    # NOTE: StatefulRandomSampler currently reuses the same permutation across
    # all epochs. For single-epoch training this is fine, but for multi-epoch
    # training you should implement per-epoch reshuffling. See StatefulRandomSampler
    # docstring for details.
    if args.num_epochs > 1:
        logger.warning(
            f"Training for {args.num_epochs} epochs with StatefulRandomSampler. "
            f"Note: All epochs will use the same shuffle order (no per-epoch reshuffling). "
            f"This may impact training quality for multi-epoch runs."
        )

    for epoch in range(start_epoch, args.num_epochs):
        logger.info(f"\n{'='*70}")
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
        logger.info(f"{'='*70}")

        # Calculate initial skip offset for mid-epoch resume (only applies to first epoch)
        initial_skip_batches = resume_batch_idx if (resume_batch_idx is not None and epoch == start_epoch) else 0

        # Train (with step-based validation)
        train_metrics = train_epoch(
            trainer,
            train_dataloader,
            val_dataloader,  # Pass validation dataloader for step-based eval
            optimizer,
            scheduler,
            args.regime,
            epoch,
            args,
            logger,
            global_step,
            best_val_loss,
            output_dir,
            wandb_run_id=wandb_run_id,
            initial_skip_batches=initial_skip_batches
        )

        # Update global step and best validation loss
        global_step = train_metrics['global_step']
        best_val_loss = train_metrics['best_val_loss']

        logger.info(
            f"Epoch {epoch+1} training: "
            f"loss={train_metrics['loss']:.4f}, "
            f"ppl={train_metrics['perplexity']:.2f}, "
            f"grad_norm={train_metrics['grad_norm']:.2f}, "
            f"throughput={train_metrics['epoch_tokens_per_sec']:.0f} tok/s "
            f"({train_metrics['epoch_duration']:.1f}s total)"
        )

        # Log epoch-level metrics to W&B
        if args.use_wandb:
            wandb.log({
                'train/epoch_loss': train_metrics['loss'],
                'train/epoch_perplexity': train_metrics['perplexity'],
                'train/epoch_tokens_per_sec': train_metrics['epoch_tokens_per_sec'],
                'train/epoch_duration_sec': train_metrics['epoch_duration'],
                'train/final_epoch': epoch + 1
            }, step=global_step)

    # Compute sampler state for final checkpoint saves
    # (must be computed BEFORE val_dataloader check to avoid NameError when training without validation)
    sampler_state = _train_sampler.get_state() if _train_sampler else None

    # Final validation (always run at end of training if validation set provided)
    final_val_loss = None
    final_val_perplexity = None
    final_is_best = False  # Initialize before validation block to avoid UnboundLocalError
    if val_dataloader:
        logger.info(f"\nRunning final validation...")
        val_metrics = evaluate(
            trainer, val_dataloader, args.regime, logger, args, global_step,
            num_qualitative_samples=args.num_qualitative_samples,
            max_generation_tokens=args.max_generation_tokens,
            output_dir=output_dir,
            eval_seed=args.eval_seed
        )
        final_val_loss = val_metrics['loss']
        final_val_perplexity = val_metrics['perplexity']

        # Save best model if final validation is best
        if final_val_loss <= best_val_loss:  # Use <= to handle ties
            best_val_loss = final_val_loss
            final_is_best = True
            if not args.no_checkpoints:
                best_checkpoint_path = output_dir / 'best_checkpoint.pt'
                save_checkpoint(
                    model, optimizer, scheduler,
                    args.num_epochs - 1, -1, global_step,  # batch_idx=-1 indicates end of epoch
                    train_metrics['loss'], final_val_loss,
                    best_val_loss,
                    args, best_checkpoint_path, logger,
                    train_perplexity=train_metrics['perplexity'],
                    val_perplexity=final_val_perplexity,
                    wandb_run_id=wandb_run_id,
                    sampler_state=sampler_state
                )
                logger.info(f"New best validation loss: {best_val_loss:.4f}, perplexity: {final_val_perplexity:.2f}")
            else:
                logger.info(f"New best validation loss: {best_val_loss:.4f}, perplexity: {final_val_perplexity:.2f} (checkpoint saving disabled)")

            # Log best model update to wandb
            if args.use_wandb:
                wandb.log({
                    'val/best_loss': best_val_loss,
                    'val/best_perplexity': final_val_perplexity
                }, step=global_step)

    # Save final checkpoint
    logger.info(f"\nTraining complete!")
    if not args.no_checkpoints:
        final_checkpoint_path = output_dir / 'final_checkpoint.pt'

        if final_is_best:
            # Final validation is best: try to create symlink to save space
            best_checkpoint_path = output_dir / 'best_checkpoint.pt'

            # Only create symlink if best checkpoint exists
            if best_checkpoint_path.exists():
                try:
                    # Clean up existing file/symlink
                    if final_checkpoint_path.exists() or final_checkpoint_path.is_symlink():
                        final_checkpoint_path.unlink()
                    os.symlink('best_checkpoint.pt', final_checkpoint_path)
                    logger.info(f"Final checkpoint is best, created symlink to save space (~2GB saved)")
                except OSError as e:
                    # Fallback: save separate checkpoint if symlink fails (e.g., unsupported filesystem)
                    logger.warning(f"Failed to create symlink ({e}). Saving separate final checkpoint.")
                    save_checkpoint(
                        model, optimizer, scheduler,
                        args.num_epochs - 1, -1, global_step,
                        train_metrics['loss'],
                        final_val_loss,
                        best_val_loss,
                        args, final_checkpoint_path, logger,
                        train_perplexity=train_metrics['perplexity'],
                        val_perplexity=final_val_perplexity,
                        wandb_run_id=wandb_run_id,
                        sampler_state=sampler_state
                    )
                    logger.info(f"Final checkpoint saved to {final_checkpoint_path}")
            else:
                # Fallback: best checkpoint doesn't exist for some reason
                # Save as BOTH best and final to honor the contract that both files exist
                logger.warning("Best checkpoint not found. Saving as both best and final checkpoints.")

                # Save as best checkpoint (since final is best)
                save_checkpoint(
                    model, optimizer, scheduler,
                    args.num_epochs - 1, -1, global_step,
                    train_metrics['loss'],
                    final_val_loss,
                    best_val_loss,
                    args, best_checkpoint_path, logger,
                    train_perplexity=train_metrics['perplexity'],
                    val_perplexity=final_val_perplexity,
                    wandb_run_id=wandb_run_id,
                    sampler_state=sampler_state
                )

                # Also save as final checkpoint
                save_checkpoint(
                    model, optimizer, scheduler,
                    args.num_epochs - 1, -1, global_step,
                    train_metrics['loss'],
                    final_val_loss,
                    best_val_loss,
                    args, final_checkpoint_path, logger,
                    train_perplexity=train_metrics['perplexity'],
                    val_perplexity=final_val_perplexity,
                    wandb_run_id=wandb_run_id,
                    sampler_state=sampler_state
                )
                logger.info(f"Saved both best and final checkpoints")
        else:
            # Final validation is not best: save separate checkpoint
            save_checkpoint(
                model, optimizer, scheduler,
                args.num_epochs - 1, -1, global_step,  # batch_idx=-1 indicates end of epoch/training
                train_metrics['loss'],
                final_val_loss,
                best_val_loss,
                args, final_checkpoint_path, logger,
                train_perplexity=train_metrics['perplexity'],
                val_perplexity=final_val_perplexity,
                wandb_run_id=wandb_run_id,
                sampler_state=sampler_state
            )
            logger.info(f"Final checkpoint saved to {final_checkpoint_path}")
    else:
        logger.info("(Checkpoint saving disabled)")

    if val_dataloader:
        best_val_perplexity = compute_perplexity(best_val_loss)
        logger.info(f"Best validation loss: {best_val_loss:.4f}, perplexity: {best_val_perplexity:.2f}")
    if not args.no_checkpoints:
        logger.info(f"Checkpoints saved to {output_dir}")

    # Finalize wandb run
    if args.use_wandb:
        wandb.finish()
        logger.info("W&B run finished")


# ============================================================================
# Argument Parsing with Auto-Resume Support
# ============================================================================

def parse_args():
    parser = create_argument_parser()
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    # Parse args first to get values regardless of syntax (space or equals)
    # This handles both "--resume path" and "--resume=path" correctly
    cli_args = parse_args()

    # Check if either resume flag was provided
    resume_checkpoint = None
    if cli_args.resume:
        resume_checkpoint = cli_args.resume
    elif cli_args.resume_from_checkpoint:
        resume_checkpoint = cli_args.resume_from_checkpoint

    if resume_checkpoint:
        # Auto-resume mode: load checkpoint args and merge with CLI
        checkpoint_path = Path(resume_checkpoint)
        checkpoint_args = load_checkpoint_args(checkpoint_path)
        args = merge_args(checkpoint_args, cli_args)
        # Set canonical field so downstream code sees resume
        args.resume_from_checkpoint = resume_checkpoint
        print("\nUsing configuration from checkpoint with CLI overrides applied.\n")
    else:
        # Normal mode: use CLI args only
        args = cli_args

    main(args)
