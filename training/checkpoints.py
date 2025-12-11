"""Checkpoint management utilities for training resumption and two-stage training."""

import os
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Checkpoint format version for tracking evolution
CHECKPOINT_FORMAT_VERSION = "2.0"  # Post-refactor version (added after checkpoint management extraction)


def _validate_checkpoint_version(checkpoint: dict, checkpoint_path: Path, logger=None):
    """
    Validate checkpoint format version compatibility.

    Args:
        checkpoint: Loaded checkpoint dictionary
        checkpoint_path: Path to checkpoint file (for error messages)
        logger: Optional logger for warnings

    Raises:
        ValueError: If checkpoint version is incompatible
    """
    ckpt_version = checkpoint.get('format_version')

    # Legacy checkpoints without version are assumed to be compatible
    if ckpt_version is None:
        if logger:
            logger.warning(
                f"Checkpoint {checkpoint_path.name} has no format_version field. "
                f"Assuming compatibility with current version {CHECKPOINT_FORMAT_VERSION}. "
                f"This checkpoint was created before versioning was added."
            )
        return

    # Check version compatibility
    if ckpt_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"\n{'='*70}\n"
            f"CHECKPOINT FORMAT VERSION MISMATCH:\n"
            f"  Checkpoint: {checkpoint_path}\n"
            f"  Checkpoint version: {ckpt_version}\n"
            f"  Current version:    {CHECKPOINT_FORMAT_VERSION}\n\n"
            f"This checkpoint was created with a different format version and may\n"
            f"not be compatible with the current training code.\n\n"
            f"Options:\n"
            f"  1. Use a checkpoint with version {CHECKPOINT_FORMAT_VERSION}\n"
            f"  2. Update your code to version {ckpt_version}\n"
            f"  3. If you're sure about compatibility, temporarily modify\n"
            f"     CHECKPOINT_FORMAT_VERSION in training/checkpoints.py\n"
            f"{'='*70}"
        )


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    batch_idx,
    global_step,
    train_loss,
    val_loss,
    best_val_loss,
    args,
    checkpoint_path: Path,
    logger,
    train_perplexity: Optional[float] = None,
    val_perplexity: Optional[float] = None,
    wandb_run_id: Optional[str] = None,
    *,
    sampler_state: Optional[dict] = None
):
    """Save training checkpoint with essential metadata

    Args:
        batch_idx: The batch index within the current epoch (used for mid-epoch resume)
        train_perplexity: Training perplexity (optional)
        val_perplexity: Validation perplexity (optional)
        wandb_run_id: Weights & Biases run ID (optional, for resumption)
        sampler_state: Sampler state dict for mid-epoch resume (optional, pass None if unavailable)
    """
    checkpoint = {
        # Checkpoint format metadata
        'format_version': CHECKPOINT_FORMAT_VERSION,

        # Core training state
        'epoch': epoch,
        'batch_idx': batch_idx,  # Batch index within epoch for mid-epoch resume
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'args': vars(args),

        # Essential metadata
        'timestamp': datetime.now().isoformat(),
        'train_loss': train_loss,
        'train_perplexity': train_perplexity,
        'val_loss': val_loss,
        'val_perplexity': val_perplexity,
        'best_val_loss': best_val_loss,
        'learning_rate': optimizer.param_groups[0]['lr'],
        'regime': args.regime,
        'vision_mode': args.vision_mode if args.regime == 'vision' else None,
        'train_encoder': args.train_encoder,  # Track encoder training mode

        # W&B run ID (for resumption)
        'wandb_run_id': wandb_run_id,

        # Output directory (for resumption)
        'output_dir': str(checkpoint_path.parent),

        # RNG states for reproducibility (enables bit-exact resume)
        'rng_states': {
            'python': random.getstate(),
            'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },

        # Sampler state for mid-epoch resume (prevents duplicate data)
        'sampler_state': sampler_state,

        # Legacy compatibility (kept for older code)
        'step': batch_idx,
    }

    # Atomic save pattern: write to temp file, then atomic rename
    # This prevents corruption if process is killed during save
    temp_path = checkpoint_path.with_suffix('.pt.tmp')

    try:
        # Clean up stale temp file if exists
        if temp_path.exists():
            temp_path.unlink()

        # Save to temporary file
        torch.save(checkpoint, temp_path)

        # Atomic rename (replaces old file safely on POSIX systems)
        os.replace(temp_path, checkpoint_path)

        logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Log checkpoint to tracking file for easy run ID → checkpoint mapping
        checkpoint_log_path = checkpoint_path.parent / 'checkpoint_log.txt'
        try:
            with open(checkpoint_log_path, 'a') as f:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                run_id = wandb_run_id if wandb_run_id else 'none'
                filename = checkpoint_path.name
                val_loss_str = f"{val_loss:.4f}" if val_loss is not None else 'N/A'

                f.write(f"{timestamp} | run_id: {run_id} | checkpoint: {filename} | "
                       f"global_step: {global_step} | val_loss: {val_loss_str}\n")
        except Exception as e:
            # Don't fail checkpoint save if log write fails
            logger.warning(f"Failed to write to checkpoint_log.txt: {e}")
    except (OSError, RuntimeError) as e:
        logger.error(f"Failed to save checkpoint to {checkpoint_path}: {e}")
        # Clean up failed temp file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass  # Best effort cleanup
        # Re-raise to alert operators that checkpoint save failed
        raise


def peek_checkpoint_metadata(checkpoint_path: Path, logger):
    """
    Load checkpoint metadata only (without applying model/optimizer state)

    This is used early in the training setup to extract metadata (especially
    wandb_run_id) before initializing wandb, while the actual model/optimizer
    state loading happens later after these objects are created.

    Also used for two-stage training to validate architecture before loading weights.

    Note: This loads the full checkpoint to extract metadata, then discards it.
    The checkpoint will be loaded again later by load_checkpoint(). This is
    simpler and safer than caching, at the cost of loading the file twice.

    Returns:
        Dict with keys: epoch, batch_idx, best_val_loss, global_step, wandb_run_id,
        regime, objective, train_encoder, and all regime-specific hyperparameters
    """
    logger.info(f"Peeking checkpoint metadata from {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=False,  # Required: checkpoint contains RNG states (numpy arrays)
        )
    except (RuntimeError, EOFError, OSError, pickle.UnpicklingError) as e:
        logger.error(
            f"\n{'='*70}\n"
            f"FAILED TO LOAD CHECKPOINT (file may be corrupted):\n"
            f"  Path: {checkpoint_path}\n"
            f"  Error: {type(e).__name__}: {e}\n\n"
            f"Recovery options:\n"
            f"  1. Delete the corrupted checkpoint: rm {checkpoint_path}\n"
            f"  2. Resume from an earlier checkpoint (check {checkpoint_path.parent}/)\n"
            f"  3. If this is 'latest.pt', it may have been interrupted during save.\n"
            f"     Try 'best_checkpoint.pt' or the most recent periodic checkpoint.\n"
            f"{'='*70}"
        )
        raise

    # Validate checkpoint format version
    _validate_checkpoint_version(checkpoint, checkpoint_path, logger)

    ckpt_args = checkpoint.get('args', {})

    metadata = {
        # Training state
        'checkpoint_path': str(checkpoint_path),
        'epoch': checkpoint['epoch'],
        'batch_idx': checkpoint.get('batch_idx', checkpoint.get('step', -1)),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'global_step': checkpoint.get('global_step', 0),
        'wandb_run_id': checkpoint.get('wandb_run_id', None),
        'batch_size': ckpt_args.get('batch_size', None),
        'sampler_state': checkpoint.get('sampler_state', None),

        # Architecture metadata (for two-stage training validation)
        'regime': checkpoint.get('regime'),
        'objective': ckpt_args.get('objective'),
        'train_encoder': checkpoint.get('train_encoder'),

        # Regime-specific hyperparameters
        'compression_target': ckpt_args.get('compression_target'),
        'conv_kernel': ckpt_args.get('conv_kernel'),
        'vision_mode': ckpt_args.get('vision_mode'),
        'hybrid_text_tokens': ckpt_args.get('hybrid_text_tokens'),
        'projection_dim': ckpt_args.get('projection_dim'),
        'compression_window_size': ckpt_args.get('compression_window_size'),
        'compression_stride': ckpt_args.get('compression_stride'),
        'subsample_count': ckpt_args.get('subsample_count'),
        'subsample_strategy': ckpt_args.get('subsample_strategy'),
    }

    logger.info(
        f"Checkpoint metadata: epoch={metadata['epoch']}, "
        f"batch_idx={metadata['batch_idx']}, global_step={metadata['global_step']}"
    )
    if metadata['wandb_run_id']:
        logger.info(f"  W&B run ID: {metadata['wandb_run_id']}")

    return metadata


def _normalize_state_dict_keys(checkpoint_state_dict: dict, model_state_dict: dict, logger=None) -> dict:
    """
    Normalize checkpoint state dict keys to match model's expected keys.

    Handles torch.compile() prefix mismatch:
    - checkpoint has _orig_mod.* but model expects * (Stage 1 compiled, Stage 2 not)
    - checkpoint has * but model expects _orig_mod.* (Stage 1 not compiled, Stage 2 compiled)

    Args:
        checkpoint_state_dict: State dict from checkpoint
        model_state_dict: State dict from current model (to detect expected key format)
        logger: Optional logger for info messages

    Returns:
        Normalized state dict with keys matching model's expectations
    """
    # Check if model uses _orig_mod prefix (is compiled)
    model_has_prefix = any(k.startswith('_orig_mod.') for k in model_state_dict.keys())

    # Check if checkpoint uses _orig_mod prefix
    ckpt_has_prefix = any(k.startswith('_orig_mod.') for k in checkpoint_state_dict.keys())

    # No mismatch - return as-is
    if model_has_prefix == ckpt_has_prefix:
        return checkpoint_state_dict

    # Log the normalization
    if logger:
        ckpt_status = "compiled" if ckpt_has_prefix else "uncompiled"
        model_status = "compiled" if model_has_prefix else "uncompiled"
        direction = "removing" if ckpt_has_prefix else "adding"
        logger.info(
            f"torch.compile mismatch: checkpoint={ckpt_status}, model={model_status}. "
            f"Normalizing keys by {direction} _orig_mod. prefix."
        )

    # Normalize keys
    normalized = {}
    for key, value in checkpoint_state_dict.items():
        if ckpt_has_prefix and not model_has_prefix:
            # Strip _orig_mod. prefix
            new_key = key.replace('_orig_mod.', '', 1) if key.startswith('_orig_mod.') else key
        else:
            # Add _orig_mod. prefix
            new_key = f'_orig_mod.{key}' if not key.startswith('_orig_mod.') else key
        normalized[new_key] = value

    return normalized


def load_checkpoint(checkpoint_path: Path, model, optimizer=None, scheduler=None, device='cuda', logger=None, args=None, weights_only=False):
    """
    Load training checkpoint state (model, optimizer, scheduler)

    Note: This function loads the actual model/optimizer/scheduler state.
    For metadata-only loading (used before model creation), use peek_checkpoint_metadata().

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (ignored if weights_only=True)
        scheduler: Scheduler to load state into (ignored if weights_only=True)
        device: Device to load checkpoint on
        logger: Logger instance
        args: Optional training args to check for encoder_trainable mismatch
        weights_only: If True, load only model weights (skip optimizer/scheduler/RNG)
                     Use for two-stage training (reconstruction → LM fine-tuning)

    Returns:
        If weights_only=False (full resume):
            Tuple of (epoch, batch_idx, best_val_loss, global_step, wandb_run_id)
            where batch_idx=-1 indicates end-of-epoch checkpoint
        If weights_only=True (two-stage training):
            Dict with checkpoint metadata for logging

    Note:
        sampler_state is available via peek_checkpoint_metadata() for DataLoader creation
    """
    if weights_only:
        logger.info(f"Loading model weights for two-stage training from {checkpoint_path}")
    else:
        logger.info(f"Loading checkpoint state (model/optimizer/scheduler) from {checkpoint_path}")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,  # Required: checkpoint contains RNG states (numpy arrays)
        )
    except (RuntimeError, EOFError, OSError, pickle.UnpicklingError) as e:
        logger.error(
            f"\n{'='*70}\n"
            f"FAILED TO LOAD CHECKPOINT (file may be corrupted):\n"
            f"  Path: {checkpoint_path}\n"
            f"  Error: {type(e).__name__}: {e}\n\n"
            f"Recovery options:\n"
            f"  1. Delete the corrupted checkpoint: rm {checkpoint_path}\n"
            f"  2. Resume from an earlier checkpoint (check {checkpoint_path.parent}/)\n"
            f"  3. If this is 'latest.pt', it may have been interrupted during save.\n"
            f"     Try 'best_checkpoint.pt' or the most recent periodic checkpoint.\n"
            f"{'='*70}"
        )
        raise

    # Validate checkpoint format version
    _validate_checkpoint_version(checkpoint, checkpoint_path, logger)

    # Check for encoder training mode mismatch (only for full resume, not two-stage training)
    # For two-stage training (weights_only=True), optimizer/scheduler are intentionally
    # skipped, so this warning about optimizer state would be misleading
    if args is not None and not weights_only:
        # Try new unified name first, fall back to legacy name for old checkpoints
        ckpt_train_encoder = checkpoint.get('train_encoder')
        if ckpt_train_encoder is None:
            ckpt_train_encoder = checkpoint.get('train_projection', False)
        current_train_encoder = getattr(args, 'train_encoder', False)

        if ckpt_train_encoder != current_train_encoder:
            logger.warning(
                f"\n{'='*70}\n"
                f"ENCODER TRAINING MODE MISMATCH DETECTED:\n"
                f"  Checkpoint: train_encoder={ckpt_train_encoder}\n"
                f"  Current:    train_encoder={current_train_encoder}\n"
                f"\n"
                f"IMPORTANT: Optimizer/scheduler state will NOT be loaded due to\n"
                f"param group mismatch. This means:\n"
                f"  - Adam momentum and variance history will be lost\n"
                f"  - Training dynamics will change at resume point\n"
                f"  - Loss curves before/after resume are NOT directly comparable\n"
                f"\n"
                f"For scientifically valid experiments, consider:\n"
                f"  1. Training from scratch with --train_encoder, OR\n"
                f"  2. Resuming with same encoder mode as checkpoint\n"
                f"{'='*70}\n"
            )

    # Normalize state dict keys for torch.compile() prefix mismatch
    # (handles compiled checkpoint → uncompiled model and vice versa)
    checkpoint_state_dict = _normalize_state_dict_keys(
        checkpoint['model_state_dict'],
        model.state_dict(),
        logger=logger
    )

    # Try to load model state dict
    try:
        model.load_state_dict(checkpoint_state_dict)
    except RuntimeError as e:
        # Handle old checkpoints that don't have separator/randproj/encoder parameters
        error_msg = str(e)

        # Check for known missing/unexpected key patterns
        has_separator = 'separator' in error_msg
        has_randproj = 'randproj_matrix' in error_msg
        has_conv1d = 'conv1d_pyramid' in error_msg or 'conv1d_residual_pyramid' in error_msg
        has_encoder = ('sam_model' in error_msg or 'vision_model' in error_msg or
                      'projector' in error_msg or 'image_newline' in error_msg or
                      'view_seperator' in error_msg)

        if has_separator or has_randproj or has_conv1d or has_encoder:
            logger.warning("Checkpoint architecture mismatch detected - loading with strict=False")

            # Load with strict=False to skip missing/unexpected parameters
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint_state_dict, strict=False)

            # Log missing regime-specific keys (model lacks these, checkpoint may have them)
            regime_missing = [k for k in missing_keys if ('separator' in k or 'randproj_matrix' in k or
                                                          'conv1d_pyramid' in k or 'conv1d_residual_pyramid' in k)]
            if regime_missing:
                logger.info(f"Initialized new parameters (not in checkpoint): {', '.join(regime_missing)}")

            # Log unexpected encoder keys (checkpoint has these, model doesn't need them)
            encoder_keys = [k for k in unexpected_keys if ('sam_model' in k or 'vision_model' in k or
                                                           'projector' in k or 'image_newline' in k or
                                                           'view_seperator' in k)]
            if encoder_keys:
                logger.info(f"Skipped {len(encoder_keys)} vision encoder parameters (model loaded without encoder)")
        else:
            # Some other error - re-raise
            raise

    # Skip optimizer/scheduler/RNG loading if weights_only=True (two-stage training)
    if not weights_only:
        # Load optimizer state with error handling for param group mismatches
        # This handles cases like frozen→trainable encoder where param groups change
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("✓ Successfully loaded optimizer state from checkpoint")
        except ValueError as e:
            # Expected when param groups change (e.g., frozen→trainable encoder)
            error_str = str(e).lower()
            if 'param_groups' in error_str or 'expected' in error_str or 'state_dict' in error_str:
                logger.warning(
                    f"Cannot load optimizer state due to param group mismatch: {e}\n"
                    f"This is expected when switching between frozen/trainable encoder modes.\n"
                    f"Optimizer will start fresh with current configuration.\n"
                    f"NOTE: Adam momentum/variance history will be lost."
                )
            else:
                # Unexpected ValueError - could indicate corruption
                logger.error(f"Unexpected ValueError loading optimizer state: {e}")
                raise
        except KeyError as e:
            # Missing key in checkpoint - could indicate version mismatch or corruption
            logger.error(f"Missing key in checkpoint optimizer state: {e}")
            logger.error("Checkpoint may be corrupted or from incompatible version")
            raise
        except RuntimeError as e:
            # Runtime errors are typically unexpected
            logger.error(f"RuntimeError loading optimizer state: {e}")
            raise

        # Load scheduler state with error handling for param group mismatches
        try:
            if scheduler and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("✓ Successfully loaded scheduler state from checkpoint")
        except ValueError as e:
            # Expected when param groups change
            error_str = str(e).lower()
            if 'param_groups' in error_str or 'expected' in error_str:
                logger.warning(
                    f"Cannot load scheduler state due to param group mismatch: {e}\n"
                    f"Scheduler will restart from current step."
                )
            else:
                logger.error(f"Unexpected ValueError loading scheduler state: {e}")
                raise
        except (KeyError, RuntimeError) as e:
            logger.error(f"Error loading scheduler state: {e}")
            raise

        # Restore RNG states for reproducibility (backward compatible with old checkpoints)
        if 'rng_states' in checkpoint:
            try:
                random.setstate(checkpoint['rng_states']['python'])
                np.random.set_state(checkpoint['rng_states']['numpy'])
                # Torch RNG state must be a CPU ByteTensor, but map_location='cuda'
                # may have moved it to GPU. Ensure it's on CPU before restoring.
                torch_rng = checkpoint['rng_states']['torch']
                if hasattr(torch_rng, 'device') and torch_rng.device.type != 'cpu':
                    torch_rng = torch_rng.cpu()
                torch.set_rng_state(torch_rng)
                if checkpoint['rng_states']['torch_cuda'] and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(checkpoint['rng_states']['torch_cuda'])
                logger.info("✓ Restored RNG states from checkpoint (enables bit-exact resume)")
            except Exception as e:
                logger.warning(f"Failed to restore RNG states: {e}. Continuing with current RNG state.")
        else:
            logger.warning(
                "Checkpoint does not contain RNG states. Resume will not be bit-exact.\n"
                "  This is expected for checkpoints created before RNG state saving was added.\n"
                "  Training will continue but randomness (dropout, transforms) may differ slightly."
            )
    else:
        logger.info("✓ Skipping optimizer/scheduler/RNG states (two-stage training)")


    # Return different values based on mode
    if weights_only:
        # Two-stage training: return checkpoint metadata for logging
        ckpt_args = checkpoint.get('args', {})
        metadata = {
            'checkpoint_path': str(checkpoint_path),
            'regime': checkpoint.get('regime'),
            'objective': ckpt_args.get('objective'),
            'epoch': checkpoint['epoch'],
            'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
            'wandb_run_id': checkpoint.get('wandb_run_id', None),
        }
        return metadata
    else:
        # Full resume: return training state for main() to restore
        epoch = checkpoint['epoch']
        batch_idx = checkpoint.get('batch_idx', checkpoint.get('step', -1))
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        global_step = checkpoint.get('global_step', 0)
        wandb_run_id = checkpoint.get('wandb_run_id', None)

        return epoch, batch_idx, best_val_loss, global_step, wandb_run_id


# ============================================================================
# Two-Stage Training Validation
# ============================================================================

def validate_and_load_init_checkpoint(checkpoint_metadata, args, logger):
    """
    Validate that Stage 1 checkpoint is compatible with Stage 2 configuration.

    Args:
        checkpoint_metadata: Metadata dict returned from load_checkpoint(..., weights_only=True)
        args: Current training arguments
        logger: Logger instance

    Raises:
        ValueError: If incompatible configuration detected
    """
    ckpt_regime = checkpoint_metadata['regime']
    ckpt_objective = checkpoint_metadata['objective']

    # Handle legacy checkpoints without objective metadata
    if ckpt_objective is None:
        logger.warning(
            f"\n{'='*70}\n"
            f"LEGACY CHECKPOINT DETECTED:\n"
            f"This checkpoint does not contain objective metadata.\n"
            f"Skipping objective validation - ensure you're using compatible settings.\n"
            f"{'='*70}\n"
        )
        # Skip objective validation for legacy checkpoints
        ckpt_objective = args.objective  # Assume same objective to skip checks below

    # Validate regime matches
    if ckpt_regime != args.regime:
        raise ValueError(
            f"Regime mismatch: Stage 1 used '{ckpt_regime}', Stage 2 uses '{args.regime}'.\n"
            f"\n"
            f"Two-stage training requires matching regimes.\n"
            f"Solution: Change Stage 2 regime to match Stage 1:\n"
            f"  python train.py --regime {ckpt_regime} --objective {args.objective} \\\n"
            f"                  --init_from_checkpoint {args.init_from_checkpoint} \\\n"
            f"                  --allow_objective_switch"
        )

    # Validate objective switching
    if ckpt_objective != args.objective:
        if not args.allow_objective_switch:
            raise ValueError(
                f"Objective mismatch: Stage 1 used '{ckpt_objective}', Stage 2 uses '{args.objective}'.\n"
                f"\n"
                f"This is typically intentional for two-stage training (reconstruction → lm).\n"
                f"To proceed, add the --allow_objective_switch flag:\n"
                f"\n"
                f"  python train.py --regime {args.regime} --objective {args.objective} \\\n"
                f"                  --init_from_checkpoint {args.init_from_checkpoint} \\\n"
                f"                  --allow_objective_switch\n"
                f"\n"
                f"If you meant to RESUME training (not switch objectives), use --resume_from_checkpoint instead."
            )
        else:
            logger.info(f"✓ Objective switch: {ckpt_objective} → {args.objective} (two-stage training)")
    else:
        # Same objective - warn user (unusual for two-stage training)
        logger.warning(
            f"\n{'='*70}\n"
            f"WARNING: Both stages use objective='{args.objective}'\n"
            f"This defeats the purpose of two-stage training!\n"
            f"\n"
            f"Did you mean:\n"
            f"  Stage 1: --objective reconstruction (learn representation)\n"
            f"  Stage 2: --objective lm (learn language modeling)?\n"
            f"\n"
            f"Continuing anyway...\n"
            f"{'='*70}\n"
        )

    # Validate regime-specific hyperparameters
    if args.regime in ['conv1d', 'conv1d_residual', 'conv1d_residual_auxloss']:
        ckpt_target = checkpoint_metadata.get('compression_target')
        if ckpt_target and ckpt_target != args.compression_target:
            raise ValueError(
                f"Compression target mismatch: Stage 1 used {ckpt_target}, Stage 2 uses {args.compression_target}.\n"
                f"\n"
                f"These must match for two-stage training (architecture must be identical).\n"
                f"Add to your command:\n"
                f"  --compression_target {ckpt_target}"
            )

        ckpt_kernel = checkpoint_metadata.get('conv_kernel')
        if ckpt_kernel and ckpt_kernel != args.conv_kernel:
            raise ValueError(
                f"Conv kernel mismatch: Stage 1 used {ckpt_kernel}, Stage 2 uses {args.conv_kernel}.\n"
                f"\n"
                f"Add to your command:\n"
                f"  --conv_kernel {ckpt_kernel}"
            )

        # Log hybrid_text_tokens change (allowed - text tokens don't affect model architecture)
        ckpt_hybrid = checkpoint_metadata.get('hybrid_text_tokens', 0)
        if ckpt_hybrid != args.hybrid_text_tokens:
            logger.info(f"Hybrid text tokens: Stage 1 used {ckpt_hybrid}, Stage 2 uses {args.hybrid_text_tokens}")

    if args.regime == 'vision':
        ckpt_mode = checkpoint_metadata.get('vision_mode')
        if ckpt_mode and ckpt_mode != args.vision_mode:
            raise ValueError(
                f"Vision mode mismatch: Stage 1 used '{ckpt_mode}', Stage 2 uses '{args.vision_mode}'.\n"
                f"\n"
                f"Add to your command:\n"
                f"  --vision_mode {ckpt_mode}"
            )

        # Log hybrid_text_tokens change (allowed - text tokens don't affect model architecture)
        ckpt_hybrid = checkpoint_metadata.get('hybrid_text_tokens')
        if ckpt_hybrid is not None and ckpt_hybrid != args.hybrid_text_tokens:
            logger.info(f"Hybrid text tokens: Stage 1 used {ckpt_hybrid}, Stage 2 uses {args.hybrid_text_tokens}")

    if args.regime == 'randproj':
        ckpt_proj_dim = checkpoint_metadata.get('projection_dim')
        if ckpt_proj_dim and ckpt_proj_dim != args.projection_dim:
            raise ValueError(
                f"Projection dim mismatch: Stage 1 used {ckpt_proj_dim}, Stage 2 uses {args.projection_dim}.\n"
                f"\n"
                f"Add to your command:\n"
                f"  --projection_dim {ckpt_proj_dim}"
            )

    if args.regime == 'meanpool':
        ckpt_window = checkpoint_metadata.get('compression_window_size')
        if ckpt_window and ckpt_window != args.compression_window_size:
            raise ValueError(
                f"Window size mismatch: Stage 1 used {ckpt_window}, Stage 2 uses {args.compression_window_size}.\n"
                f"\n"
                f"Add to your command:\n"
                f"  --compression_window_size {ckpt_window}"
            )

        ckpt_stride = checkpoint_metadata.get('compression_stride')
        if ckpt_stride and ckpt_stride != args.compression_stride:
            raise ValueError(
                f"Stride mismatch: Stage 1 used {ckpt_stride}, Stage 2 uses {args.compression_stride}.\n"
                f"\n"
                f"Add to your command:\n"
                f"  --compression_stride {ckpt_stride}"
            )

        # Log hybrid_text_tokens change (allowed - text tokens don't affect model architecture)
        ckpt_hybrid = checkpoint_metadata.get('hybrid_text_tokens')
        if ckpt_hybrid is not None and ckpt_hybrid != args.hybrid_text_tokens:
            logger.info(f"Hybrid text tokens: Stage 1 used {ckpt_hybrid}, Stage 2 uses {args.hybrid_text_tokens}")

    if args.regime == 'subsample':
        ckpt_count = checkpoint_metadata.get('subsample_count')
        if ckpt_count and ckpt_count != args.subsample_count:
            raise ValueError(
                f"Subsample count mismatch: Stage 1 used {ckpt_count}, Stage 2 uses {args.subsample_count}.\n"
                f"\n"
                f"Add to your command:\n"
                f"  --subsample_count {ckpt_count}"
            )

        ckpt_strategy = checkpoint_metadata.get('subsample_strategy')
        if ckpt_strategy and ckpt_strategy != args.subsample_strategy:
            raise ValueError(
                f"Subsample strategy mismatch: Stage 1 used '{ckpt_strategy}', Stage 2 uses '{args.subsample_strategy}'.\n"
                f"\n"
                f"This changes the compression pipeline behavior.\n"
                f"Add to your command:\n"
                f"  --subsample_strategy {ckpt_strategy}"
            )


# ============================================================================
# Checkpoint Argument Loading (for --resume flag)
# ============================================================================

def load_checkpoint_args(checkpoint_path: Path):
    """
    Load training arguments from checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary of training arguments from checkpoint

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If checkpoint has no args metadata
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading configuration from checkpoint: {checkpoint_path}")

    # Load checkpoint (map to CPU to avoid CUDA requirements)
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu',
            weights_only=False  # Required for RNG states
        )
    except (RuntimeError, EOFError, OSError, pickle.UnpicklingError) as e:
        print(
            f"\n{'='*70}\n"
            f"FAILED TO LOAD CHECKPOINT (file may be corrupted):\n"
            f"  Path: {checkpoint_path}\n"
            f"  Error: {type(e).__name__}: {e}\n\n"
            f"Recovery options:\n"
            f"  1. Delete the corrupted checkpoint: rm {checkpoint_path}\n"
            f"  2. Resume from an earlier checkpoint (check {checkpoint_path.parent}/)\n"
            f"  3. If this is 'latest.pt', it may have been interrupted during save.\n"
            f"     Try 'best_checkpoint.pt' or the most recent periodic checkpoint.\n"
            f"{'='*70}",
            file=sys.stderr
        )
        raise

    # Validate checkpoint format version (use print since no logger available)
    ckpt_version = checkpoint.get('format_version')
    if ckpt_version is None:
        print(f"Warning: Checkpoint has no format_version field. "
              f"Assuming compatibility with current version {CHECKPOINT_FORMAT_VERSION}.")
    elif ckpt_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"\n{'='*70}\n"
            f"CHECKPOINT FORMAT VERSION MISMATCH:\n"
            f"  Checkpoint: {checkpoint_path}\n"
            f"  Checkpoint version: {ckpt_version}\n"
            f"  Current version:    {CHECKPOINT_FORMAT_VERSION}\n\n"
            f"This checkpoint was created with a different format version and may\n"
            f"not be compatible with the current training code.\n\n"
            f"Options:\n"
            f"  1. Use a checkpoint with version {CHECKPOINT_FORMAT_VERSION}\n"
            f"  2. Update your code to version {ckpt_version}\n"
            f"  3. If you're sure about compatibility, temporarily modify\n"
            f"     CHECKPOINT_FORMAT_VERSION in training/checkpoints.py\n"
            f"{'='*70}"
        )

    if 'args' not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} has no 'args' metadata")

    ckpt_args = checkpoint['args']

    # Display key configuration
    print("Checkpoint configuration:")
    print(f"  Regime:      {ckpt_args.get('regime', 'N/A')}")
    print(f"  Objective:   {ckpt_args.get('objective', 'N/A')}")
    print(f"  Batch size:  {ckpt_args.get('batch_size', 'N/A')}")
    if 'vision_mode' in ckpt_args and ckpt_args['vision_mode']:
        print(f"  Vision mode: {ckpt_args['vision_mode']}")

    return ckpt_args


def merge_args(checkpoint_args: dict, cli_args):
    """
    Merge checkpoint arguments with CLI arguments.

    Strategy:
    - Start with all checkpoint args as defaults
    - Allow CLI to override safe settings (logging, validation)
    - Error on critical settings that would break training (architecture, batch size)

    Args:
        checkpoint_args: Args dict from checkpoint
        cli_args: Parsed CLI arguments

    Returns:
        Merged argparse.Namespace

    Raises:
        ValueError: If critical args conflict between checkpoint and CLI
    """
    # Critical args that MUST match checkpoint (architecture/training critical)
    # Note: hybrid_text_tokens is NOT critical - it's just input config, not model architecture
    CRITICAL_ARGS = {
        'regime', 'objective', 'batch_size', 'gradient_accumulation_steps',
        'train_encoder', 'vision_mode',
        'compression_target', 'conv_kernel',
        'compression_window_size', 'compression_stride',
        'subsample_count', 'subsample_strategy', 'projection_dim',
        'text_context_tokens',
        'aux_loss_weight',  # Affects loss computation
        'vision_prompt',    # Affects input tokenization
        'encoder_lr'        # Differential learning rate for encoder
    }

    # Safe args that CAN be overridden (logging/validation/runtime settings)
    SAFE_OVERRIDES = {
        'eval_steps', 'log_steps', 'save_steps',
        'num_epochs', 'initial_validation',
        'num_qualitative_samples', 'max_generation_tokens',
        'no_checkpoints', 'validation_only',
        'use_wandb', 'wandb_project', 'wandb_run_name',
        'num_workers', 'prefetch_factor',
        'compile', 'use_optimized_model', 'use_encoder_checkpointing',
        'debug_log_sample_ids',  # Debug flag
        'device',                # Runtime setting
        'eval_seed',             # Validation setting
        'data_path',             # Training data path (can switch JSONL <-> Arrow)
        'val_data_path'          # Validation data path
    }

    # Start with CLI defaults to ensure all current arguments exist
    # This prevents AttributeError when new arguments are added after checkpoint was created
    import argparse
    import sys
    merged = argparse.Namespace(**vars(cli_args))

    # Overlay checkpoint values (preserving CLI defaults for arguments that didn't exist when checkpoint was created)
    for key, checkpoint_value in checkpoint_args.items():
        # Don't blindly override - we'll check for explicit CLI overrides below
        if key not in ['resume', 'resume_from_checkpoint', 'init_from_checkpoint', 'allow_objective_switch']:
            setattr(merged, key, checkpoint_value)

    # Build a set of destination names that were explicitly provided on CLI
    # This handles both positive and negative boolean flag forms (e.g., --compile and --no-compile)
    explicitly_set_args = set()

    # Map CLI flag names to their destination names by checking sys.argv
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            # Extract flag name (remove -- prefix, handle --arg=value syntax)
            flag_name = arg[2:].split('=')[0]

            # Convert to potential destination name (replace dashes with underscores)
            # This handles standard flags like --use-wandb → use_wandb
            dest_name = flag_name.replace('-', '_')
            explicitly_set_args.add(dest_name)

            # Also handle negative boolean flags mapping to positive destinations
            # E.g., --no-compile should mark 'compile' as explicitly set
            if flag_name.startswith('no-'):
                # Strip 'no-' prefix to get the positive form
                positive_dest = flag_name[3:].replace('-', '_')
                explicitly_set_args.add(positive_dest)
            elif flag_name.startswith('no_'):
                # Handle --no_flag format
                positive_dest = flag_name[3:]
                explicitly_set_args.add(positive_dest)

            # Special case: --checkpoints is the negative form of --no_checkpoints
            if flag_name == 'checkpoints':
                explicitly_set_args.add('no_checkpoints')

    # Now apply explicit CLI overrides (for args that were explicitly provided on command line)
    cli_dict = vars(cli_args)
    for key in explicitly_set_args:
        # Skip if key not in CLI args (shouldn't happen but be defensive)
        if key not in cli_dict:
            continue

        cli_value = cli_dict[key]
        checkpoint_value = checkpoint_args.get(key)

        # Skip special resume-only args (already set above)
        if key in ['resume', 'resume_from_checkpoint', 'init_from_checkpoint', 'allow_objective_switch']:
            continue

        # Check for critical arg conflicts
        if key in CRITICAL_ARGS:
            if checkpoint_value is not None and checkpoint_value != cli_value:
                raise ValueError(
                    f"\n{'='*70}\n"
                    f"ERROR: Cannot override '{key}' when resuming from checkpoint.\n"
                    f"  Checkpoint: {key}={checkpoint_value}\n"
                    f"  CLI:        {key}={cli_value}\n\n"
                    f"This setting is critical for training and must match the checkpoint.\n"
                    f"Remove --{key.replace('_', '-')} from your command line arguments.\n\n"
                    f"If you want to change this setting, you need to start a new training run,\n"
                    f"not resume from this checkpoint.\n"
                    f"{'='*70}"
                )

        # Allow safe overrides
        if key in SAFE_OVERRIDES:
            if checkpoint_value != cli_value:
                print(f"Overriding {key}: {checkpoint_value} → {cli_value}")
            setattr(merged, key, cli_value)
        elif key not in CRITICAL_ARGS and key not in SAFE_OVERRIDES:
            # Warn about unclassified arguments (likely hyperparameters or infrastructure settings)
            print(f"\nWARNING: Cannot override '{key}' when resuming from checkpoint.")
            print(f"  This argument is not in the safe override list.")
            print(f"  Using checkpoint value: {checkpoint_value}")
            print(f"  To change this setting, start a new training run.\n")

    return merged
