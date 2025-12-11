"""Unified naming logic for output directories and W&B run names.

This module provides a single source of truth for generating consistent names
across the bash wrapper and Python training script. Previously, naming logic
was duplicated between run_production_train.sh and train.py, leading to:
- Naming inconsistencies (conv1d_residual 't' vs 'r' prefix, etc.)
- Maintenance burden (changes needed in 2 places)

Now both scripts use these functions to ensure OUTPUT_DIR and wandb_run_name
are always identical.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional
import argparse


def generate_run_name(
    regime: str,
    objective: str,
    timestamp: Optional[str] = None,
    # Vision regime params
    vision_mode: Optional[str] = None,
    # Text regime params
    text_context_tokens: Optional[int] = None,
    # Meanpool regime params
    compression_window_size: Optional[int] = None,
    compression_stride: Optional[int] = None,
    # Conv1D regime params
    compression_target: Optional[int] = None,
    conv_kernel: Optional[int] = None,
    train_encoder: Optional[bool] = None,
    # Hybrid mode (vision, meanpool, conv1d_residual)
    hybrid_text_tokens: Optional[int] = None,
) -> str:
    """Generate consistent run name for output directory and W&B.

    Args:
        regime: Training regime (vision, text, meanpool, conv1d_residual)
        objective: Training objective (lm or reconstruction)
        timestamp: Timestamp string (YYYYMMDD_HHMMSS). If None, auto-generated.
        **kwargs: Regime-specific parameters

    Returns:
        Run name string: 'production_{regime}_{params}_{objective}_{timestamp}'

    Examples:
        >>> generate_run_name('vision', 'lm', '20250120_143022', vision_mode='small')
        'production_vision_small_lm_20250120_143022'

        >>> generate_run_name('conv1d_residual', 'reconstruction', '20250120_143022',
        ...                   compression_target=125, conv_kernel=5, hybrid_text_tokens=100)
        'production_conv1d_residual_r125_k5_hybrid100_reconstruction_20250120_143022'
    """
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build name parts
    name_parts = ['production', regime]

    if regime == 'vision':
        # Vision regime: include vision_mode and hybrid tokens
        # Note: vision_prompt not included (can be arbitrary string)
        if vision_mode is None:
            raise ValueError("vision_mode required for vision regime")
        name_parts.append(vision_mode)
        if hybrid_text_tokens and hybrid_text_tokens > 0:
            name_parts.append(f'hybrid{hybrid_text_tokens}')

    elif regime == 'text':
        # Text regime: include context truncation if specified
        if text_context_tokens is not None:
            name_parts.append(f'ctx{text_context_tokens}')

    elif regime == 'meanpool':
        # Meanpool regime: include window size and stride
        if compression_window_size is None or compression_stride is None:
            raise ValueError("compression_window_size and compression_stride required for meanpool regime")
        name_parts.append(f'w{compression_window_size}')
        name_parts.append(f's{compression_stride}')
        if hybrid_text_tokens and hybrid_text_tokens > 0:
            name_parts.append(f'hybrid{hybrid_text_tokens}')

    elif regime == 'conv1d_residual':
        # Conv1D Residual regime: include compression target and kernel size
        # UNIFIED: Use 't' prefix (bash convention) instead of 'r' (old Python convention)
        # This matches bash OUTPUT_DIR naming for consistency
        if compression_target is None or conv_kernel is None:
            raise ValueError("compression_target and conv_kernel required for conv1d_residual regime")
        name_parts.append(f't{compression_target}')
        name_parts.append(f'k{conv_kernel}')
        if hybrid_text_tokens and hybrid_text_tokens > 0:
            name_parts.append(f'hybrid{hybrid_text_tokens}')

    else:
        raise ValueError(f"Unknown regime: {regime}")

    # Add objective (critical for distinguishing training goals)
    name_parts.append(objective)

    # Add timestamp
    name_parts.append(timestamp)

    return '_'.join(name_parts)


def generate_output_dir(
    regime: str,
    objective: str,
    timestamp: Optional[str] = None,
    base_dir: str = "outputs",
    **kwargs
) -> Path:
    """Generate output directory path.

    Args:
        regime: Training regime
        objective: Training objective
        timestamp: Timestamp string (YYYYMMDD_HHMMSS). If None, auto-generated.
        base_dir: Base directory for outputs (default: 'outputs')
        **kwargs: Regime-specific parameters (passed to generate_run_name)

    Returns:
        Path object for output directory

    Example:
        >>> generate_output_dir('vision', 'lm', '20250120_143022', vision_mode='small')
        PosixPath('outputs/production_vision_small_lm_20250120_143022')
    """
    run_name = generate_run_name(regime, objective, timestamp, **kwargs)
    return Path(base_dir) / run_name


def generate_from_args(args: argparse.Namespace) -> tuple[Path, str]:
    """Generate output_dir and wandb_run_name from argparse Namespace.

    This is a convenience function for train.py to extract all necessary
    parameters from args and generate both paths at once.

    Args:
        args: Parsed command-line arguments

    Returns:
        (output_dir_path, wandb_run_name) tuple

    Example:
        >>> output_dir, wandb_name = generate_from_args(args)
        >>> print(output_dir)
        outputs/production_vision_small_lm_20250120_143022
    """
    # Extract regime-specific params
    kwargs = {
        'regime': args.regime,
        'objective': args.objective,
        'timestamp': args.timestamp if hasattr(args, 'timestamp') and args.timestamp else None,
    }

    # Add regime-specific parameters
    if args.regime == 'vision':
        kwargs['vision_mode'] = args.vision_mode
        kwargs['hybrid_text_tokens'] = args.hybrid_text_tokens
    elif args.regime == 'text':
        kwargs['text_context_tokens'] = args.text_context_tokens
    elif args.regime == 'meanpool':
        kwargs['compression_window_size'] = args.compression_window_size
        kwargs['compression_stride'] = args.compression_stride
        kwargs['hybrid_text_tokens'] = args.hybrid_text_tokens
    elif args.regime == 'conv1d_residual':
        kwargs['compression_target'] = args.compression_target
        kwargs['conv_kernel'] = args.conv_kernel
        kwargs['hybrid_text_tokens'] = args.hybrid_text_tokens

    run_name = generate_run_name(**kwargs)
    output_dir = Path('outputs') / run_name

    return output_dir, run_name
