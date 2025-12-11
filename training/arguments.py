"""Command-line argument parser for training script."""

import argparse
import os


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser for DeepSeek-OCR training.

    Returns:
        Configured ArgumentParser with all training arguments
    """
    parser = argparse.ArgumentParser(description="Train DeepSeek-OCR with vision compression")

    # Required arguments (unless resuming from checkpoint)
    parser.add_argument('--regime', type=str, required=False,
                       choices=['vision', 'text', 'meanpool', 'subsample', 'randproj', 'conv1d', 'conv1d_residual', 'conv1d_residual_auxloss'],
                       help='Training regime: '
                            'vision (DeepSeek-OCR vision compression), '
                            'text (text baseline), '
                            'meanpool (sliding window mean pooling), '
                            'subsample (token subsampling), '
                            'randproj (random projection compression), '
                            'conv1d (1d convolutional pyramid compression), '
                            'conv1d_residual (1d conv pyramid with residual skip connections)')
    parser.add_argument('--data_path', type=str, required=False,
                       help='Path to training data (JSONL format). Required for training, not needed for --validation_only')
    parser.add_argument('--output_dir', type=str, required=False, default=None,
                       help='Output directory for checkpoints and logs. '
                            'If not provided, auto-generated from regime/objective/timestamp '
                            '(e.g., outputs/production_vision_small_lm_20250120_143022)')
    parser.add_argument('--objective', type=str, default='lm',
                       choices=['lm', 'reconstruction'],
                       help='Training objective: '
                            'lm (predict continuation), '
                            'reconstruction (predict original context)')

    # Optional data arguments
    parser.add_argument('--val_data_path', type=str, default=None,
                       help='Path to validation data (JSONL format)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to use from dataset (useful for quick testing). '
                            'If None, use all samples.')
    parser.add_argument('--vision_mode', type=str, default='small',
                       choices=['tiny', 'small', 'base', 'large'],
                       help='Vision compression mode (required for --regime=vision): ' +
                            'tiny (512x512, 73 tokens), small (640x640, 111 tokens), ' +
                            'base (1024x1024, 273 tokens), large (1280x1280, 421 tokens)')
    parser.add_argument('--text_context_tokens', type=int, default=None,
                       help='For text regime: number of context tokens to keep (truncate from end). '
                            'Default (None) keeps the full context provided in the dataset.')
    parser.add_argument('--hybrid_text_tokens', type=int, default=0,
                       help='For vision, meanpool, conv1d_residual regimes: number of uncompressed text tokens '
                            'to append after compressed representation (0 = compression-only, >0 = hybrid mode). '
                            'Example: 100 means compressed tokens + 100 text tokens.')
    parser.add_argument('--vision_prompt', type=str, default=None,
                       help='For vision regime: optional prompt to insert after vision tokens. '
                            'Can be a preset key or custom string (must start with \\n). '
                            'Presets: free_ocr, ocr_grounding, markdown, parse_figure, describe. '
                            'Custom example: "\\nMy custom prompt." '
                            'Makes training closer to DeepSeek-OCR pre-training. Default (None) = no prompt.')
    parser.add_argument('--train_encoder', action='store_true', dest='train_encoder',
                       help='Train the encoder/compression module (default: True). '
                            'For vision regime: trains SAM+ViT+Projector (~4.4 GB additional VRAM). '
                            'For conv1d regimes: trains the conv pyramid. '
                            'For randproj: trains the projection matrix. '
                            'Use with --encoder_lr to set differential learning rate.')
    parser.add_argument('--no-train-encoder', action='store_false', dest='train_encoder',
                       help='Freeze encoder/compression module. '
                            'For vision regime: use pretrained vision encoder as-is. '
                            'For randproj: true Johnson-Lindenstrauss (frozen random projection).')
    parser.set_defaults(train_encoder=True)
    parser.add_argument('--encoder_lr', type=float, default=1e-5,
                       help='Learning rate for vision encoder when --train_encoder is set. '
                            'Default 1e-5 (10x smaller than decoder to prevent catastrophic forgetting). '
                            'Only effective when --train_encoder is enabled.')

    # Compression settings (for meanpool and subsample regimes)
    parser.add_argument('--compression_window_size', type=int, default=9,
                       help='For meanpool regime: sliding window size for mean pooling. '
                            'Compressed tokens depend on both window_size and stride. '
                            'Example: window=9, stride=9 → 113 tokens. (default: 9)')
    parser.add_argument('--compression_stride', type=int, default=9,
                       help='For meanpool regime: stride for sliding window. '
                            'stride=window_size gives no overlap (fastest). '
                            'stride<window_size gives overlap (richer context). '
                            '(default: 9)')
    parser.add_argument('--subsample_strategy', type=str, default='regular',
                       choices=['regular', 'random'],
                       help='For subsample regime: sampling strategy. '
                            'regular = deterministic evenly-spaced positions, '
                            'random = stochastic random positions (order preserved). '
                            '(default: regular)')
    parser.add_argument('--subsample_count', type=int, required=False,
                       help='For subsample regime: number of tokens to keep (required when regime=subsample). '
                            'Example: --subsample_count 100 keeps 100 tokens.')
    parser.add_argument('--projection_dim', type=int, required=False,
                       help='For randproj regime: number of projected dimensions (required when regime=randproj). '
                            'Example: --projection_dim 111 projects to 111 tokens for ~9x compression matching vision-small.')
    parser.add_argument('--train_projection', action='store_true',
                       help='For randproj regime: make projection matrix trainable (default: frozen like Johnson-Lindenstrauss). '
                            'Checkpoints support switching between frozen/trainable.')
    parser.add_argument('--compression_target', type=int, required=False,
                       help='For conv1d/conv1d_residual/conv1d_residual_auxloss regimes: target number of compressed tokens (required for these regimes). '
                            'Example: --compression_target 111 to match vision-small compression.')
    parser.add_argument('--conv_kernel', type=int, default=5,
                       help='For conv1d/conv1d_residual/conv1d_residual_auxloss regimes: kernel size for convolutional layers (default: 5). '
                            'Must be odd. Larger kernels capture more local context.')

    # Timestamp (for consistent naming with shell script)
    parser.add_argument('--timestamp', type=str, default=None,
                       help='Timestamp for run naming (auto-generated if not provided). '
                            'Typically passed from shell script to ensure OUTPUT_DIR and W&B run names match exactly.')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size per GPU (default: 1, only 1 supported currently)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                       help='Number of gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Ratio of warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='Max gradient norm for clipping (CRITICAL for MoE)')

    # Logging and checkpointing
    parser.add_argument('--log_steps', type=int, default=10,
                       help='Log every N steps (set to 0 to disable periodic logging)')
    parser.add_argument('--save_steps', type=int, default=1000,
                       help='Save checkpoint every N gradient updates. Set to 0 to disable periodic '
                            'checkpoints (best/final are still saved when validation runs). '
                            'Use --no_checkpoints to disable all checkpoint saving.')
    parser.add_argument('--eval_steps', type=int, default=500,
                       help='Evaluate every N steps')
    parser.add_argument('--initial_validation', action='store_true', dest='initial_validation',
                       help='Run validation before training starts to establish baseline '
                            '(useful for production runs to show starting point)')
    parser.add_argument('--no-initial-validation', action='store_false', dest='initial_validation',
                       help='Disable initial validation (override checkpoint default)')
    parser.add_argument('--validation_only', action='store_true', dest='validation_only',
                       help='Run validation only (no training). Loads model, runs single validation pass, '
                            'logs to W&B if enabled, then exits. Useful for quickly computing baseline '
                            'perplexity without training. Incompatible with --resume_from_checkpoint.')
    parser.add_argument('--no-validation-only', action='store_false', dest='validation_only',
                       help='Disable validation-only mode (override checkpoint default)')
    parser.add_argument('--no_checkpoints', action='store_true', dest='no_checkpoints',
                       help='Disable all checkpoint saving (useful for debugging)')
    parser.add_argument('--checkpoints', action='store_false', dest='no_checkpoints',
                       help='Enable checkpoint saving (override checkpoint default)')

    # Qualitative evaluation
    parser.add_argument('--num_qualitative_samples', type=int, default=5,
                       help='Number of samples to generate text for during validation (default: 5, 0 = skip)')
    parser.add_argument('--max_generation_tokens', type=int, default=200,
                       help='Maximum tokens to generate during qualitative evaluation (default: 200)')

    # Weights & Biases integration
    parser.add_argument('--use_wandb', action='store_true', dest='use_wandb',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--no-use-wandb', action='store_false', dest='use_wandb',
                       help='Disable Weights & Biases logging (override checkpoint default)')
    parser.add_argument('--wandb_project', type=str,
                       default=os.getenv('WANDB_PROJECT', 'bad-autoencoding'),
                       help='W&B project name (default: bad-autoencoding, or WANDB_PROJECT env var)')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='W&B run name (default: auto-generated from regime and timestamp)')

    # Resume training
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--resume', type=str, default=None, dest='resume',
                       help='Simplified alias for --resume_from_checkpoint. '
                            'Automatically loads ALL configuration from checkpoint (regime, batch_size, etc.). '
                            'You can override safe settings with other flags (e.g., --eval-steps 100). '
                            'Example: python train.py --resume checkpoint.pt --eval-steps 250')
    parser.add_argument('--init_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint for two-stage training (loads only model weights, starts fresh optimization). '
                            'Use for reconstruction→LM fine-tuning. Requires --allow_objective_switch if objectives differ. '
                            'Example: --init_from_checkpoint stage1/best_checkpoint.pt --allow_objective_switch '
                            'Mutually exclusive with --resume_from_checkpoint.')
    parser.add_argument('--allow_objective_switch', action='store_true',
                       help='Allow objective switch when using --init_from_checkpoint (for two-stage training). '
                            'Example: Stage 1 used objective=reconstruction, Stage 2 uses objective=lm. '
                            'Without this flag, objective mismatch will cause an error.')
    parser.add_argument('--aux_loss_weight', type=float, default=0.5,
                       help='Weight for auxiliary losses in conv1d_residual_auxloss regime. '
                            '0.0 = only final loss, 1.0 = only auxiliary losses, 0.5 = equal weight. '
                            '(default: 0.5)')

    # DataLoader settings
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of DataLoader worker processes (0 = main process only)')
    parser.add_argument('--prefetch_factor', type=int, default=16,
                       help='Number of batches each worker pre-loads (default: 16). '
                            'Higher values hide data loading latency but use more CPU RAM. '
                            'Only applies when num_workers > 0')

    # Reproducibility
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducible data ordering (default: None = random). '
                            'Note: Only controls data shuffle order and weight initialization, '
                            'not CUDA operations (training remains fast but not bit-exact reproducible)')
    parser.add_argument('--eval_seed', type=int, default=42,
                       help='Seed for deterministic validation (ensures reproducible random subsampling). '
                            'Default: 42')
    parser.add_argument('--debug_log_sample_ids', action='store_true', dest='debug_log_sample_ids',
                       help='[DEBUG] Log sample IDs for each batch to verify checkpoint resume correctness. '
                            'Creates {output_dir}/sample_ids_log.jsonl with batch-level sample tracking.')
    parser.add_argument('--no-debug-log-sample-ids', action='store_false', dest='debug_log_sample_ids',
                       help='Disable debug sample ID logging (override checkpoint default)')

    # GPU
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--compile', action='store_true', dest='compile',
                       help='Compile model with torch.compile for 5-10%% speedup. '
                            'Note: First forward pass will take several minutes to compile. '
                            'Recommended for production runs with many training steps.')
    parser.add_argument('--no-compile', action='store_false', dest='compile',
                       help='Disable torch.compile (override checkpoint default)')
    parser.add_argument('--compile_mode', type=str, default='default',
                       choices=['default', 'reduce-overhead', 'max-autotune'],
                       help='torch.compile optimization mode (only used if --compile is set). '
                            'default: balanced compilation time and performance (2-5min compile, ~5-10%% speedup). '
                            'reduce-overhead: fast compilation, minimal speedup (1-2min compile, ~3-5%% speedup). '
                            'max-autotune: aggressive optimization for long runs (10-30min compile, ~10-20%% speedup). '
                            'Recommended: max-autotune for multi-day production runs. (default: default)')

    # Model optimization flags
    parser.add_argument('--use_optimized_model', action='store_true', dest='use_optimized_model',
                       help='Use training-optimized DeepSeek-OCR implementation with fixed graph breaks '
                            'and reduced device transfers. Enables better torch.compile support. '
                            'See CLAUDE.md for benchmarking results.')
    parser.add_argument('--no-use-optimized-model', action='store_false', dest='use_optimized_model',
                       help='Disable optimized model (override checkpoint default)')
    parser.add_argument('--use_encoder_checkpointing', action='store_true', dest='use_encoder_checkpointing',
                       help='Enable gradient checkpointing for vision encoder (SAM + CLIP + Projector). '
                            'Reduces memory usage by ~20-25%% but adds ~30%% compute overhead. '
                            'Useful for enabling larger batch sizes or encoder fine-tuning.')
    parser.add_argument('--no-use-encoder-checkpointing', action='store_false', dest='use_encoder_checkpointing',
                       help='Disable encoder checkpointing (override checkpoint default)')
    parser.add_argument('--use_decoder_checkpointing', action='store_true', dest='use_decoder_checkpointing',
                       help='Enable gradient checkpointing for decoder transformer layers (30 layers). '
                            'Reduces activation memory by ~30-50%% but adds ~15-20%% compute overhead. '
                            'Essential for enabling larger batch sizes when hitting OOM.')
    parser.add_argument('--no-use-decoder-checkpointing', action='store_false', dest='use_decoder_checkpointing',
                       help='Disable decoder checkpointing (override checkpoint default)')
    parser.add_argument('--use_8bit_optimizer', action='store_true', dest='use_8bit_optimizer',
                       help='Use 8-bit AdamW optimizer (bitsandbytes) instead of standard FP32 AdamW. '
                            'Reduces optimizer state memory by ~75%% (~16.8GB savings for 2.8B trainable params) '
                            'with minimal impact on convergence (<0.1%% perplexity difference). '
                            'Adds ~2-5%% compute overhead. Requires bitsandbytes library. '
                            'Mutually exclusive with fused AdamW (8-bit optimizer is preferred for VRAM-constrained training).')
    parser.add_argument('--no-use-8bit-optimizer', action='store_false', dest='use_8bit_optimizer',
                       help='Disable 8-bit optimizer (override checkpoint default)')

    return parser
