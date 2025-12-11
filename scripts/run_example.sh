#!/bin/bash
# Example training commands for bad-autoencoding
# See python train.py --help for all available options

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data paths (REQUIRED - modify these for your setup)
# After running the data preparation pipeline, your data will be here:
DATA_PATH="./data/training/full/train.jsonl"
VAL_DATA_PATH="./data/training/full/val.jsonl"
OUTPUT_DIR="./outputs"

# Optional: Weights & Biases (comment out to disable)
# export WANDB_PROJECT="bad-autoencoding"

# =============================================================================
# EXAMPLE 1: Vision Compression (Main Paper Result)
# =============================================================================
# Train with vision compression using rendered text images
# GPU: ~24-32GB recommended

train_vision() {
    python train.py \
        --regime vision \
        --vision_mode small \
        --objective lm \
        --data_path "$DATA_PATH" \
        --val_data_path "$VAL_DATA_PATH" \
        --output_dir "$OUTPUT_DIR/vision_small_lm" \
        --batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-4 \
        --num_epochs 1 \
        --eval_steps 500 \
        --save_steps 1000 \
        --use_wandb
}

# =============================================================================
# EXAMPLE 2: Text Baseline
# =============================================================================
# Standard text-only training (no compression)

train_text_baseline() {
    python train.py \
        --regime text \
        --objective lm \
        --data_path "$DATA_PATH" \
        --val_data_path "$VAL_DATA_PATH" \
        --output_dir "$OUTPUT_DIR/text_baseline_lm" \
        --batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-4 \
        --num_epochs 1 \
        --eval_steps 500 \
        --save_steps 1000 \
        --use_wandb
}

# =============================================================================
# EXAMPLE 3: Conv1d Residual Compression
# =============================================================================
# Learned compression with conv1d pyramid + residual connections

train_conv1d_residual() {
    python train.py \
        --regime conv1d_residual \
        --compression_target 111 \
        --objective lm \
        --data_path "$DATA_PATH" \
        --val_data_path "$VAL_DATA_PATH" \
        --output_dir "$OUTPUT_DIR/conv1d_residual_lm" \
        --batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-4 \
        --num_epochs 1 \
        --eval_steps 500 \
        --save_steps 1000 \
        --use_wandb
}

# =============================================================================
# EXAMPLE 4: Mean Pool Compression
# =============================================================================
# Simple mean pooling baseline

train_meanpool() {
    python train.py \
        --regime meanpool \
        --compression_window_size 9 \
        --compression_stride 9 \
        --objective lm \
        --data_path "$DATA_PATH" \
        --val_data_path "$VAL_DATA_PATH" \
        --output_dir "$OUTPUT_DIR/meanpool_lm" \
        --batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 1e-4 \
        --num_epochs 1 \
        --eval_steps 500 \
        --save_steps 1000 \
        --use_wandb
}

# =============================================================================
# EXAMPLE 5: Debug Mode (Quick Test)
# =============================================================================
# Small subset for testing setup

train_debug() {
    python train.py \
        --regime vision \
        --vision_mode small \
        --objective lm \
        --data_path "$DATA_PATH" \
        --val_data_path "$VAL_DATA_PATH" \
        --output_dir "$OUTPUT_DIR/debug" \
        --max_samples 100 \
        --batch_size 1 \
        --gradient_accumulation_steps 2 \
        --learning_rate 1e-4 \
        --num_epochs 1 \
        --eval_steps 10 \
        --save_steps 0 \
        --log_steps 5
}

# =============================================================================
# USAGE
# =============================================================================
# Uncomment one of the following to run:

# train_vision
# train_text_baseline
# train_conv1d_residual
# train_meanpool
# train_debug

echo "Edit this script to uncomment the training function you want to run."
echo "See python train.py --help for all available options."
