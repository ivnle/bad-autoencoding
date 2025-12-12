# Optical Context Compression Is Just (Bad) Autoencoding

[![arXiv](https://img.shields.io/badge/arXiv-2512.03643-b31b1b.svg)](https://arxiv.org/abs/2512.03643)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Models-yellow)](https://huggingface.co/ivnle/bad-autoencoding)

Code for reproducing experiments from our paper investigating whether vision-based context compression (as proposed in DeepSeek-OCR) can serve as an effective compression mechanism for LLM context.

- Model checkpoints *(coming soon)*
- [Documentation on the hype surrounding DeepSeek-OCR](HYPE.md)

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and requires Python 3.12+.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo
git clone https://github.com/ivnle/bad-autoencoding.git
cd bad-autoencoding
```

Dependencies are automatically installed on first `uv run` command.

## Quick Start

### 1. Prepare Data

**Disk space required:** ~920GB total (37GB download + 57GB filtered + 828GB rendered images)

See [data/README.md](data/README.md) for the full data preparation pipeline. In brief:

```bash
# 1. Download FineWiki dataset (~37GB)
uv run python data/scripts/download_finewiki.py

# 2. Filter articles to ≥2000 tokens (~57GB)
uv run python data/scripts/filter_finewiki.py

# 3. Chunk (1000+1000 tokens) and render images (~828GB for full dataset)
uv run python data/scripts/prepare_training_data.py \
    --input_jsonl data/filtered/finewiki_filtered.jsonl \
    --output_dir data/training/full \
    --num_workers 16

# 4. Create train/val splits
uv run python data/scripts/create_quick_splits.py \
    --input data/training/full/samples.jsonl \
    --output_dir data/training/full \
    --max_samples 510000 \
    --val_size 10000
```

### 2. Train

Model weights are automatically downloaded from HuggingFace (`deepseek-ai/DeepSeek-OCR`) on first run.

```bash
# Vision compression (main paper result)
uv run python train.py \
    --regime vision \
    --vision_mode small \
    --objective lm \
    --data_path ./data/training/full/train.jsonl \
    --val_data_path ./data/training/full/val.jsonl \
    --output_dir ./outputs/vision_small_lm \
    --use_wandb

# Text baseline (no compression)
uv run python train.py \
    --regime text \
    --objective lm \
    --data_path ./data/training/full/train.jsonl \
    --val_data_path ./data/training/full/val.jsonl \
    --output_dir ./outputs/text_baseline_lm \
    --use_wandb
```

See `scripts/run_example.sh` for more training configurations, or run `python train.py --help` for all options.

## Available Regimes

| Regime | Description |
|--------|-------------|
| `vision` | DeepSeek-OCR vision compression (renders text as image) |
| `text` | Text baseline (no compression) |
| `meanpool` | Sliding window mean pooling |
| `conv1d_residual` | Conv1d with residual skip connections |

### Text Regime Truncation

The `text` regime supports context truncation via `--text_context_tokens N`:
- Keeps the **last** N tokens of context (most recent)
- Default (omit flag) = full context, no truncation
- `--text_context_tokens 0` = empty context (target-only)

Example:
```bash
uv run python train.py \
    --regime text \
    --text_context_tokens 512 \
    --data_path ./data/training/full/train.jsonl \
    --val_data_path ./data/training/full/val.jsonl \
    --output_dir ./outputs/text_truncated_512
```

## Reproducing Paper Results

The paper results used these hyperparameters:

| Parameter | Value |
|-----------|-------|
| Effective batch size | 48 |
| Learning rate | 1e-4 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Max grad norm | 1.0 |
| Dataset | 510k samples (500k train + 10k val) |

To match effective batch size on your hardware:
- **80GB GPU (A100)**: `--batch_size 16 --gradient_accumulation_steps 3`
- **48GB GPU (A6000)**: `--batch_size 8 --gradient_accumulation_steps 6`
- **24GB GPU (4090)**: `--batch_size 1 --gradient_accumulation_steps 48`

The Quick Start examples use conservative settings (`batch_size=1`) for compatibility with smaller GPUs.

## Requirements

- **GPU**: 24-32GB VRAM recommended (memory optimizations enabled by default)
- **Python**: 3.12+
- **Dependencies**: Managed via `uv` (see `pyproject.toml`)

## Weights & Biases

Training logs to W&B by default when `--use_wandb` is passed. Configure with:

```bash
export WANDB_PROJECT="bad-autoencoding"  # Optional, defaults to bad-autoencoding
```

## Vendor Modifications

This repository includes a modified version of DeepSeek-OCR (`vendor/deepseek_ocr/`) with two changes:

1. **Trainable encoder support**: Allows gradient flow through the vision encoder (SAM + ViT + Projector) during training. Use `--train_encoder` flag.

2. **Conditional encoder loading**: Saves ~2GB GPU memory for text-only regimes by skipping vision encoder initialization when not needed.

## Citation

```bibtex
@misc{lee2025opticalcontextcompressionjust,
      title={Optical Context Compression Is Just (Bad) Autoencoding},
      author={Ivan Yee Lee and Cheng Yang and Taylor Berg-Kirkpatrick},
      year={2025},
      eprint={2512.03643},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.03643},
}
```

## License

This code is released under the MIT License. Note that the DeepSeek-OCR model weights have their own license - see [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) for details.
