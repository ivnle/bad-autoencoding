# Data Preparation Pipeline

This directory contains scripts for preparing training data from FineWiki (English Wikipedia).

## Disk Space Requirements

| Phase | Output Size | Cumulative |
|-------|-------------|------------|
| 1. Download | ~37GB | ~37GB |
| 2. Filter | ~57GB | ~94GB |
| 3. Render | ~828GB | ~922GB |
| 4. Split | <1MB | ~922GB |

**Total: ~920GB** (can delete intermediate files after each phase to reduce)

## Pipeline Overview

```
download_finewiki.py → filter_finewiki.py → prepare_training_data.py → create_quick_splits.py
     (Phase 1)             (Phase 2)              (Phase 3)                  (Phase 4)
```

## Phase 1: Download FineWiki

**Script:** `scripts/download_finewiki.py`

Downloads the FineWiki English Wikipedia subset from HuggingFace (~6M articles).

```bash
uv run python data/scripts/download_finewiki.py
```

**Output:** `data/finewiki/data/enwiki/*.parquet` (~37GB)

## Phase 2: Filter & Tokenize

**Script:** `scripts/filter_finewiki.py`

Filters articles to ≥2000 tokens and pre-computes token IDs using the DeepSeek-OCR tokenizer.

```bash
uv run python data/scripts/filter_finewiki.py
```

**Input:** `data/finewiki/data/enwiki/*.parquet` (default)
**Output:** `data/filtered/finewiki_filtered.jsonl` (~1M articles, ~57GB)

**Output format:**
```json
{
  "id": "enwiki/34833",
  "title": "120s",
  "text": "# 120s\nThe 120s was a decade...",
  "tokens": [5, 223, 4870, ...],
  "total_tokens": 2969
}
```

## Phase 3: Chunk & Render Images

**Script:** `scripts/prepare_training_data.py`

Extracts 2000-token chunks (1000 context + 1000 continuation) and renders context text as images at 4 resolutions.

```bash
uv run python data/scripts/prepare_training_data.py \
    --input_jsonl data/filtered/finewiki_filtered.jsonl \
    --output_dir data/training/full \
    --num_workers 16
```

**Input:** `data/filtered/finewiki_filtered.jsonl`
**Output:** `data/training/full/samples.jsonl` + `data/training/full/images/`

**Output format:**
```json
{
  "id": "sample_000001_chunk_0",
  "context_tokens": [5, 223, ...],
  "continuation_tokens": [1167, 260, ...],
  "context_text": "# 120s\nThe 120s...",
  "images": {
    "tiny": "images/tiny/sample_000001_chunk_0.png",
    "small": "images/small/sample_000001_chunk_0.png",
    "base": "images/base/sample_000001_chunk_0.png",
    "large": "images/large/sample_000001_chunk_0.png"
  }
}
```

## Phase 4: Create Train/Val Splits

**Script:** `scripts/create_quick_splits.py`

Creates reproducible train/validation splits from the rendered samples.

```bash
uv run python data/scripts/create_quick_splits.py \
    --input data/training/full/samples.jsonl \
    --output_dir data/training/full \
    --max_samples 510000 \
    --val_size 10000
```

**Input:** `data/training/full/samples.jsonl`
**Output:**
- `data/training/full/train.jsonl` (500K samples)
- `data/training/full/val.jsonl` (10K samples)
- `data/training/full/split_manifest.json` (metadata)

## Vision Compression Modes

| Mode  | Resolution | Vision Tokens | Compression Ratio |
|-------|------------|---------------|-------------------|
| tiny  | 512×512    | 73            | ~13.7x            |
| small | 640×640    | 111           | ~9.0x             |
| base  | 1024×1024  | 273           | ~3.7x             |
| large | 1280×1280  | 421           | ~2.4x             |

## Training Regimes

- **Text regime:** Uses `context_tokens` + `continuation_tokens` directly (no compression)
- **Vision regime:** Uses rendered image + `continuation_tokens` (visual compression)

## Optional: Convert to Arrow Format

For memory-constrained environments, convert JSONL to Arrow format to reduce per-experiment memory from ~40GB to ~100MB:

```bash
uv run python data/scripts/convert_to_arrow.py \
    --input data/training/full/train.jsonl \
    --output data/training/full/train_arrow

uv run python data/scripts/convert_to_arrow.py \
    --input data/training/full/val.jsonl \
    --output data/training/full/val_arrow
```

Then use `--data_path data/training/full/train_arrow` in training commands.
