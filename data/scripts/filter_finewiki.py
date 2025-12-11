"""
Phase 1: Filter FineWiki Dataset

Filter FineWiki articles by token count and save with pre-computed tokens.

Output: JSONL file with:
- id: Article ID
- title: Article title
- text: Article text
- tokens: Pre-tokenized token IDs
- total_tokens: Token count

Usage:
    python filter_finewiki.py \
        --input_dir finewiki/data/enwiki \
        --output_file finewiki_filtered.jsonl \
        --min_tokens 2000
"""

import argparse
import glob
import json
import logging
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def filter_and_save_finewiki(
    input_dir: str,
    output_file: str,
    min_tokens: int = 2000,
    num_workers: int = 16,
    tokenizer_name: str = "deepseek-ai/DeepSeek-OCR"
):
    """
    Filter FineWiki dataset and save articles with pre-computed tokens.

    Loads entire dataset into memory for faster processing with parallelization.

    Args:
        input_dir: Path to FineWiki parquet files
        output_file: Output JSONL file path
        min_tokens: Minimum token count threshold
        num_workers: Number of parallel workers for tokenization
        tokenizer_name: HuggingFace tokenizer name
    """
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    logger.info(f"Loading FineWiki dataset from {input_dir}...")
    parquet_files = sorted(glob.glob(f"{input_dir}/*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files")

    if not parquet_files:
        raise ValueError(f"No parquet files found in {input_dir}")

    # Load entire dataset into memory (no streaming)
    logger.info("Loading all data into memory...")
    dataset = load_dataset(
        'parquet',
        data_files=parquet_files,
        split='train',
        streaming=False  # Load everything into RAM
    )

    total_articles = len(dataset)
    logger.info(f"Loaded {total_articles:,} articles into memory")
    logger.info(f"Filtering articles with ≥{min_tokens} tokens...")

    # Character pre-filter heuristic (1 token ≈ 3-4 chars)
    MIN_CHARS = min_tokens * 3

    def prefilter_and_tokenize(batch):
        """Pre-filter by character count, then batch tokenize survivors."""
        texts = []
        indices = []

        for idx, text in enumerate(batch['text']):
            # Quick character check
            if text and len(str(text).strip()) >= MIN_CHARS:
                texts.append(str(text))
                indices.append(idx)

        # Batch tokenize only candidates
        if texts:
            encodings = tokenizer(texts, add_special_tokens=False)

            # Add token IDs and counts to batch
            batch['tokens'] = [[] for _ in range(len(batch['text']))]
            batch['num_tokens'] = [0] * len(batch['text'])

            for i, idx in enumerate(indices):
                batch['tokens'][idx] = encodings['input_ids'][i]
                batch['num_tokens'][idx] = len(encodings['input_ids'][i])
        else:
            batch['tokens'] = [[] for _ in range(len(batch['text']))]
            batch['num_tokens'] = [0] * len(batch['text'])

        return batch

    # Apply batched tokenization with parallel workers
    logger.info(f"Tokenizing with {num_workers} workers...")
    dataset = dataset.map(
        prefilter_and_tokenize,
        batched=True,
        batch_size=1000,
        num_proc=num_workers,
        desc="Tokenizing"
    )

    # Filter for articles with sufficient tokens
    logger.info("Filtering by token count...")
    dataset = dataset.filter(
        lambda x: x['num_tokens'] >= min_tokens,
        num_proc=num_workers,
        desc="Filtering"
    )

    filtered_count = len(dataset)
    logger.info(f"Filtered to {filtered_count:,} articles ({filtered_count/total_articles*100:.1f}% kept)")

    # Save to JSONL
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving to {output_file}...")

    with open(output_file, 'w') as f:
        for idx, article in enumerate(tqdm(dataset, desc="Writing JSONL")):
            # Prepare article data
            article_data = {
                'id': str(article.get('id', article.get('page_id', 'unknown'))),
                'title': str(article['title']),
                'text': str(article['text']),
                'tokens': article['tokens'],
                'total_tokens': article['num_tokens']
            }

            # Write to JSONL
            f.write(json.dumps(article_data) + '\n')

            # Periodic flush to ensure data is saved
            if (idx + 1) % 1000 == 0:
                f.flush()

    logger.info(f"\nFiltering complete!")
    logger.info(f"  Total articles processed: {total_articles:,}")
    logger.info(f"  Total filtered articles: {filtered_count:,}")
    logger.info(f"  Output file: {output_file}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Filter FineWiki dataset by token count"
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        default='data/finewiki/data/enwiki',
        help='Path to FineWiki parquet files'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='data/filtered/finewiki_filtered.jsonl',
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--min_tokens',
        type=int,
        default=2000,
        help='Minimum token count threshold'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=16,
        help='Number of parallel workers for processing'
    )

    args = parser.parse_args()

    filter_and_save_finewiki(
        input_dir=args.input_dir,
        output_file=args.output_file,
        min_tokens=args.min_tokens,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()
