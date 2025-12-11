"""
Phase 2: Training Data Preparation - Rendering & Chunking

Generate training dataset with pre-rendered images at 4 compression modes:
- Tiny: 512×512 → 64 tokens (~15.6x compression)
- Small: 640×640 → 100 tokens (~10.0x compression)
- Base: 1024×1024 → 256 tokens (~3.9x compression)
- Large: 1280×1280 → 400 tokens (~2.5x compression)

Pipeline:
1. Stream pre-filtered articles from Phase 1 (with pre-computed tokens)
2. Extract non-overlapping 2000-token chunks (1000 context + 1000 continuation)
3. Render context text as images at 4 resolutions in parallel
4. Save to samples.jsonl with:
   - Token IDs for text regime training
   - Image paths for vision regime training
   - Context text for sanity checking

Features:
- Streaming batch processing (constant memory usage)
- Checkpoint-based resume capability (article-level granularity)
- Parallel image rendering with ProcessPoolExecutor
- Sequential sample ID assignment (prevents duplicates)

Phase 1 (filter_finewiki.py) produces: finewiki_filtered.jsonl
Phase 2 (this script) produces: samples.jsonl + images/
Phase 3 (create_quick_splits.py) creates: train.jsonl + val.jsonl
"""

import argparse
import json
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Vision mode configurations
VISION_MODES = {
    'tiny': {'resolution': 512, 'tokens': 73},
    'small': {'resolution': 640, 'tokens': 111},
    'base': {'resolution': 1024, 'tokens': 273},
    'large': {'resolution': 1280, 'tokens': 421},
}

# Fixed context/continuation lengths
CONTEXT_LENGTH = 1000
CONTINUATION_LENGTH = 1000
MIN_ARTICLE_TOKENS = CONTEXT_LENGTH + CONTINUATION_LENGTH  # 2000

# Font cache for performance (avoid repeated disk I/O)
_FONT_CACHE = {}

# Worker tokenizer cache (loaded once per worker process)
_worker_tokenizer = None


def init_worker(tokenizer_name: str):
    """
    Initialize worker process with tokenizer (called once per worker).

    This avoids reloading the tokenizer for every article (961K times),
    reducing disk I/O and initialization overhead.

    Args:
        tokenizer_name: HuggingFace tokenizer identifier
    """
    global _worker_tokenizer
    _worker_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True
    )


def get_cached_font(size: int) -> ImageFont.FreeTypeFont:
    """
    Get font from cache, loading from disk only if needed.

    Args:
        size: Font size in pixels

    Returns:
        PIL FreeTypeFont object
    """
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                size
            )
        except Exception:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def wrap_text_accurate(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """
    Wrap text using exact font metrics (font.getbbox()).

    Args:
        text: Text to wrap
        font: PIL font object
        max_width: Maximum width in pixels

    Returns:
        List of wrapped lines
    """
    words = text.split()
    lines = []
    current_line = []

    for word in words:
        test_line = ' '.join(current_line + [word])
        # Use exact font metrics to measure pixel width
        bbox = font.getbbox(test_line)
        text_width = bbox[2] - bbox[0]

        if text_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]

    if current_line:
        lines.append(' '.join(current_line))

    return lines


def render_text_to_image(text: str, width: int, height: int) -> Tuple[Image.Image, Dict]:
    """
    Render text to image with adaptive font sizing.

    Uses binary search to find optimal font size that fits all text.

    Args:
        text: Text to render
        width: Image width
        height: Image height

    Returns:
        Tuple of (PIL Image with rendered text, metadata dict)
        Metadata contains: font_size, total_lines, rendered_lines, is_truncated
    """
    # Create white background
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)

    # Proportional margins (5% of image height)
    margin = int(height * 0.05)
    max_width = width - 2 * margin
    available_height = height - 2 * margin

    # Helper function to check if text fits at a given font size
    def text_fits_at_size(size: int) -> bool:
        font = get_cached_font(size)
        line_height = int(size * 1.2)
        lines = wrap_text_accurate(text, font, max_width)
        total_height = len(lines) * line_height
        return total_height <= available_height

    # Binary search to find largest font size that fits
    min_font_size = 4
    max_font_size = 30
    low = min_font_size
    high = max_font_size

    # Binary search for rightmost (largest) fitting size
    while low < high:
        mid = (low + high + 1) // 2  # Round up to bias toward larger sizes
        if text_fits_at_size(mid):
            low = mid  # mid fits, try larger
        else:
            high = mid - 1  # mid doesn't fit, try smaller

    font_size = low

    # Get final font and wrap text for rendering
    font = get_cached_font(font_size)
    line_height = int(font_size * 1.2)
    lines = wrap_text_accurate(text, font, max_width)

    # Warn if we ended up at minimum font size
    if font_size == min_font_size:
        logger.warning(f"Text truncated: Required font size at minimum ({min_font_size}px)")

    # Draw lines and track how many we actually render
    total_lines = len(lines)
    rendered_lines = 0
    y_offset = margin

    for line in lines:
        if y_offset + line_height > height - margin:
            break  # Stop if we run out of space
        draw.text((margin, y_offset), line, fill='black', font=font)
        y_offset += line_height
        rendered_lines += 1

    # Prepare metadata
    is_truncated = (rendered_lines < total_lines)
    metadata = {
        'font_size': font_size,
        'total_lines': total_lines,
        'rendered_lines': rendered_lines,
        'is_truncated': is_truncated
    }

    if is_truncated:
        logger.warning(f"Text truncated: {rendered_lines}/{total_lines} lines rendered at {width}x{height}")

    return image, metadata


def load_processed_articles(checkpoint_file: Path) -> set:
    """
    Load set of already-processed article IDs from checkpoint file.

    Args:
        checkpoint_file: Path to checkpoint file containing article IDs

    Returns:
        Set of processed article IDs
    """
    if not checkpoint_file.exists():
        return set()

    with open(checkpoint_file, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def mark_article_processed(checkpoint_file: Path, article_id: str):
    """
    Append article ID to checkpoint file.

    Uses append mode which is atomic for small writes on POSIX systems.
    The file is opened, written, and closed atomically for each article.

    Args:
        checkpoint_file: Path to checkpoint file
        article_id: Article ID to mark as processed
    """
    # Write to checkpoint file (append mode is atomic on POSIX for small writes)
    with open(checkpoint_file, 'a') as f:
        f.write(f"{article_id}\n")


def clean_partial_jsonl(jsonl_file: Path):
    """
    Remove incomplete last line from JSONL file if interrupted mid-write.

    Uses seek-from-end approach to avoid loading entire file into memory.
    Only reads up to 64KB from the end to find the last line.

    Args:
        jsonl_file: Path to JSONL file to clean
    """
    if not jsonl_file.exists() or jsonl_file.stat().st_size == 0:
        return

    try:
        with open(jsonl_file, 'rb+') as f:
            # Get file size
            f.seek(0, 2)  # Seek to end
            file_size = f.tell()

            if file_size == 0:
                return

            # Read last chunk (up to 64KB or entire file if smaller)
            chunk_size = min(65536, file_size)
            f.seek(-chunk_size, 2)  # Seek to chunk_size bytes before end
            chunk = f.read(chunk_size)

            # Find the last line
            # If file ends with newline, last "line" is empty - we want second-to-last
            lines = chunk.split(b'\n')

            # Determine which line to validate
            if lines[-1]:
                # No trailing newline - validate last line
                last_line = lines[-1]
                truncate_to_line_count = len(lines) - 1
            elif len(lines) >= 2:
                # Trailing newline - validate second-to-last line
                last_line = lines[-2]
                truncate_to_line_count = len(lines) - 2
            else:
                # File is just a newline
                return

            # Try to parse last line as JSON
            try:
                json.loads(last_line.decode('utf-8'))
                # Last line is valid, nothing to clean
                return
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Last line is incomplete, truncate file to remove it
                logger.warning(f"Detected incomplete line in {jsonl_file}, removing it")

                # Calculate truncation position
                # Sum of lengths of all lines we want to keep, plus newlines between them
                truncate_pos = file_size - chunk_size + sum(len(line) + 1 for line in lines[:truncate_to_line_count])

                # Truncate file
                f.seek(truncate_pos)
                f.truncate()
                logger.info(f"Cleaned {jsonl_file} (truncated to position {truncate_pos})")

    except Exception as e:
        logger.warning(f"Could not clean {jsonl_file}: {e}")


def clean_temp_files(output_dir: Path):
    """
    Remove temporary image files from incomplete parallel processing.

    Temp files are created during parallel processing but not renamed if the
    process crashes before completion. They start with "temp_" prefix.

    Args:
        output_dir: Output directory containing images/
    """
    images_dir = output_dir / 'images'
    if not images_dir.exists():
        return

    temp_files_removed = 0
    for mode_name in VISION_MODES.keys():
        mode_dir = images_dir / mode_name
        if not mode_dir.exists():
            continue

        # Find and remove temp files
        for temp_file in mode_dir.glob('temp_*.png'):
            try:
                temp_file.unlink()
                temp_files_removed += 1
            except Exception as e:
                logger.warning(f"Could not remove temp file {temp_file}: {e}")

    if temp_files_removed > 0:
        logger.info(f"Cleaned {temp_files_removed} temporary image files from incomplete processing")


def load_filtered_dataset_streaming(
    filtered_jsonl: str,
    checkpoint_file: Path,
    batch_size: int = 1000,
    num_articles: int = None,
    seed: int = 42
):
    """
    Stream articles in batches, automatically skipping already-processed ones.

    This is a generator that yields batches of unprocessed articles,
    enabling constant memory usage regardless of dataset size.

    Args:
        filtered_jsonl: Path to filtered JSONL file from Phase 2
        checkpoint_file: Path to checkpoint file with processed article IDs
        batch_size: Number of articles per batch
        num_articles: Maximum number of articles to process (None = all)
        seed: (Deprecated) No longer used since shuffling was removed

    Yields:
        List of article dictionaries (batch)
    """
    # Load already-processed articles
    processed = load_processed_articles(checkpoint_file)

    if processed:
        logger.info(f"Found {len(processed)} already-processed articles, skipping them...")

    # Calculate how many NEW articles to process (accounting for already-processed)
    remaining_articles = None
    if num_articles is not None:
        remaining_articles = num_articles - len(processed)
        if remaining_articles < 0:
            remaining_articles = 0

    batch = []
    articles_yielded = 0
    articles_skipped = 0

    # Stream articles (optionally limited by num_articles)
    if num_articles is not None:
        if processed:
            logger.info(f"Streaming up to {remaining_articles} more articles (target: {num_articles} total, {len(processed)} already processed)...")
        else:
            logger.info(f"Streaming up to {num_articles} unprocessed articles...")
    else:
        logger.info("Streaming all unprocessed articles...")

    with open(filtered_jsonl, 'r') as f:
        for line in f:
            article = json.loads(line)
            article_id = article['id']

            # Skip if already processed
            if article_id in processed:
                articles_skipped += 1
                continue

            # Stop if we've reached the limit (check BEFORE appending)
            if remaining_articles is not None and articles_yielded >= remaining_articles:
                break

            batch.append(article)
            articles_yielded += 1

            if len(batch) >= batch_size:
                yield batch
                batch = []

    # Yield remaining articles
    if batch:
        yield batch

    logger.info(f"Streamed {articles_yielded} articles in total")
    if articles_skipped > 0:
        logger.info(f"Skipped {articles_skipped} already-processed articles")


def process_article_chunk(
    article: Dict,
    temp_id: str,
    chunk_index: int,
    chunk_start: int,
    tokenizer,
    output_dir: Path
) -> Dict:
    """
    Process a single chunk from an article:
    - Extract chunk tokens (2000 tokens: 1000 context + 1000 continuation)
    - Render context at 4 resolutions
    - Return metadata

    Note: Uses temporary ID during parallel processing. Final sample_id will be
    assigned sequentially after successful completion.

    Args:
        article: Article dictionary with pre-computed tokens from Phase 1
        temp_id: Temporary ID string for filename (will be renamed later)
        chunk_index: Chunk index within article (0, 1, 2, ...)
        chunk_start: Starting token position for this chunk
        tokenizer: HuggingFace tokenizer (for decoding only)
        output_dir: Output directory

    Returns:
        Sample metadata dictionary
    """
    # Use pre-computed tokens from Phase 1 (no re-tokenization!)
    tokens = article['tokens']

    # Extract chunk (2000 tokens starting at chunk_start)
    context_start = chunk_start
    context_end = chunk_start + CONTEXT_LENGTH
    continuation_start = context_end
    continuation_end = continuation_start + CONTINUATION_LENGTH

    context_tokens = tokens[context_start:context_end]
    continuation_tokens = tokens[continuation_start:continuation_end]

    # Validate token counts (fail fast if article tokens are incorrect)
    if len(context_tokens) != CONTEXT_LENGTH or len(continuation_tokens) != CONTINUATION_LENGTH:
        logger.warning(
            f"Skipping invalid chunk from article '{article.get('title', 'unknown')}': "
            f"expected {CONTEXT_LENGTH}+{CONTINUATION_LENGTH} tokens, "
            f"got {len(context_tokens)}+{len(continuation_tokens)}"
        )
        return None

    # Decode context for rendering (preserve exact whitespace and skip special tokens)
    context_text = tokenizer.decode(
        context_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    # Render context at 4 resolutions and collect metadata
    # Use temp_id (will be renamed to final sample_id after successful completion)
    sample_name = f"temp_{temp_id}_chunk_{chunk_index}"
    font_metadata = {}
    image_paths = {}

    for mode_name, config in VISION_MODES.items():
        resolution = config['resolution']

        # Render image and get metadata
        image, metadata = render_text_to_image(context_text, resolution, resolution)

        # Store font metadata for this resolution
        font_metadata[mode_name] = metadata

        # Save image
        mode_dir = output_dir / 'images' / mode_name
        mode_dir.mkdir(parents=True, exist_ok=True)

        image_path = mode_dir / f"{sample_name}.png"
        image.save(image_path, optimize=True)

        # Store relative path for JSONL
        image_paths[mode_name] = f"images/{mode_name}/{sample_name}.png"

    # Return metadata (output format for both text and vision regimes)
    return {
        'id': sample_name,
        'wiki_id': article['id'],
        'title': article['title'],
        'chunk_index': chunk_index,

        # Shared fields for both regimes
        'context_tokens': context_tokens,
        'continuation_tokens': continuation_tokens,
        'context_text': context_text,  # For sanity checking

        # Vision regime (image paths)
        'images': image_paths,

        # Metadata
        'total_article_tokens': article['total_tokens'],
        'font_metadata': font_metadata
    }


def extract_samples_from_article(
    article: Dict,
    temp_id: str,
    tokenizer,
    output_dir: Path
) -> List[Dict]:
    """
    Extract multiple non-overlapping 2000-token samples from a single article.

    Uses temporary ID during parallel processing. Final sample_id will be assigned
    sequentially after successful completion.

    Examples:
    - Article with 2000 tokens → 1 sample (tokens 0-2000)
    - Article with 4000 tokens → 2 samples (tokens 0-2000, 2000-4000)
    - Article with 6000 tokens → 3 samples (tokens 0-2000, 2000-4000, 4000-6000)

    Args:
        article: Article dictionary
        temp_id: Temporary ID string for this article
        tokenizer: HuggingFace tokenizer
        output_dir: Output directory

    Returns:
        List of sample metadata dictionaries (with temp IDs)
    """
    total_tokens = article['total_tokens']
    chunk_size = MIN_ARTICLE_TOKENS  # 2000 tokens per chunk
    num_chunks = total_tokens // chunk_size

    samples = []
    for chunk_index in range(num_chunks):
        chunk_start = chunk_index * chunk_size
        sample = process_article_chunk(
            article=article,
            temp_id=temp_id,
            chunk_index=chunk_index,
            chunk_start=chunk_start,
            tokenizer=tokenizer,
            output_dir=output_dir
        )
        # Skip invalid chunks (process_article_chunk returns None for invalid chunks)
        if sample is not None:
            samples.append(sample)

    return samples


def rename_sample_files(samples: List[Dict], temp_id: str, real_sample_id: int, output_dir: Path) -> List[Dict]:
    """
    Rename image files from temporary ID to final sample_id and update metadata.

    This is called after successful article processing to assign sequential sample_ids.

    Args:
        samples: List of sample dicts with temp_id in filenames
        temp_id: Temporary ID string used during parallel processing
        real_sample_id: Final sequential sample_id to use
        output_dir: Output directory containing images/

    Returns:
        Updated samples with renamed files and corrected metadata
    """
    updated_samples = []

    for sample in samples:
        chunk_index = sample['chunk_index']

        # Update sample ID in metadata
        old_sample_name = f"temp_{temp_id}_chunk_{chunk_index}"
        new_sample_name = f"sample_{real_sample_id:06d}_chunk_{chunk_index}"
        sample['id'] = new_sample_name

        # Rename image files for each resolution mode
        new_image_paths = {}
        for mode_name, old_rel_path in sample['images'].items():
            # Construct absolute paths
            old_path = output_dir / old_rel_path
            new_filename = f"{new_sample_name}.png"
            new_path = old_path.parent / new_filename

            # Rename file (fail fast if temp file is missing)
            if not old_path.exists():
                raise FileNotFoundError(
                    f"Temporary image file not found: {old_path}. "
                    f"This indicates incomplete parallel processing for sample {new_sample_name}"
                )
            old_path.rename(new_path)

            # Update relative path in metadata
            new_image_paths[mode_name] = f"images/{mode_name}/{new_filename}"

        sample['images'] = new_image_paths
        updated_samples.append(sample)

    return updated_samples


# Note: Train/val splitting moved to separate create_quick_splits.py script
# This allows for flexible post-processing without regenerating images


def process_article_wrapper(args: Tuple) -> Tuple[str, Dict, List[Dict]]:
    """
    Wrapper function for multiprocessing to process a single article.

    Uses the global _worker_tokenizer initialized by init_worker().
    Returns temp_id, article, and samples for sequential ID assignment in main thread.

    Args:
        args: Tuple of (temp_id, article, output_dir)

    Returns:
        Tuple of (temp_id, article_dict, samples_list)
        Returns empty list for samples if processing fails
    """
    temp_id, article, output_dir = args
    global _worker_tokenizer

    try:
        samples = extract_samples_from_article(article, temp_id, _worker_tokenizer, output_dir)
        return (temp_id, article, samples)
    except Exception as e:
        logger.warning(f"Failed to process article '{article.get('title', 'unknown')}'): {e}")
        return (temp_id, article, [])


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Render training data from filtered articles"
    )
    parser.add_argument(
        '--input_jsonl',
        type=str,
        required=True,
        help='Path to filtered JSONL from Phase 1 (e.g., finewiki_filtered.jsonl)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/training',
        help='Output directory for rendered images and JSONL'
    )
    parser.add_argument(
        '--num_articles',
        type=int,
        default=None,
        help='Number of articles to process (None = all). '
             'Uses streaming (constant memory, sequential order).'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='(Deprecated) No longer used since shuffling was removed'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=16,
        help='Number of parallel workers for rendering'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Clear and overwrite existing output directory if not empty'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='Number of articles to process per batch (default: 1000, affects memory and checkpoint granularity)'
    )

    args = parser.parse_args()

    # Setup paths
    output_dir = Path(args.output_dir)
    samples_file = output_dir / 'samples.jsonl'
    checkpoint_file = output_dir / '.processed_articles.txt'

    # Check for resume scenario
    is_resuming = checkpoint_file.exists()

    if is_resuming:
        processed_count = len(load_processed_articles(checkpoint_file))
        logger.info("=" * 80)
        logger.info("RESUME DETECTED")
        logger.info("=" * 80)
        logger.info(f"Found checkpoint with {processed_count} already-processed articles")
        logger.info(f"Samples will be appended to: {samples_file}")
        logger.info(f"Images directory: {output_dir / 'images'}")
        logger.info("")

        # Clean partial JSONL if necessary
        clean_partial_jsonl(samples_file)

        # Clean temporary image files from incomplete parallel processing
        clean_temp_files(output_dir)

        response = input("Do you want to resume from checkpoint? [y/N]: ")
        if response.lower() != 'y':
            logger.info("Resume cancelled by user")
            exit(0)
        logger.info("Resuming from checkpoint...")
        logger.info("=" * 80)
    else:
        # Check if directory exists and handle overwrite
        if output_dir.exists() and any(output_dir.iterdir()):
            if not args.overwrite:
                logger.error(f"Output directory is not empty: {output_dir}")
                logger.error("Contents will be cleared and replaced.")
                logger.error("Use --overwrite flag to confirm you want to proceed.")
                exit(1)

            # Clear directory
            logger.warning(f"Clearing existing output directory: {output_dir}")
            shutil.rmtree(output_dir)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting fresh dataset generation in: {output_dir}")

    # Load tokenizer (only for decoding text, not tokenizing)
    logger.info("Loading tokenizer...")
    tokenizer_name = "deepseek-ai/DeepSeek-OCR"

    # Initialize counters (count existing articles if resuming)
    total_articles_processed = 0  # For sample_id offset (article index)
    total_samples_generated = 0   # For logging only (actual sample count)

    if is_resuming and checkpoint_file.exists():
        # Count processed articles from checkpoint to continue sample_id numbering
        # Note: sample_id represents article index, not sample count
        logger.info("Counting processed articles for resume...")
        with open(checkpoint_file, 'r') as f:
            num_processed_articles = sum(1 for _ in f)
        total_articles_processed = num_processed_articles  # sample_id = article_index
        logger.info(f"Found {num_processed_articles} processed articles, will continue from sample_{total_articles_processed:06d}")

    logger.info(f"Streaming articles from {args.input_jsonl}...")
    logger.info(f"Batch size: {args.batch_size} articles")
    logger.info(f"Workers per batch: {args.num_workers}")

    # Process batches
    batch_generator = load_filtered_dataset_streaming(
        filtered_jsonl=args.input_jsonl,
        checkpoint_file=checkpoint_file,
        batch_size=args.batch_size,
        num_articles=args.num_articles,
        seed=args.seed
    )

    # Create ProcessPoolExecutor once for all batches (not per-batch)
    # Workers are initialized with tokenizer to avoid repeated loading
    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=init_worker,
        initargs=(tokenizer_name,)
    ) as executor:
        for batch_idx, article_batch in enumerate(batch_generator):
            logger.info(f"Processing batch {batch_idx}: {len(article_batch)} articles")

            # Generate temporary IDs for parallel processing
            # Final sample_ids will be assigned sequentially as articles complete
            article_args = [
                (f"{batch_idx}_{idx}", article, output_dir)
                for idx, article in enumerate(article_batch)
            ]

            # Process batch in parallel with incremental writing
            batch_sample_count = 0
            batch_article_count = 0
            # Submit all tasks
            futures = [
                executor.submit(process_article_wrapper, args)
                for args in article_args
            ]

            # Open samples file for incremental writing
            with open(samples_file, 'a') as f:
                # Process results as they complete
                for future in tqdm(
                    as_completed(futures),
                    total=len(article_batch),
                    desc=f"Batch {batch_idx}",
                    leave=False
                ):
                    try:
                        # Unpack result
                        temp_id, article, article_samples = future.result()

                        # Skip articles that failed or produced no samples
                        # BUT still checkpoint them to prevent infinite reprocessing
                        if not article_samples:
                            mark_article_processed(checkpoint_file, article['id'])
                            logger.debug(
                                f"Article '{article.get('title', 'unknown')}' produced no samples "
                                "(too short or all chunks invalid), checkpointed to prevent reprocessing"
                            )
                            continue

                        # Assign sequential sample_id (critical: only successful articles get IDs)
                        real_sample_id = total_articles_processed

                        # Rename files from temp_id to real sample_id
                        renamed_samples = rename_sample_files(
                            article_samples, temp_id, real_sample_id, output_dir
                        )

                        # Checkpoint article FIRST to prevent duplicates on crash
                        # Tradeoff: If crash before flush, we lose this article's samples
                        # but avoid duplicate sample IDs (more critical issue)
                        mark_article_processed(checkpoint_file, article['id'])

                        # Write samples immediately (no accumulation)
                        for sample in renamed_samples:
                            f.write(json.dumps(sample) + '\n')

                        # Flush to disk after checkpoint
                        f.flush()

                        # Update counters (only successful articles increment article count)
                        batch_sample_count += len(renamed_samples)
                        batch_article_count += 1
                        total_articles_processed += 1  # Increment immediately for next article

                    except Exception as e:
                        logger.warning(f"Failed to process article: {e}")
                        # Don't checkpoint failed articles - they'll be retried on resume
                        continue

            # Update global sample counter
            # Note: total_articles_processed is incremented in the loop above
            total_samples_generated += batch_sample_count    # For logging

            logger.info(
                f"Batch {batch_idx} complete: "
                f"{batch_sample_count} samples generated from {batch_article_count} articles. "
                f"Total: {total_articles_processed} articles, {total_samples_generated} samples"
            )

    logger.info("=" * 80)
    logger.info(f"Processing complete!")
    logger.info(f"Total articles processed: {total_articles_processed}")
    logger.info(f"Total samples generated: {total_samples_generated}")
    logger.info(f"Output: {samples_file}")
    logger.info("=" * 80)
    logger.info("")
    logger.info("Dataset preparation complete!")
    logger.info("")
    logger.info("Next step:")
    logger.info(f"  uv run python data/scripts/create_quick_splits.py --input {samples_file} --output_dir {output_dir}")
    logger.info("")


if __name__ == '__main__':
    main()
