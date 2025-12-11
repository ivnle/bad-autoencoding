"""Configuration constants and data classes for compression trainers."""

from dataclasses import dataclass


# Constants from source code analysis
IMAGE_TOKEN_ID = 128815

# Vision mode configurations
VISION_MODES = {
    'tiny': {'base_size': 512, 'image_size': 512, 'tokens': 73, 'crop_mode': False},
    'small': {'base_size': 640, 'image_size': 640, 'tokens': 111, 'crop_mode': False},
    'base': {'base_size': 1024, 'image_size': 1024, 'tokens': 273, 'crop_mode': False},
    'large': {'base_size': 1280, 'image_size': 1280, 'tokens': 421, 'crop_mode': False},
}

# Vision prompt presets (from DeepSeek-OCR README)
# All presets include leading '\n' and special tokens where appropriate
VISION_PROMPT_PRESETS = {
    'free_ocr': '\nFree OCR.',
    'ocr_grounding': '\n<|grounding|>OCR this image.',
    'markdown': '\n<|grounding|>Convert the document to markdown.',
    'parse_figure': '\nParse the figure.',
    'describe': '\nDescribe this image in detail.',
}

# Legacy constants (kept for backwards compatibility)
GUNDAM_PRESET = {'base_size': 1024, 'image_size': 640, 'crop_mode': True, 'tokens': 273}
VISION_TOKEN_COUNT = 273  # For Gundam preset without cropping

# Encoder component parameter prefixes (for freeze/unfreeze and param grouping)
# These must match the actual model architecture (DeepseekOCRForCausalLM -> DeepseekOCRModel)
ENCODER_COMPONENTS = ['model.sam_model', 'model.vision_model', 'model.projector']


@dataclass
class MemoryProfile:
    """Memory usage profile for a batch"""
    batch_size: int
    before_forward_mb: float
    after_forward_mb: float
    after_backward_mb: float
    peak_mb: float
