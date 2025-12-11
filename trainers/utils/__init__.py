"""Utility functions and classes for compression trainers."""

# Model loading
from .model import load_model_and_tokenizer

# Freezing/unfreezing and parameter management
from .freezing import (
    freeze_vision_encoder,
    unfreeze_vision_encoder,
    get_encoder_params,
    get_decoder_params,
    unfreeze_decoder,
    count_parameters
)

# Generation
from .generation import manual_generate_with_cache

# Image processing
from .image import BasicImageTransform, create_dummy_image

# Memory profiling
from .memory import (
    get_gpu_memory_mb,
    get_peak_gpu_memory_mb,
    reset_peak_memory,
    validate_gradients
)

__all__ = [
    # Model loading
    'load_model_and_tokenizer',
    # Freezing/unfreezing
    'freeze_vision_encoder',
    'unfreeze_vision_encoder',
    'get_encoder_params',
    'get_decoder_params',
    'unfreeze_decoder',
    'count_parameters',
    # Generation
    'manual_generate_with_cache',
    # Image processing
    'BasicImageTransform',
    'create_dummy_image',
    # Memory profiling
    'get_gpu_memory_mb',
    'get_peak_gpu_memory_mb',
    'reset_peak_memory',
    'validate_gradients',
]
