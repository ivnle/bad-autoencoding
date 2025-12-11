"""Compression trainers for DeepSeek-OCR experiments."""

# Import all trainer classes
from .vision import VisionCompressionTrainer
from .text import TextBaselineTrainer
from .subsample import SubsampleCompressionTrainer
from .meanpool import MeanPoolCompressionTrainer
from .randproj import RandomProjectionCompressionTrainer
from .conv1d import Conv1dPyramidCompressionTrainer
from .conv1d_residual import Conv1dResidualCompressionTrainer
from .conv1d_residual_auxloss import Conv1dResidualAuxLossTrainer

# Import configuration
from .config import (
    IMAGE_TOKEN_ID,
    VISION_MODES,
    GUNDAM_PRESET,
    VISION_TOKEN_COUNT,
    ENCODER_COMPONENTS,
    MemoryProfile
)

# Import utilities
from .utils import (
    # Model loading
    load_model_and_tokenizer,
    # Freezing/unfreezing
    freeze_vision_encoder,
    unfreeze_vision_encoder,
    get_encoder_params,
    get_decoder_params,
    unfreeze_decoder,
    count_parameters,
    # Generation
    manual_generate_with_cache,
    # Image processing
    BasicImageTransform,
    create_dummy_image,
    # Memory profiling
    get_gpu_memory_mb,
    get_peak_gpu_memory_mb,
    reset_peak_memory,
    validate_gradients,
)

__all__ = [
    # Trainers
    'VisionCompressionTrainer',
    'TextBaselineTrainer',
    'SubsampleCompressionTrainer',
    'MeanPoolCompressionTrainer',
    'RandomProjectionCompressionTrainer',
    'Conv1dPyramidCompressionTrainer',
    'Conv1dResidualCompressionTrainer',
    'Conv1dResidualAuxLossTrainer',
    # Configuration
    'IMAGE_TOKEN_ID',
    'VISION_MODES',
    'GUNDAM_PRESET',
    'VISION_TOKEN_COUNT',
    'ENCODER_COMPONENTS',
    'MemoryProfile',
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
