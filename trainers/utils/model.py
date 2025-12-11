"""Model loading utilities."""

import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


def load_model_and_tokenizer(
    model_name: str = './vendor/deepseek_ocr',
    weights_name: str = 'deepseek-ai/DeepSeek-OCR',
    device: str = 'cuda',
    use_optimized_model: bool = False,
    use_encoder_checkpointing: bool = False,
    load_vision_encoder: bool = True
):
    """
    Load DeepSeek-OCR model and tokenizer

    Args:
        model_name: Path to vendored DeepSeek-OCR code and config
        weights_name: HuggingFace model identifier (or cache path) for weights
        device: Device to load model to ('cuda', 'cuda:0', 'cpu', etc.)
        use_optimized_model: If True, use training-optimized implementation
        use_encoder_checkpointing: If True, enable gradient checkpointing for vision encoder
        load_vision_encoder: If True, load vision encoder components (SAM, ViT, projector).
                            Set to False for text-only regimes to save ~1.5-2GB GPU memory.

    Returns:
        Tuple of (model, tokenizer)
    """
    vendor_path = Path(model_name).expanduser().resolve()
    if not vendor_path.exists():
        raise FileNotFoundError(f"Vendored model path not found: {vendor_path}")

    # Add vendor parent directory to sys.path to enable package imports
    vendor_parent = vendor_path.parent.resolve()
    vendor_parent_str = str(vendor_parent)
    if vendor_parent_str not in sys.path:
        sys.path.insert(0, vendor_parent_str)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(vendor_path),
        trust_remote_code=True
    )

    # Validate flag combination
    if use_encoder_checkpointing and not use_optimized_model:
        raise ValueError(
            "Error: --use_encoder_checkpointing requires --use_optimized_model\n"
            "The original implementation does not support gradient checkpointing.\n"
            "Please add --use_optimized_model flag or remove --use_encoder_checkpointing."
        )

    # Import appropriate implementation based on flag
    if use_optimized_model:
        print("🚀 Using OPTIMIZED DeepSeek-OCR implementation")
        from deepseek_ocr.modeling_deepseekocr_optimized import (
            DeepseekOCRConfigOptimized as DeepseekOCRConfig,
            DeepseekOCRForCausalLMOptimized as DeepseekOCRForCausalLM
        )
        if use_encoder_checkpointing:
            print("   ✓ Gradient checkpointing enabled for vision encoder")
    else:
        print("📦 Using ORIGINAL DeepSeek-OCR implementation")
        from deepseek_ocr.modeling_deepseekocr import DeepseekOCRConfig, DeepseekOCRForCausalLM

    print("Loading config...")
    config = DeepseekOCRConfig.from_pretrained(str(vendor_path))
    config._attn_implementation = "flash_attention_2"

    print("Loading model...")
    if not load_vision_encoder:
        print("   ⚠️  Vision encoder disabled - saving ~1.5-2GB GPU memory")

    # Pass checkpointing flag to optimized version if applicable
    if use_optimized_model:
        model = DeepseekOCRForCausalLM.from_pretrained(
            weights_name,
            config=config,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            local_files_only=False,
            use_encoder_checkpointing=use_encoder_checkpointing,
            load_vision_encoder=load_vision_encoder
        )
    else:
        model = DeepseekOCRForCausalLM.from_pretrained(
            weights_name,
            config=config,
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            local_files_only=False,
            load_vision_encoder=load_vision_encoder
        )

    device_obj = torch.device(device)
    print(f"Ensuring bfloat16 dtype and moving to {device}...")
    model = model.eval().to(device_obj).to(torch.bfloat16)

    return model, tokenizer
