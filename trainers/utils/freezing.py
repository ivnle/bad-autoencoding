"""Model freezing/unfreezing and parameter management utilities."""

from typing import Dict

from ..config import ENCODER_COMPONENTS


def freeze_vision_encoder(model):
    """
    Freeze vision encoder components (SAM + ViT + Projector)

    For Regime 1: Vision Compression
    """
    frozen_params = 0
    for name, param in model.named_parameters():
        # Strip _orig_mod. prefix added by torch.compile()
        param_name = name.removeprefix('_orig_mod.')
        for component in ENCODER_COMPONENTS:
            if param_name.startswith(component):
                param.requires_grad = False
                frozen_params += param.numel()
                break

    print(f"Frozen vision encoder: {frozen_params:,} parameters")

    # Clear encoder trainability cache to force recomputation on next forward pass
    # Cache lives on the inner model (model.model), not the outer wrapper
    inner_model = getattr(model, 'model', model)
    if hasattr(inner_model, '_encoder_trainable_cache'):
        delattr(inner_model, '_encoder_trainable_cache')

    return frozen_params


def unfreeze_vision_encoder(model):
    """
    Unfreeze vision encoder components (SAM + ViT + Projector)

    Opposite of freeze_vision_encoder() - enables training of encoder.
    For trainable encoder experiments.
    """
    trainable_params = 0
    for name, param in model.named_parameters():
        # Strip _orig_mod. prefix added by torch.compile()
        param_name = name.removeprefix('_orig_mod.')
        for component in ENCODER_COMPONENTS:
            if param_name.startswith(component):
                param.requires_grad = True
                trainable_params += param.numel()
                break

    print(f"Trainable vision encoder: {trainable_params:,} parameters")

    # Clear encoder trainability cache to force recomputation on next forward pass
    # Cache lives on the inner model (model.model), not the outer wrapper
    inner_model = getattr(model, 'model', model)
    if hasattr(inner_model, '_encoder_trainable_cache'):
        delattr(inner_model, '_encoder_trainable_cache')

    return trainable_params


def get_encoder_params(model):
    """
    Get encoder parameters for differential learning rates

    Returns list of parameters belonging to the ENCODER (context compression):
    - Vision regime: SAM, ViT, and Projector components
    - All compression modules:
      * Separator embeddings (meanpool_separator, conv1d_residual_separator, etc.)
      * Conv1D pyramids (conv1d_residual_pyramid)
      * Conv1D output norms (conv1d_output_norm)

    Used for creating optimizer with separate encoder/decoder param groups.
    """
    encoder_params = []

    for name, param in model.named_parameters():
        # Strip _orig_mod. prefix added by torch.compile()
        param_name = name.removeprefix('_orig_mod.')

        # Vision encoder components (SAM + ViT + Projector)
        if any(param_name.startswith(comp) for comp in ENCODER_COMPONENTS):
            if param.requires_grad:
                encoder_params.append(param)

        # Compression module: separator embeddings (all regimes)
        # Matches: meanpool_separator, conv1d_residual_separator
        elif param_name.endswith('_separator'):
            if param.requires_grad:
                encoder_params.append(param)

        # Compression module: conv1d pyramid networks
        # Matches: conv1d_residual_pyramid.*
        elif param_name.startswith('conv1d_residual_pyramid.'):
            if param.requires_grad:
                encoder_params.append(param)

        # Compression module: conv1d output normalization
        elif param_name.startswith('conv1d_output_norm.'):
            if param.requires_grad:
                encoder_params.append(param)

    return encoder_params


def get_decoder_params(model):
    """
    Get decoder parameters for differential learning rates

    Returns list of parameters belonging to the DECODER (language model only).
    Includes:
    - Text token embeddings (model.embed_tokens)
    - Transformer decoder layers (model.layers)
    - Language model head (lm_head)
    - Final layer norm (model.norm)

    Excludes ALL encoder/compression components:
    - Vision encoder: SAM, ViT, Projector
    - Compression modules: separators, conv pyramids, etc.

    Used for creating optimizer with separate encoder/decoder param groups.
    """
    decoder_params = []

    for name, param in model.named_parameters():
        # Strip _orig_mod. prefix added by torch.compile()
        param_name = name.removeprefix('_orig_mod.')

        # Exclude vision encoder components
        if any(param_name.startswith(comp) for comp in ENCODER_COMPONENTS):
            continue

        # Exclude compression module: separator embeddings (all regimes)
        if param_name.endswith('_separator'):
            continue

        # Exclude compression module: conv1d pyramid networks
        if param_name.startswith('conv1d_residual_pyramid.'):
            continue

        # Exclude compression module: conv1d output normalization
        if param_name.startswith('conv1d_output_norm.'):
            continue

        # Include everything else that's trainable (language model components)
        if param.requires_grad:
            decoder_params.append(param)

    return decoder_params


def unfreeze_decoder(model):
    """
    Unfreeze language decoder components

    For both regimes
    """
    components_to_train = [
        'model.layers',
        'lm_head'
    ]

    trainable_params = 0
    for name, param in model.named_parameters():
        # Strip _orig_mod. prefix added by torch.compile()
        param_name = name.removeprefix('_orig_mod.')
        for component in components_to_train:
            if param_name.startswith(component):
                param.requires_grad = True
                trainable_params += param.numel()
                break

    print(f"Trainable decoder: {trainable_params:,} parameters")
    return trainable_params


def count_parameters(model) -> Dict[str, int]:
    """Count parameters by component"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable

    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
        'trainable_pct': 100.0 * trainable / total
    }
