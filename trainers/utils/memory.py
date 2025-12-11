"""Memory profiling and gradient validation utilities."""

from typing import Dict
import torch


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB"""
    return torch.cuda.memory_allocated() / 1024 / 1024


def get_peak_gpu_memory_mb() -> float:
    """Get peak GPU memory usage in MB"""
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def reset_peak_memory():
    """Reset peak memory statistics"""
    torch.cuda.reset_peak_memory_stats()


def validate_gradients(model, regime: str = 'vision') -> Dict[str, any]:
    """
    Validate that gradients flow to correct components

    Args:
        model: The model to validate
        regime: 'vision' or 'text'

    Returns:
        Dict with validation results
    """
    results = {
        'total_params': 0,
        'trainable_params': 0,
        'params_with_grad': 0,
        'params_without_grad': 0,
        'moe_params_without_grad': 0,
        'critical_errors': []
    }

    for name, param in model.named_parameters():
        results['total_params'] += 1

        if param.requires_grad:
            results['trainable_params'] += 1

            if param.grad is None:
                results['params_without_grad'] += 1

                # Missing gradients in MoE experts are normal (only some experts activated)
                if 'experts' in name:
                    results['moe_params_without_grad'] += 1
                # Other missing gradients might be concerning (but not critical)

            else:
                results['params_with_grad'] += 1
                grad_norm = param.grad.norm().item()

                # Check for frozen components with gradients (this IS an error)
                if any(frozen in name for frozen in ['sam_model', 'vision_model', 'projector']):
                    if regime == 'vision':
                        results['critical_errors'].append(f"CRITICAL: Frozen component has gradient: {name} (norm={grad_norm:.4f})")

    # Success criteria for MoE models:
    # 1. At least some parameters have gradients (model is learning)
    # 2. No frozen components have gradients (freezing works)
    # 3. It's OK if MoE experts don't all have gradients (only some are activated)

    has_some_gradients = results['params_with_grad'] > 0
    no_critical_errors = len(results['critical_errors']) == 0

    results['success'] = has_some_gradients and no_critical_errors

    return results
