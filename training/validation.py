"""Argument validation utilities for training script."""

from typing import Optional


def compute_valid_conv1d_targets(context_length: int = 1000) -> list[int]:
    """
    Compute valid native compression targets for stride-2 conv1d pyramid.

    Conv1d compression regimes use stride-2 convolutions to progressively
    halve the sequence length. This function computes all valid "native"
    targets that can be reached using integer divisions with no interpolation.

    Args:
        context_length: Input sequence length (default: 1000)

    Returns:
        List of valid compression targets in descending order.
        For context_length=1000, returns [500, 250, 125, 63, 32, 16, 8, 4, 2, 1]

    Examples:
        >>> compute_valid_conv1d_targets(1000)
        [500, 250, 125, 63, 32, 16, 8, 4, 2, 1]

        >>> compute_valid_conv1d_targets(512)
        [256, 128, 64, 32, 16, 8, 4, 2, 1]
    """
    valid_targets = []
    temp_length = context_length

    # Repeatedly halve sequence length until we reach 1
    while temp_length > 1:
        temp_length = (temp_length + 1) // 2  # Ceiling division
        valid_targets.append(temp_length)

    return valid_targets


def validate_conv1d_params(
    compression_target: Optional[int],
    conv_kernel: int,
    regime: str,
    context_length: int = 1000
) -> None:
    """
    Validate compression_target and conv_kernel parameters for conv1d regimes.

    This function consolidates validation logic for the conv1d_residual regime.

    Args:
        compression_target: Target number of compressed tokens (required)
        conv_kernel: Convolutional kernel size (must be odd and >= 1)
        regime: Regime name (for error messages)
        context_length: Input sequence length (default: 1000)

    Raises:
        ValueError: If compression_target is None, not a native target, or if
                    conv_kernel is invalid (even or < 1)

    Examples:
        >>> # Valid parameters
        >>> validate_conv1d_params(125, 5, 'conv1d')  # OK

        >>> # Invalid: compression_target is None
        >>> validate_conv1d_params(None, 5, 'conv1d')
        ValueError: --compression_target is required when regime=conv1d

        >>> # Invalid: non-native target (requires interpolation)
        >>> validate_conv1d_params(100, 5, 'conv1d')
        ValueError: --compression_target must be a native target...

        >>> # Invalid: even kernel size
        >>> validate_conv1d_params(125, 4, 'conv1d')
        ValueError: --conv_kernel must be odd and >= 1, got 4
    """
    # Validate compression_target is provided
    if compression_target is None:
        raise ValueError(f"--compression_target is required when regime={regime}")

    # Validate compression_target is a native pyramid target
    valid_targets = compute_valid_conv1d_targets(context_length)

    if compression_target not in valid_targets:
        raise ValueError(
            f"--compression_target must be a native target (no adaptive pooling). "
            f"Got {compression_target}, valid options: {valid_targets}\n"
            f"Native targets use stride-2 convolutions with no interpolation.\n"
            f"Formula: ceil({context_length} / 2^num_layers) for num_layers >= 1"
        )

    # Validate conv_kernel is odd and positive
    if conv_kernel < 1 or conv_kernel % 2 == 0:
        raise ValueError(f"--conv_kernel must be odd and >= 1, got {conv_kernel}")
