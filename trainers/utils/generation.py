"""Text generation utilities with KV-cache support."""

from typing import List, Optional

import torch


def manual_generate_with_cache(
    model,
    tokenizer,
    initial_ids: torch.Tensor,
    initial_embeds: torch.Tensor,
    images: List,
    images_spatial_crop: List,
    images_seq_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 100,
    temperature: float = 0.0,
    device: Optional[torch.device] = None
) -> List[int]:
    """
    Manual autoregressive generation with KV-cache support.

    This function provides a unified generation implementation that works correctly
    with custom embedding injection (e.g., mean pooling, conv1d compression).

    Unlike HuggingFace's generate(), which has a bug when using custom inputs_embeds
    with multi-position injection, this implementation maintains the custom embeddings
    throughout the generation process by using the KV-cache correctly.

    Args:
        model: The DeepSeek-OCR model
        tokenizer: The tokenizer
        initial_ids: Initial token IDs with shape (batch_size, seq_len)
                    Used for shape/position tracking
                    NOTE: Currently only supports batch_size=1
        initial_embeds: Initial embeddings with shape (batch_size, seq_len, hidden_size)
                       May contain custom injected embeddings (pooled, compressed, etc.)
        images: List of (crop, global) image tuples for vision bypass
        images_spatial_crop: List of [rows, cols] for each batch item
        images_seq_mask: Optional mask for vision tokens (batch_size, seq_len)
        max_new_tokens: Maximum number of new tokens to generate
        temperature: Sampling temperature (0.0 = greedy, >0.0 = sampling)
        device: Device to use (if None, inferred from model)

    Returns:
        List of generated token IDs (not including initial sequence)

    Raises:
        AssertionError: If batch_size != 1
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    with torch.inference_mode():
        # Ensure inputs are on correct device
        initial_ids = initial_ids.to(device)
        initial_embeds = initial_embeds.to(device)

        batch_size = initial_ids.shape[0]

        # Enforce batch_size=1 requirement (current implementation assumes single sequence)
        assert batch_size == 1, (
            f"manual_generate_with_cache only supports batch_size=1, got {batch_size}. "
            f"The implementation hardcodes next_token_id[0].item() which only handles the first sequence."
        )

        # Step 0: Prefill - forward pass with full initial sequence
        # This captures the custom embeddings in the KV-cache
        outputs = model.forward(
            input_ids=initial_ids,
            inputs_embeds=initial_embeds,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            use_cache=True,
            return_dict=True
        )

        # Extract KV-cache and logits for last position
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]  # Shape: (batch_size, vocab_size)

        # Sample first token
        if temperature == 0.0:
            next_token_id = torch.argmax(next_token_logits, dim=-1)  # Shape: (batch_size,)
        else:
            probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token_id = torch.multinomial(probs, 1).squeeze(-1)  # Shape: (batch_size,)

        generated_ids = [next_token_id[0].item()]  # Assuming batch_size=1

        # Check for EOS
        if next_token_id[0].item() == tokenizer.eos_token_id:
            return generated_ids

        # Step 1+: Decode - generate remaining tokens using KV-cache
        for step in range(max_new_tokens - 1):
            # Forward pass with only the last token
            # The KV-cache contains all previous tokens (including custom embeddings)
            next_token_id_input = next_token_id.unsqueeze(1)  # Shape: (batch_size, 1)

            outputs = model.forward(
                input_ids=next_token_id_input,
                images=images,
                images_seq_mask=None,  # Not needed after prefill
                images_spatial_crop=images_spatial_crop,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )

            # Update KV-cache and get next logits
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]

            # Sample next token
            if temperature == 0.0:
                next_token_id = torch.argmax(next_token_logits, dim=-1)
            else:
                probs = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, 1).squeeze(-1)

            generated_ids.append(next_token_id[0].item())

            # Check for EOS
            if next_token_id[0].item() == tokenizer.eos_token_id:
                break

        return generated_ids
