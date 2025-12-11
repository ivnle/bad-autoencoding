"""TextBaselineTrainer trainer implementation."""

import math
from typing import List, Optional, Tuple, Dict, Union

import numpy as np
import torch
import torch.nn as nn

from .config import IMAGE_TOKEN_ID, VISION_MODES, GUNDAM_PRESET, VISION_TOKEN_COUNT
from .utils import (
    freeze_vision_encoder,
    unfreeze_vision_encoder,
    unfreeze_decoder,
    count_parameters,
    manual_generate_with_cache
)


class TextBaselineTrainer:
    """
    Regime 2: Text Baseline
    - Bypass vision encoder completely
    - Trainable language decoder (same as Regime 1)
    - Context: raw text tokens (uncompressed)
    - Loss: computed only on continuation tokens
    """

    def __init__(self, model, tokenizer, device: str = 'cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)

        # Keep vision encoder frozen but we won't call it
        # Only unfreeze decoder for training
        freeze_vision_encoder(model)  # Ensure frozen (just in case)
        unfreeze_decoder(model)

        # Print parameter counts
        params = count_parameters(model)
        print(f"\n[Regime 2: Text Baseline]")
        print(f"Total parameters: {params['total']:,}")
        print(f"Trainable: {params['trainable']:,} ({params['trainable_pct']:.1f}%)")
        print(f"Frozen: {params['frozen']:,}")

        # Pre-allocate zero tensors for vision bypass (reused across forward passes)
        # These bypass the vision encoder by passing zero-valued images
        self.empty_crop = torch.zeros(
            0, 3, GUNDAM_PRESET['image_size'], GUNDAM_PRESET['image_size'],
            dtype=torch.bfloat16, device=self.device
        )
        self.zero_global = torch.zeros(
            1, 3, GUNDAM_PRESET['base_size'], GUNDAM_PRESET['base_size'],
            dtype=torch.bfloat16, device=self.device
        )

        # Pre-create spatial_crop lists for common batch sizes to avoid Python list allocation overhead
        # Use list comprehension to create independent copies (not aliased references)
        self.spatial_crop_cache = {
            bs: [[1, 1] for _ in range(bs)] for bs in [1, 2, 4, 8, 16, 32]
        }

    def prepare_batch(
        self,
        context_tokens: torch.Tensor,
        continuation_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare training batch for Regime 2

        NOTE: This method is kept for backward compatibility with test functions.
        The main training loop now receives pre-prepared batches from the collate function.

        Args:
            context_tokens: Token IDs for text context (shape: [batch_size, seq_len])
            continuation_tokens: Token IDs for text continuation (shape: [batch_size, seq_len])

        Returns:
            Tuple of (input_ids, labels)
        """
        # Import here to avoid circular dependency
        from train import prepare_text_batch
        return prepare_text_batch(context_tokens, continuation_tokens)

    def forward(
        self,
        batch_or_context: Union[dict, torch.Tensor],
        target_tokens: Optional[torch.Tensor] = None,
        objective: str = 'lm'
    ) -> torch.Tensor:
        """
        Forward pass for Regime 2

        Args:
            batch_or_context: Either a batch dict with 'input_ids' and 'labels' keys (new path),
                             or context_tokens tensor (backward compatibility for tests)
            target_tokens: Target tokens (continuation for lm, context for reconstruction).
                          Only used if batch_or_context is a tensor (backward compat)
            objective: Training objective ('lm' or 'reconstruction')

        Returns:
            Loss tensor
        """
        # Support both new batch dict format and old (context, continuation) format
        if isinstance(batch_or_context, dict):
            # New path: batch dict prepared by collate function
            # Note: collate prepares for LM by default, need to rebuild for reconstruction
            if objective == 'reconstruction':
                # Rebuild batch for reconstruction
                # Extract context from batch (need to get from dataset side)
                # For now, assume batch has 'context' key for reconstruction
                if 'context' in batch_or_context:
                    context_tokens = batch_or_context['context']
                    # Build: [BOS] + [CONTEXT] + [CONTEXT]
                    batch_size = context_tokens.shape[0]
                    bos_id = self.tokenizer.bos_token_id
                    bos_tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long)

                    input_ids = torch.cat([bos_tokens, context_tokens, context_tokens], dim=1)
                    labels = torch.cat([
                        torch.full((batch_size, 1), -100, dtype=torch.long),  # BOS masked
                        torch.full_like(context_tokens, -100),  # First context masked
                        context_tokens  # Second context (reconstruction target)
                    ], dim=1)
                else:
                    raise ValueError("Reconstruction objective requires 'context' in batch dict")
            else:
                # LM objective: use batch as-is
                input_ids = batch_or_context['input_ids']
                labels = batch_or_context['labels']
        else:
            # Backward compatibility: prepare batch from raw tensors
            context_tokens = batch_or_context

            if objective == 'reconstruction':
                # Build: [BOS] + [CONTEXT] + [CONTEXT]
                batch_size = context_tokens.shape[0] if context_tokens.dim() > 1 else 1
                if context_tokens.dim() == 1:
                    context_tokens = context_tokens.unsqueeze(0)

                bos_id = self.tokenizer.bos_token_id
                bos_tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long)

                input_ids = torch.cat([bos_tokens, context_tokens, context_tokens], dim=1)
                labels = torch.cat([
                    torch.full((batch_size, 1), -100, dtype=torch.long),  # BOS masked
                    torch.full_like(context_tokens, -100),  # First context masked
                    context_tokens  # Second context (reconstruction target)
                ], dim=1)
            else:
                # LM objective: use prepare_batch (for backward compat)
                continuation_tokens = target_tokens
                input_ids, labels = self.prepare_batch(context_tokens, continuation_tokens)

        # Move to device (batch dimension already exists)
        # Use non_blocking=True for async transfers (pin_memory=True in DataLoader enables this)
        input_ids = input_ids.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        # To bypass vision encoder, pass zero-valued images
        # Model checks: torch.sum(images[0][1]).item() != 0
        # If sum is 0, vision encoder is skipped
        # Use pre-allocated zero tensors (initialized in __init__)
        # Note: zero_global has batch dim of 1, need to expand for actual batch size
        batch_size = input_ids.shape[0]
        zero_global_expanded = self.zero_global.expand(batch_size, -1, -1, -1)
        images = [(self.empty_crop, zero_global_expanded)]

        # Use cached spatial crop list to avoid Python list allocation overhead
        images_spatial_crop = self.spatial_crop_cache.get(batch_size, [[1, 1]] * batch_size)

        # Forward pass with zero-valued images (bypasses vision encoder)
        outputs = self.model.forward(
            input_ids=input_ids,
            images=images,
            images_spatial_crop=images_spatial_crop,
            labels=labels,
            return_dict=True
        )

        return outputs.loss, labels

    def generate_text(
        self,
        context_tokens: torch.Tensor,
        prompt_text: str = "",
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate text from text context (optionally with additional prompt)

        Args:
            context_tokens: Token IDs for text context (shape: [seq_len])
            prompt_text: Optional additional text prompt to append after context
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            **kwargs: Additional generation parameters

        Returns:
            Generated text (decoded)
        """
        # Encode additional prompt if provided
        if prompt_text:
            prompt_token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            # Create tensor on same device/dtype as context_tokens
            prompt_tokens = context_tokens.new_tensor(prompt_token_ids)
            input_ids = torch.cat([context_tokens, prompt_tokens])
        else:
            input_ids = context_tokens

        # Add BOS token at the beginning (match official implementation)
        bos_id = self.tokenizer.bos_token_id
        # Create BOS token on same device/dtype as input_ids
        bos_token = input_ids.new_full((1,), bos_id)
        input_ids = torch.cat([bos_token, input_ids])

        # Move to device and add batch dimension
        input_ids = input_ids.unsqueeze(0).to(self.device)

        # Bypass vision encoder (same as forward pass)
        # Use pre-allocated zero tensors for efficiency
        images = [(self.empty_crop, self.zero_global)]
        images_spatial_crop = [[1, 1]]

        # Get initial embeddings (standard vocabulary embeddings)
        initial_embeds = self.model.model.get_input_embeddings()(input_ids)

        # Generate using manual decode with KV-cache
        generated_ids = manual_generate_with_cache(
            model=self.model,
            tokenizer=self.tokenizer,
            initial_ids=input_ids,
            initial_embeds=initial_embeds,
            images=images,
            images_spatial_crop=images_spatial_crop,
            images_seq_mask=None,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=self.device
        )

        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text

