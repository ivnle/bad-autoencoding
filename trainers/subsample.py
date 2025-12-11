"""SubsampleCompressionTrainer trainer implementation."""

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


class SubsampleCompressionTrainer:
    """
    Simple Encoder: Token Subsampling
    - Bypass vision encoder completely
    - Trainable language decoder
    - Context: subsampled text tokens (keep every Nth token)
    - Loss: computed only on target tokens (continuation or reconstruction)

    This trainer tests whether simple token subsampling can match vision compression
    for language modeling tasks.
    """

    def __init__(
        self,
        model,
        tokenizer,
        subsample_count: int,
        subsample_strategy: str = 'regular',
        device: str = 'cuda'
    ):
        """
        Initialize subsample compression trainer

        Args:
            model: DeepSeek-OCR model
            tokenizer: Tokenizer
            subsample_count: Number of tokens to keep (required)
            subsample_strategy: 'regular' (deterministic) or 'random' (stochastic)
            device: Device to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)

        # Validate strategy
        if subsample_strategy not in ['regular', 'random']:
            raise ValueError(f"subsample_strategy must be 'regular' or 'random', got {subsample_strategy}")
        self.subsample_strategy = subsample_strategy

        # Validate and set number of tokens to keep
        context_length = 1000  # Fixed for this dataset
        if subsample_count < 1 or subsample_count > context_length:
            raise ValueError(f"subsample_count must be 1-{context_length}, got {subsample_count}")
        self.num_keep = subsample_count

        self.context_length = context_length
        self.compressed_tokens = self.num_keep + 1  # +1 for separator (matches vision pattern)

        # Freeze vision encoder (won't be used)
        freeze_vision_encoder(model)
        unfreeze_decoder(model)

        # Print parameter counts
        params = count_parameters(model)
        print(f"\n[Simple Encoder: Token Subsampling]")
        print(f"Strategy: {subsample_strategy}")
        if subsample_strategy == 'regular':
            print(f"  Keeps {self.num_keep} tokens at evenly-spaced positions")
        else:  # random
            print(f"  Randomly samples {self.num_keep} tokens per forward pass (order preserved)")
        print(f"Compression: 1000 → {self.compressed_tokens} tokens")
        print(f"Total parameters: {params['total']:,}")
        print(f"Trainable: {params['trainable']:,} ({params['trainable_pct']:.1f}%)")
        print(f"  (includes {model.config.hidden_size} separator params)")
        print(f"Frozen: {params['frozen']:,}")

        # Pre-allocate zero tensors for vision bypass (same as TextBaselineTrainer)
        self.empty_crop = torch.zeros(
            0, 3, GUNDAM_PRESET['image_size'], GUNDAM_PRESET['image_size'],
            dtype=torch.bfloat16, device=self.device
        )
        self.zero_global = torch.zeros(
            1, 3, GUNDAM_PRESET['base_size'], GUNDAM_PRESET['base_size'],
            dtype=torch.bfloat16, device=self.device
        )

        # Pre-create spatial_crop lists for common batch sizes
        self.spatial_crop_cache = {
            bs: [[1, 1] for _ in range(bs)] for bs in [1, 2, 4, 8, 16, 32]
        }

        # Create learnable separator token (match meanpool pattern)
        # Get hidden dimension from model
        hidden_dim = model.config.hidden_size
        embed_std = 1 / torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        self.separator_embed = nn.Parameter(
            torch.randn(hidden_dim, device=self.device, dtype=torch.bfloat16) * embed_std
        )

        # Register as model parameter so it's saved with checkpoints
        # Note: This adds a small learned component (~5KB for hidden_dim=2560)
        self.model.register_parameter('subsample_separator', self.separator_embed)

        # Special token ID for separator position (placeholder)
        # Use IMAGE_TOKEN_ID since model knows how to handle masked positions
        self.separator_token_id = IMAGE_TOKEN_ID

    def _subsample(self, context_tokens: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """
        Subsample tokens based on strategy

        Args:
            context_tokens: [batch_size, seq_len] token IDs
            generator: Optional torch.Generator for deterministic random sampling (validation)

        Returns:
            subsampled: [batch_size, num_keep] token IDs
        """
        batch_size, seq_len = context_tokens.shape

        # Validate sequence length
        if seq_len < self.num_keep:
            raise ValueError(
                f"Cannot subsample {self.num_keep} tokens from sequence of length {seq_len}. "
                f"Context is too short (expected at least {self.num_keep} tokens)."
            )

        if self.subsample_strategy == 'regular':
            # Deterministic: evenly-spaced indices based on num_keep
            step = max(1, seq_len // self.num_keep)
            indices = torch.arange(0, seq_len, step, device=context_tokens.device)[:self.num_keep]
            # Repeat indices for each batch item
            indices = indices.unsqueeze(0).expand(batch_size, -1)
            subsampled = torch.gather(context_tokens, 1, indices)

        else:  # random
            # Stochastic: randomly sample num_keep tokens (order preserved)
            # Generate random indices for each batch item
            indices_list = []
            for _ in range(batch_size):
                # Random sample without replacement (pass generator for deterministic eval)
                indices = torch.randperm(seq_len, device=context_tokens.device, generator=generator)[:self.num_keep]
                # Sort to preserve order
                indices = indices.sort()[0]
                indices_list.append(indices)

            # Stack into [batch_size, num_keep]
            indices = torch.stack(indices_list)

            # Gather tokens at sampled positions
            subsampled = torch.gather(context_tokens, 1, indices)

        return subsampled

    def forward(
        self,
        context_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        objective: str = 'lm',
        generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        """
        Forward pass with token subsampling

        Args:
            context_tokens: Context token IDs (shape: [batch_size, seq_len])
            target_tokens: Target token IDs (continuation for lm, context for reconstruction)
            objective: Training objective ('lm' or 'reconstruction')
            generator: Optional torch.Generator for deterministic random sampling (validation)

        Returns:
            Loss tensor
        """
        batch_size = context_tokens.shape[0]
        bos_id = self.tokenizer.bos_token_id

        # Subsample context tokens (strategy-dependent: regular or random)
        subsampled = self._subsample(context_tokens, generator=generator)

        # Build sequence based on objective
        if objective == 'lm':
            # [BOS] + [SUBSAMPLED_CONTEXT] + [SEPARATOR] + [CONTINUATION]
            bos_tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long)
            separator_tokens = torch.full((batch_size, 1), self.separator_token_id, dtype=torch.long)
            input_ids = torch.cat([bos_tokens, subsampled, separator_tokens, target_tokens], dim=1)

            # Labels: mask BOS, subsampled context, and separator; only train on continuation
            labels = torch.cat([
                torch.full((batch_size, 1), -100, dtype=torch.long),  # BOS masked
                torch.full_like(subsampled, -100),  # Subsampled context masked
                torch.full((batch_size, 1), -100, dtype=torch.long),  # Separator masked
                target_tokens  # Continuation (target for loss)
            ], dim=1)

        else:  # reconstruction
            # [BOS] + [SUBSAMPLED_CONTEXT] + [SEPARATOR] + [ORIGINAL_CONTEXT]
            # Note: target_tokens should be context_tokens for reconstruction
            original_context = target_tokens
            bos_tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long)
            separator_tokens = torch.full((batch_size, 1), self.separator_token_id, dtype=torch.long)
            input_ids = torch.cat([bos_tokens, subsampled, separator_tokens, original_context], dim=1)

            # Labels: mask BOS, subsampled context, and separator; reconstruct original context
            labels = torch.cat([
                torch.full((batch_size, 1), -100, dtype=torch.long),  # BOS masked
                torch.full_like(subsampled, -100),  # Subsampled context masked
                torch.full((batch_size, 1), -100, dtype=torch.long),  # Separator masked
                original_context  # Original context (reconstruction target)
            ], dim=1)

        # Move to device
        input_ids = input_ids.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        # Get initial embeddings (includes separator placeholder)
        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)

        # Inject learnable separator embedding at separator position
        # Separator is always at position: 1 (BOS) + num_keep (subsampled tokens) = 1 + num_keep
        separator_pos = 1 + self.num_keep
        separator_mask = torch.zeros(batch_size, input_ids.shape[1], dtype=torch.bool, device=self.device)
        separator_mask[:, separator_pos] = True

        # Expand separator for batch and inject
        separator_expanded = self.separator_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        inputs_embeds.masked_scatter_(
            separator_mask.unsqueeze(-1),
            separator_expanded.reshape(-1, separator_expanded.shape[-1])
        )

        # Bypass vision encoder with zero-valued images
        zero_global_expanded = self.zero_global.expand(batch_size, -1, -1, -1)
        images = [(self.empty_crop, zero_global_expanded)]
        images_spatial_crop = self.spatial_crop_cache.get(batch_size, [[1, 1]] * batch_size)

        # Forward pass (use inputs_embeds with injected separator)
        outputs = self.model.forward(
            input_ids=input_ids,  # Keep for shape info
            inputs_embeds=inputs_embeds,  # Use for actual embeddings
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
        generator: Optional[torch.Generator] = None,
        **kwargs
    ) -> str:
        """
        Generate text from subsampled context

        Args:
            context_tokens: Context token IDs (shape: [seq_len])
            prompt_text: Optional additional text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            generator: Optional torch.Generator for deterministic random sampling (validation)
            **kwargs: Additional generation parameters

        Returns:
            Generated text (decoded)
        """
        # Subsample context tokens using strategy
        context_batch = context_tokens.unsqueeze(0)  # [seq_len] -> [1, seq_len]
        subsampled_batch = self._subsample(context_batch, generator=generator)  # [1, num_keep]
        subsampled = subsampled_batch.squeeze(0)  # [num_keep]

        # Add separator token
        separator_token = torch.tensor([self.separator_token_id], dtype=torch.long, device=subsampled.device)

        # Add optional prompt
        if prompt_text:
            prompt_token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_tokens = torch.tensor(prompt_token_ids, dtype=torch.long, device=subsampled.device)
            input_ids = torch.cat([subsampled, separator_token, prompt_tokens])
        else:
            input_ids = torch.cat([subsampled, separator_token])

        # Add BOS token
        bos_id = self.tokenizer.bos_token_id
        bos_token = torch.tensor([bos_id], dtype=torch.long, device=subsampled.device)
        input_ids = torch.cat([bos_token, input_ids])

        # Move to device and add batch dimension
        input_ids = input_ids.unsqueeze(0).to(self.device)

        # Get initial embeddings and inject separator
        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
        separator_pos = 1 + self.num_keep  # After BOS + subsampled tokens
        separator_mask = torch.zeros(1, input_ids.shape[1], dtype=torch.bool, device=self.device)
        separator_mask[0, separator_pos] = True
        separator_expanded = self.separator_embed.unsqueeze(0).unsqueeze(0)
        inputs_embeds.masked_scatter_(
            separator_mask.unsqueeze(-1),
            separator_expanded.reshape(-1, separator_expanded.shape[-1])
        )

        # Bypass vision encoder
        images = [(self.empty_crop, self.zero_global)]
        images_spatial_crop = [[1, 1]]

        # Generate using manual decode with KV-cache
        generated_ids = manual_generate_with_cache(
            model=self.model,
            tokenizer=self.tokenizer,
            initial_ids=input_ids,
            initial_embeds=inputs_embeds,
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

