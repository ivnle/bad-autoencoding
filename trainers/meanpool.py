"""MeanPoolCompressionTrainer trainer implementation."""

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


class MeanPoolCompressionTrainer:
    """
    Simple Encoder: Sliding Window Mean Pooling
    - Bypass vision encoder completely
    - Trainable language decoder
    - Context: mean-pooled text embeddings (sliding window)
    - Loss: computed only on target tokens (continuation or reconstruction)

    This trainer tests whether simple embedding-level mean pooling can match
    vision compression for language modeling tasks.
    """

    def __init__(
        self,
        model,
        tokenizer,
        window_size: int = 9,
        stride: int = 9,
        device: str = 'cuda',
        hybrid_text_tokens: int = 0
    ):
        """
        Initialize mean pool compression trainer

        Args:
            model: DeepSeek-OCR model
            tokenizer: Tokenizer
            window_size: Sliding window size for mean pooling (default: 9)
            stride: Stride for sliding window (default: 9, no overlap)
            device: Device to use
            hybrid_text_tokens: Number of uncompressed text tokens to append after pooled tokens (default: 0)
                - 0: Compression-only mode (pooled tokens only)
                - >0: Hybrid mode (pooled tokens + last K explicit tokens from context)
                Note: These tokens are ALSO included in pooling (redundant encoding is intentional)
                This provides both compressed and explicit representations for better quality.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.window_size = window_size
        self.stride = stride
        self.device = torch.device(device)
        self.hybrid_text_tokens = hybrid_text_tokens

        # Validate parameters
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        # Validate window_size against context length
        context_length = 1000  # Fixed for this dataset

        if hybrid_text_tokens < 0:
            raise ValueError(f"hybrid_text_tokens must be >= 0, got {hybrid_text_tokens}")
        if hybrid_text_tokens >= context_length:
            raise ValueError(
                f"hybrid_text_tokens ({hybrid_text_tokens}) must be < context_length ({context_length})"
            )
        if window_size >= context_length:
            raise ValueError(
                f"Window size ({window_size}) must be < context length ({context_length})"
            )

        # Calculate compression with flexible last window
        # Calculate what unfold() will produce (regular windows)
        self.num_regular_windows = (context_length - window_size) // stride + 1

        # Calculate position where regular windows end
        regular_end_pos = (self.num_regular_windows - 1) * stride + window_size

        # Check if there's a remainder to pool
        self.last_window_size = context_length - regular_end_pos
        if self.last_window_size > 0:
            # We have remainder tokens - will pool into one more window
            self.num_pooled_windows = self.num_regular_windows + 1
        else:
            # No remainder - regular windows covered everything
            self.num_pooled_windows = self.num_regular_windows
            self.last_window_size = None  # Explicitly mark as no remainder

        # Total compressed tokens = pooled windows + separator
        self.compressed_tokens = self.num_pooled_windows + 1  # +1 for separator

        # Calculate overlap percentage and check for gaps
        if window_size > stride:
            overlap = window_size - stride
            overlap_pct = (overlap / window_size) * 100
        elif window_size == stride:
            overlap_pct = 0.0  # No overlap, perfect tiling
        else:  # stride > window_size
            overlap_pct = 0.0  # No overlap (but gaps exist)
            gap_size = stride - window_size
            # Note: Will print warning after all initialization completes

        # Freeze vision encoder (won't be used)
        freeze_vision_encoder(model)
        unfreeze_decoder(model)

        # Create learnable separator token (match vision pattern)
        # Get hidden dimension from model
        hidden_dim = model.config.hidden_size
        embed_std = 1 / torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        self.separator_embed = nn.Parameter(
            torch.randn(hidden_dim, device=self.device, dtype=torch.bfloat16) * embed_std
        )

        # Register as model parameter so it's saved with checkpoints
        # Note: This adds a small learned component (~5KB for hidden_dim=2560)
        self.model.register_parameter('meanpool_separator', self.separator_embed)

        # Special token ID for mean-pooled positions (placeholder)
        # Use IMAGE_TOKEN_ID since model knows how to handle masked positions
        self.pooled_token_id = IMAGE_TOKEN_ID

        # Print compression configuration
        params = count_parameters(model)
        print(f"\n[Simple Encoder: Sliding Window Mean Pooling]")
        print(f"Window size: {self.window_size} tokens")
        if self.stride > self.window_size:
            gap_size = self.stride - self.window_size
            print(f"Stride: {self.stride} tokens ({overlap_pct:.1f}% overlap, {gap_size}-token gaps)")
        else:
            print(f"Stride: {self.stride} tokens ({overlap_pct:.1f}% overlap)")
        print(f"Compression: 1000 → {self.compressed_tokens} tokens")
        print(f"  - {self.num_regular_windows} regular windows × {self.window_size} tokens each")
        if self.last_window_size is not None and self.last_window_size > 0:
            print(f"  - Last window: {self.last_window_size} tokens")
        print(f"  - Separator: 1 token (learnable)")
        print(f"Total parameters: {params['total']:,}")
        print(f"Trainable: {params['trainable']:,} ({params['trainable_pct']:.1f}%)")
        print(f"  (includes {hidden_dim} separator params)")
        print(f"Frozen: {params['frozen']:,}")

        # Log hybrid mode configuration
        if self.hybrid_text_tokens > 0:
            total_context_tokens = self.compressed_tokens + self.hybrid_text_tokens
            print(f"Hybrid mode: {self.compressed_tokens} compressed + {self.hybrid_text_tokens} text = {total_context_tokens} total context tokens")

        # Warn about gaps if stride > window_size
        if self.stride > self.window_size:
            gap_size = self.stride - self.window_size
            print(f"\n⚠️  WARNING: Stride ({self.stride}) > window_size ({self.window_size}) creates {gap_size}-token gaps.")
            print(f"   Tokens between windows are permanently dropped. Last window pools only remainder tokens after the final regular window.")

        # Pre-allocate zero tensors for vision bypass
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

    def _sliding_window_mean_pool(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        Apply sliding window mean pooling to embeddings

        Args:
            embeds: Token embeddings (shape: [batch_size, seq_len, hidden_dim])

        Returns:
            Pooled embeddings (shape: [batch_size, num_windows, hidden_dim])
        """
        batch_size, seq_len, hidden_dim = embeds.shape

        # Validate inputs for unfold operation
        if seq_len < self.window_size:
            raise ValueError(
                f"Sequence length ({seq_len}) must be >= window_size ({self.window_size}). "
                f"Context is too short for mean pooling compression."
            )
        if self.stride < 1:
            raise ValueError(
                f"Stride must be >= 1, got {self.stride}. "
                f"Try using larger window_size or smaller stride value."
            )

        # Use unfold to extract regular windows
        # unfold(dimension, size, step)
        windows = embeds.unfold(1, self.window_size, self.stride)
        # Shape: (batch_size, num_windows, hidden_dim, window_size)

        # Mean pool each regular window along the window_size dimension
        pooled_regular = windows.mean(dim=-1)
        # Shape: (batch_size, num_windows, hidden_dim)

        # Calculate where regular windows end
        num_regular = pooled_regular.shape[1]
        regular_end_pos = (num_regular - 1) * self.stride + self.window_size

        # Pool remaining tokens if any exist (flexible last window)
        if regular_end_pos < seq_len:
            remainder = embeds[:, regular_end_pos:, :]
            pooled_remainder = remainder.mean(dim=1, keepdim=True)
            # Shape: (batch_size, 1, hidden_dim)

            # Concatenate regular windows + last window
            pooled = torch.cat([pooled_regular, pooled_remainder], dim=1)
        else:
            # No remainder - just use regular windows
            pooled = pooled_regular

        return pooled

    def forward(
        self,
        context_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        hybrid_text: Optional[torch.Tensor] = None,
        objective: str = 'lm'
    ) -> torch.Tensor:
        """
        Forward pass with mean pooling embedding injection

        Args:
            context_tokens: Context token IDs (shape: [batch_size, seq_len])
            target_tokens: Target token IDs (continuation for lm, context for reconstruction)
            hybrid_text: Optional hybrid text token IDs (shape: [batch_size, hybrid_len])
            objective: Training objective ('lm' or 'reconstruction')

        Returns:
            Loss tensor
        """
        batch_size = context_tokens.shape[0]
        bos_id = self.tokenizer.bos_token_id

        # Move context_tokens to device before embedding lookup
        context_tokens = context_tokens.to(self.device, non_blocking=True)
        # Keep target_tokens on CPU for concatenation (moved to device later)

        # 1. Get embeddings for context tokens
        context_embeds = self.model.model.get_input_embeddings()(context_tokens)
        # Shape: (batch_size, context_len, hidden_dim)

        # 2. Apply sliding window mean pooling
        pooled_embeds = self._sliding_window_mean_pool(context_embeds)
        num_pooled = pooled_embeds.shape[1]
        # Shape: (batch_size, num_pooled, hidden_dim)

        # 3. Add separator token at end (learnable, like view_seperator)
        separator = self.separator_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        pooled_with_sep = torch.cat([pooled_embeds, separator], dim=1)
        # Shape: (batch_size, num_pooled+1, hidden_dim)

        # 4. Build input_ids sequence with placeholders for pooled positions
        num_compressed = num_pooled + 1  # pooled + separator

        if objective == 'lm':
            # [BOS] + [POOLED_PLACEHOLDERS] + [HYBRID (optional)] + [CONTINUATION]
            bos_tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long)
            pooled_placeholders = torch.full(
                (batch_size, num_compressed), self.pooled_token_id, dtype=torch.long
            )

            # Build input_ids and labels incrementally
            input_ids_parts = [bos_tokens, pooled_placeholders]
            label_parts = [
                torch.full((batch_size, 1), -100, dtype=torch.long),  # BOS masked
                torch.full((batch_size, num_compressed), -100, dtype=torch.long)  # Pooled masked
            ]

            # Add hybrid text if present
            if hybrid_text is not None and hybrid_text.numel() > 0:
                hybrid_len = hybrid_text.shape[1]
                input_ids_parts.append(hybrid_text)
                label_parts.append(torch.full((batch_size, hybrid_len), -100, dtype=torch.long))  # Hybrid masked

            # Add continuation
            input_ids_parts.append(target_tokens)
            label_parts.append(target_tokens)  # Continuation (target for loss)

            input_ids = torch.cat(input_ids_parts, dim=1)
            labels = torch.cat(label_parts, dim=1)

        else:  # reconstruction
            # [BOS] + [POOLED_PLACEHOLDERS] + [HYBRID (optional)] + [ORIGINAL_CONTEXT]
            original_context = target_tokens
            bos_tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long)
            pooled_placeholders = torch.full(
                (batch_size, num_compressed), self.pooled_token_id, dtype=torch.long
            )

            # Build input_ids and labels incrementally
            input_ids_parts = [bos_tokens, pooled_placeholders]
            label_parts = [
                torch.full((batch_size, 1), -100, dtype=torch.long),  # BOS masked
                torch.full((batch_size, num_compressed), -100, dtype=torch.long)  # Pooled masked
            ]

            # Add hybrid text if present
            if hybrid_text is not None and hybrid_text.numel() > 0:
                hybrid_len = hybrid_text.shape[1]
                input_ids_parts.append(hybrid_text)
                label_parts.append(torch.full((batch_size, hybrid_len), -100, dtype=torch.long))  # Hybrid masked

            # Add original context
            input_ids_parts.append(original_context)
            label_parts.append(original_context)  # Original context (reconstruction target)

            input_ids = torch.cat(input_ids_parts, dim=1)
            labels = torch.cat(label_parts, dim=1)

        # Move input_ids to device before embedding lookup
        input_ids = input_ids.to(self.device, non_blocking=True)

        # 5. Get initial embeddings (includes placeholders)
        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
        # Shape: (batch_size, seq_len, hidden_dim)

        # 6. Create mask for pooled positions (True = inject pooled embedding)
        mask_parts = [
            torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device),  # BOS
            torch.ones(batch_size, num_compressed, dtype=torch.bool, device=self.device)  # Pooled positions
        ]

        # Add hybrid text mask if present
        if hybrid_text is not None and hybrid_text.numel() > 0:
            hybrid_len = hybrid_text.shape[1]
            mask_parts.append(torch.zeros(batch_size, hybrid_len, dtype=torch.bool, device=self.device))  # Hybrid (no injection)

        # Add target tokens mask
        mask_parts.append(torch.zeros(batch_size, target_tokens.shape[1], dtype=torch.bool, device=self.device))  # Target tokens

        pooled_mask = torch.cat(mask_parts, dim=1)

        # 7. INJECT pooled embeddings at marked positions (same as vision!)
        # masked_scatter_ expects flattened source tensor
        inputs_embeds.masked_scatter_(
            pooled_mask.unsqueeze(-1),  # Broadcast mask to hidden_dim
            pooled_with_sep.reshape(-1, pooled_with_sep.shape[-1])  # Flatten batch+seq dims
        )

        # Move to device
        labels = labels.to(self.device, non_blocking=True)

        # 8. Bypass vision encoder with zero-valued images
        zero_global_expanded = self.zero_global.expand(batch_size, -1, -1, -1)
        images = [(self.empty_crop, zero_global_expanded)]
        images_spatial_crop = self.spatial_crop_cache.get(batch_size, [[1, 1]] * batch_size)

        # 9. Forward pass (pass input_ids for shape checks, use inputs_embeds for embeddings)
        outputs = self.model.forward(
            input_ids=input_ids,  # Keep for shape info (model uses inputs_embeds)
            inputs_embeds=inputs_embeds,
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
        hybrid_text_tokens: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate text from mean-pooled context

        Args:
            context_tokens: Context token IDs (shape: [seq_len])
            prompt_text: Optional additional text prompt
            hybrid_text_tokens: Optional hybrid text token IDs (shape: [seq_len]) to match training distribution
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            **kwargs: Additional generation parameters

        Returns:
            Generated text (decoded)
        """
        # Add batch dimension if needed
        if context_tokens.dim() == 1:
            context_tokens = context_tokens.unsqueeze(0)

        # Move context_tokens to device before embedding lookup
        context_tokens = context_tokens.to(self.device, non_blocking=True)

        # Get embeddings and pool
        context_embeds = self.model.model.get_input_embeddings()(context_tokens)
        pooled_embeds = self._sliding_window_mean_pool(context_embeds)
        num_pooled = pooled_embeds.shape[1]

        # Add separator
        separator = self.separator_embed.unsqueeze(0).unsqueeze(0)
        pooled_with_sep = torch.cat([pooled_embeds, separator], dim=1)
        num_compressed = num_pooled + 1

        # Build input_ids: [BOS] + [POOLED] + [HYBRID (optional)] + [PROMPT (optional)]
        bos_id = self.tokenizer.bos_token_id
        bos_tokens = torch.tensor([[bos_id]], dtype=torch.long)
        pooled_placeholders = torch.full((1, num_compressed), self.pooled_token_id, dtype=torch.long)

        input_ids_parts = [bos_tokens, pooled_placeholders]

        # Add hybrid text if provided (to match training distribution)
        if hybrid_text_tokens is not None and len(hybrid_text_tokens) > 0:
            # Ensure it's on CPU and convert to tensor if needed
            if not isinstance(hybrid_text_tokens, torch.Tensor):
                hybrid_text_tokens = torch.tensor(hybrid_text_tokens, dtype=torch.long)
            hybrid_text_tokens = hybrid_text_tokens.cpu().unsqueeze(0)  # Add batch dim
            input_ids_parts.append(hybrid_text_tokens)

        # Add additional user-provided prompt if specified
        if prompt_text:
            prompt_token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_tokens = torch.tensor([prompt_token_ids], dtype=torch.long)
            input_ids_parts.append(prompt_tokens)

        input_ids = torch.cat(input_ids_parts, dim=1)

        # Move input_ids to device before embedding lookup
        input_ids = input_ids.to(self.device, non_blocking=True)

        # Get embeddings and inject pooled
        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)

        # Create mask: inject pooled embeddings only at pooled positions
        # Calculate hybrid + prompt length
        rest_len = input_ids.shape[1] - 1 - num_compressed  # Total minus BOS minus pooled

        pooled_mask = torch.cat([
            torch.zeros(1, 1, dtype=torch.bool),  # BOS
            torch.ones(1, num_compressed, dtype=torch.bool),  # Pooled
            torch.zeros(1, rest_len, dtype=torch.bool)  # Hybrid + prompt
        ], dim=1).to(self.device)

        inputs_embeds.masked_scatter_(
            pooled_mask.unsqueeze(-1),
            pooled_with_sep.reshape(-1, pooled_with_sep.shape[-1])
        )

        # Move to device
        inputs_embeds = inputs_embeds.to(self.device)

        # Bypass vision encoder (create batch-1 tensors to match vision regime pattern)
        image_size = GUNDAM_PRESET['image_size']
        empty_crop = torch.zeros((1, 3, image_size, image_size),
                                 dtype=torch.bfloat16, device=self.device)
        zero_global_expanded = self.zero_global.expand(1, -1, -1, -1)
        images = [(empty_crop, zero_global_expanded)]
        images_spatial_crop = [[1, 1]]

        # Create images_seq_mask (all False since we don't have vision tokens, only pooled embeddings)
        batch_size = inputs_embeds.shape[0]
        seq_len = inputs_embeds.shape[1]
        images_seq_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device)

        # Generate using manual decode with KV-cache
        generated_ids = manual_generate_with_cache(
            model=self.model,
            tokenizer=self.tokenizer,
            initial_ids=input_ids,
            initial_embeds=inputs_embeds,
            images=images,
            images_spatial_crop=images_spatial_crop,
            images_seq_mask=images_seq_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            device=self.device
        )

        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text

