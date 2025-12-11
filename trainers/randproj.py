"""RandomProjectionCompressionTrainer trainer implementation."""

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


class RandomProjectionCompressionTrainer:
    """
    Simple Encoder: Random Projection (Johnson-Lindenstrauss)
    - Bypass vision encoder completely
    - Trainable language decoder
    - Context: random-projected text embeddings (optionally trainable)
    - Loss: computed only on target tokens (continuation or reconstruction)

    This trainer tests whether random projection can match vision compression
    for language modeling tasks.
    """

    def __init__(
        self,
        model,
        tokenizer,
        projection_dim: int,
        train_encoder: bool = False,
        device: str = 'cuda'
    ):
        """
        Initialize random projection compression trainer

        Args:
            model: DeepSeek-OCR model
            tokenizer: Tokenizer
            projection_dim: Number of projected dimensions (required)
            train_encoder: Make projection matrix trainable (default: frozen like JL)
            device: Device to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.projection_dim = projection_dim
        self.train_encoder = train_encoder

        # Validate parameters
        context_length = 1000  # Fixed for this dataset
        if projection_dim < 1 or projection_dim > context_length:
            raise ValueError(f"projection_dim must be 1-{context_length}, got {projection_dim}")

        self.context_length = context_length
        self.compressed_tokens = self.projection_dim + 1  # +1 for separator

        # Freeze vision encoder (won't be used)
        freeze_vision_encoder(model)
        unfreeze_decoder(model)

        # Initialize projection matrix: N(0, 1/√S) to preserve embedding variance
        # Note: Standard JL uses 1/√K for feature projection, but we project sequence dimension
        # Each projected token sums S inputs, so use 1/√S to maintain unit variance
        proj_std = 1 / math.sqrt(context_length)
        proj_matrix = (
            torch.randn(context_length, projection_dim, dtype=torch.float32)
            * proj_std
        ).to(dtype=torch.bfloat16, device=self.device)

        # Conditional trainability (same name for checkpoint compatibility)
        if train_encoder:
            # Trainable projection matrix
            self.projection_matrix = nn.Parameter(proj_matrix)
            self.model.register_parameter('randproj_matrix', self.projection_matrix)
        else:
            # Fixed projection (true Johnson-Lindenstrauss)
            self.projection_matrix = proj_matrix
            self.model.register_buffer('randproj_matrix', self.projection_matrix)

        # Create learnable separator embedding (match meanpool/subsample pattern)
        hidden_dim = model.config.hidden_size
        embed_std = 1 / torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        self.separator_embed = nn.Parameter(
            torch.randn(hidden_dim, device=self.device, dtype=torch.bfloat16) * embed_std
        )

        # Register as model parameter for checkpointing
        self.model.register_parameter('randproj_separator', self.separator_embed)

        # Special token ID for projected positions (placeholder)
        # Use IMAGE_TOKEN_ID since model knows how to handle masked positions
        self.projected_token_id = IMAGE_TOKEN_ID

        # Print compression configuration
        params = count_parameters(model)
        print(f"\n[Simple Encoder: Random Projection]")
        print(f"Projection matrix: {'trainable' if train_encoder else 'frozen (Johnson-Lindenstrauss)'}")
        print(f"Compression: 1000 → {self.compressed_tokens} tokens")
        print(f"  - {self.projection_dim} projected dimensions")
        print(f"  - Separator: 1 token (learnable)")
        if train_encoder:
            proj_params = context_length * projection_dim
            print(f"Total parameters: {params['total']:,}")
            print(f"Trainable: {params['trainable']:,} ({params['trainable_pct']:.1f}%)")
            print(f"  (includes {proj_params:,} projection params + {hidden_dim} separator params)")
        else:
            print(f"Total parameters: {params['total']:,}")
            print(f"Trainable: {params['trainable']:,} ({params['trainable_pct']:.1f}%)")
            print(f"  (includes {hidden_dim} separator params)")
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

    def _random_project(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        Apply random projection to embeddings

        CRITICAL: Use matmul, NOT einsum (5x faster based on benchmarking)
        Projects along sequence dimension (contracts 1000 tokens into k tokens)
        Each output token is a weighted combination of all 1000 input embeddings

        Args:
            embeds: Token embeddings (shape: [batch_size, seq_len=1000, hidden_dim])

        Returns:
            Projected embeddings (shape: [batch_size, projection_dim, hidden_dim])
        """
        batch_size, seq_len, hidden_dim = embeds.shape

        # Validate sequence length
        if seq_len != self.context_length:
            raise ValueError(
                f"Expected sequence length {self.context_length}, got {seq_len}. "
                f"Context is incompatible with random projection."
            )

        # Apply random projection via matmul (verified 5x faster than einsum)
        # embeds: [B, 1000, D]
        # projection_matrix: [1000, k]
        # Result: [B, k, D]
        # Transpose embeds: [B, D, 1000]
        # Matmul: [B, D, 1000] @ [1000, k] = [B, D, k]
        # Transpose back: [B, k, D]
        projected = embeds.transpose(1, 2) @ self.projection_matrix
        projected = projected.transpose(1, 2)

        return projected

    def forward(
        self,
        context_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        objective: str = 'lm'
    ) -> torch.Tensor:
        """
        Forward pass with random projection embedding injection

        Args:
            context_tokens: Context token IDs (shape: [batch_size, seq_len])
            target_tokens: Target token IDs (continuation for lm, context for reconstruction)
            objective: Training objective ('lm' or 'reconstruction')

        Returns:
            Tuple of (loss, labels)
        """
        batch_size = context_tokens.shape[0]
        bos_id = self.tokenizer.bos_token_id

        # Move to device before embedding lookup
        context_tokens = context_tokens.to(self.device, non_blocking=True)
        target_tokens = target_tokens.to(self.device, non_blocking=True)

        # 1. Get embeddings for context tokens
        context_embeds = self.model.model.get_input_embeddings()(context_tokens)
        # Shape: (batch_size, context_len=1000, hidden_dim)

        # 2. Apply random projection (matmul - 5x faster than einsum)
        projected_embeds = self._random_project(context_embeds)
        num_projected = projected_embeds.shape[1]  # = projection_dim
        # Shape: (batch_size, projection_dim, hidden_dim)

        # 3. Add separator token at end (learnable, like meanpool/subsample)
        separator = self.separator_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        projected_with_sep = torch.cat([projected_embeds, separator], dim=1)
        # Shape: (batch_size, projection_dim+1, hidden_dim)

        # 4. Build input_ids sequence with placeholders for projected positions
        num_compressed = num_projected + 1  # projected + separator

        if objective == 'lm':
            # [BOS] + [PROJECTED_PLACEHOLDERS] + [CONTINUATION]
            bos_tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=self.device)
            projected_placeholders = torch.full(
                (batch_size, num_compressed), self.projected_token_id, dtype=torch.long, device=self.device
            )
            input_ids = torch.cat([bos_tokens, projected_placeholders, target_tokens], dim=1)

            # Labels: mask BOS and projected positions, only train on continuation
            labels = torch.cat([
                torch.full((batch_size, 1), -100, dtype=torch.long, device=self.device),  # BOS masked
                torch.full((batch_size, num_compressed), -100, dtype=torch.long, device=self.device),  # Projected masked
                target_tokens  # Continuation (target for loss)
            ], dim=1)

        else:  # reconstruction
            # [BOS] + [PROJECTED_PLACEHOLDERS] + [ORIGINAL_CONTEXT]
            original_context = target_tokens
            bos_tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=self.device)
            projected_placeholders = torch.full(
                (batch_size, num_compressed), self.projected_token_id, dtype=torch.long, device=self.device
            )
            input_ids = torch.cat([bos_tokens, projected_placeholders, original_context], dim=1)

            # Labels: mask BOS and projected positions, reconstruct original context
            labels = torch.cat([
                torch.full((batch_size, 1), -100, dtype=torch.long, device=self.device),  # BOS masked
                torch.full((batch_size, num_compressed), -100, dtype=torch.long, device=self.device),  # Projected masked
                original_context  # Original context (reconstruction target)
            ], dim=1)

        # Move input_ids to device before embedding lookup
        input_ids = input_ids.to(self.device, non_blocking=True)

        # 5. Get initial embeddings (includes placeholders)
        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
        # Shape: (batch_size, seq_len, hidden_dim)

        # 6. Create mask for projected positions (True = inject projected embedding)
        projected_mask = torch.cat([
            torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device),  # BOS
            torch.ones(batch_size, num_compressed, dtype=torch.bool, device=self.device),  # Projected positions
            torch.zeros(batch_size, target_tokens.shape[1], dtype=torch.bool, device=self.device)  # Target tokens
        ], dim=1)

        # 7. INJECT projected embeddings at marked positions (same technique as meanpool!)
        # masked_scatter_ expects flattened source tensor
        inputs_embeds.masked_scatter_(
            projected_mask.unsqueeze(-1),  # Broadcast mask to hidden_dim
            projected_with_sep.reshape(-1, projected_with_sep.shape[-1])  # Flatten batch+seq dims
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
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate text from random-projected context

        Args:
            context_tokens: Context token IDs (shape: [seq_len])
            prompt_text: Optional additional text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            **kwargs: Additional generation parameters

        Returns:
            Generated text (decoded)
        """
        # Add batch dimension if needed
        if context_tokens.dim() == 1:
            context_tokens = context_tokens.unsqueeze(0)

        # Move to device before embedding lookup
        context_tokens = context_tokens.to(self.device, non_blocking=True)

        # Get embeddings and project
        context_embeds = self.model.model.get_input_embeddings()(context_tokens)
        projected_embeds = self._random_project(context_embeds)
        num_projected = projected_embeds.shape[1]

        # Add separator
        separator = self.separator_embed.unsqueeze(0).unsqueeze(0)
        projected_with_sep = torch.cat([projected_embeds, separator], dim=1)
        num_compressed = num_projected + 1

        # Build input_ids with optional prompt
        bos_id = self.tokenizer.bos_token_id
        bos_tokens = torch.tensor([[bos_id]], dtype=torch.long)
        projected_placeholders = torch.full((1, num_compressed), self.projected_token_id, dtype=torch.long)

        if prompt_text:
            prompt_token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_tokens = torch.tensor([prompt_token_ids], dtype=torch.long)
            input_ids = torch.cat([bos_tokens, projected_placeholders, prompt_tokens], dim=1)
        else:
            input_ids = torch.cat([bos_tokens, projected_placeholders], dim=1)

        # Move to device before embedding lookup
        input_ids = input_ids.to(self.device, non_blocking=True)

        # Get embeddings and inject projected
        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)

        projected_mask = torch.cat([
            torch.zeros(1, 1, dtype=torch.bool),  # BOS
            torch.ones(1, num_compressed, dtype=torch.bool),  # Projected
            torch.zeros(1, input_ids.shape[1] - 1 - num_compressed, dtype=torch.bool)  # Prompt
        ], dim=1).to(self.device)

        inputs_embeds.masked_scatter_(
            projected_mask.unsqueeze(-1),
            projected_with_sep.reshape(-1, projected_with_sep.shape[-1])
        )

        # Move to device
        inputs_embeds = inputs_embeds.to(self.device)

        # Bypass vision encoder
        image_size = GUNDAM_PRESET['image_size']
        empty_crop = torch.zeros((1, 3, image_size, image_size),
                                 dtype=torch.bfloat16, device=self.device)
        zero_global_expanded = self.zero_global.expand(1, -1, -1, -1)
        images = [(empty_crop, zero_global_expanded)]
        images_spatial_crop = [[1, 1]]

        # Create images_seq_mask (all False since we don't have vision tokens)
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

