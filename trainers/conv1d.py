"""Conv1dPyramidCompressionTrainer trainer implementation."""

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


class Conv1dPyramidCompressionTrainer:
    """
    Learned Compression: 1D Convolutional Pyramid Encoder
    - Bypass vision encoder completely
    - Trainable conv1d pyramid for progressive downsampling
    - Trainable language decoder
    - Context: conv1d-compressed text embeddings
    - Loss: computed only on target tokens (continuation or reconstruction)

    This trainer implements a straightforward learned text compressor using stacked
    1D convolutions with stride-2 downsampling (similar to U-Net encoder half).
    It serves as a baseline showing what simple learned text compression can achieve
    compared to vision-based compression.
    """

    def __init__(
        self,
        model,
        tokenizer,
        compression_target: int,
        conv_kernel: int = 5,
        device: str = 'cuda'
    ):
        """
        Initialize conv1d pyramid compression trainer

        Args:
            model: DeepSeek-OCR model
            tokenizer: Tokenizer
            compression_target: Target number of compressed tokens (e.g., 111 to match vision-small)
            conv_kernel: Kernel size for conv layers (default: 5)
            device: Device to use
        """
        self.model = model
        self.tokenizer = tokenizer
        self.compression_target = compression_target
        self.conv_kernel = conv_kernel
        self.device = torch.device(device)

        # Validate parameters
        context_length = 1000  # Fixed for this dataset
        if compression_target < 1 or compression_target >= context_length:
            raise ValueError(
                f"compression_target must be in [1, {context_length}), got {compression_target}"
            )
        if conv_kernel < 1 or conv_kernel % 2 == 0:
            raise ValueError(f"conv_kernel must be odd and >= 1, got {conv_kernel}")

        # Get hidden dimension from model
        hidden_dim = model.config.hidden_size

        # Build conv1d pyramid layers
        # Strategy: Use exact native compression targets (powers of 2)
        # No adaptive pooling - compression_target must be a native value

        # Compute valid native targets
        # Native values: 500, 250, 125, 63, 32, 16, 8, ... (for input=1000)
        # Formula: ceil(context_length / 2^num_layers)
        valid_targets = []
        temp_length = context_length
        while temp_length > 1:
            valid_targets.append(temp_length)
            temp_length = (temp_length + 1) // 2

        # Validate compression_target is native
        if compression_target not in valid_targets:
            raise ValueError(
                f"compression_target={compression_target} is not a native target. "
                f"Native targets avoid wasteful up/downsampling. "
                f"Valid native targets for {context_length}-token input: {valid_targets[:8]}"
            )

        # Compute number of layers (since we validated, this will match exactly)
        current_length = context_length
        num_layers = 0
        while current_length > compression_target:
            current_length = (current_length + 1) // 2
            num_layers += 1

        # Build conv layers
        conv_layers = []
        for i in range(num_layers):
            conv_layers.extend([
                nn.Conv1d(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=conv_kernel,
                    stride=2,
                    padding=(conv_kernel - 1) // 2,  # Keep dimensions predictable
                    bias=False,
                    dtype=torch.bfloat16,
                    device=device
                ),
                nn.GroupNorm(1, hidden_dim, dtype=torch.bfloat16, device=device),
                nn.GELU()
            ])

        self.conv_layers = nn.ModuleList(conv_layers)

        # Total compressed tokens = target + separator
        self.compressed_tokens = compression_target + 1  # +1 for separator

        print(f"\n[Conv1D Pyramid] Configuration:")
        print(f"  Context length: {context_length}")
        print(f"  Compression target: {compression_target}")
        print(f"  Num conv layers: {num_layers} (stride-2)")
        print(f"  Kernel size: {conv_kernel}")
        print(f"  Native output length: {current_length} (no adaptive pooling)")
        print(f"  Compressed tokens (with separator): {self.compressed_tokens}")
        print(f"  Compression ratio: {context_length / compression_target:.2f}x")

        # Freeze vision encoder (won't be used)
        freeze_vision_encoder(model)
        unfreeze_decoder(model)

        # Register conv pyramid with model for checkpointing and optimizer
        # This allows gradients to flow and parameters to be saved/loaded properly
        self.model.conv1d_pyramid = self.conv_layers

        # Create learnable separator token (match other compression regimes)
        embed_std = 1 / torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        self.separator_embed = nn.Parameter(
            torch.randn(hidden_dim, device=self.device, dtype=torch.bfloat16) * embed_std
        )
        self.model.register_parameter('conv1d_separator', self.separator_embed)

        # LayerNorm to match decoder embedding scale (standard for multimodal injection)
        self.output_norm = nn.LayerNorm(hidden_dim, eps=1e-5, dtype=torch.bfloat16, device=device)
        self.model.register_module('conv1d_output_norm', self.output_norm)

        # Report parameter counts
        params = count_parameters(model)
        print(f"Total parameters: {params['total']:,}")
        print(f"Trainable: {params['trainable']:,} ({params['trainable_pct']:.1f}%)")
        print(f"  (includes conv pyramid + {hidden_dim} separator params)")
        print(f"Frozen: {params['frozen']:,}")

        # Special token ID for compressed positions (placeholder)
        self.compressed_token_id = IMAGE_TOKEN_ID

        # Zero images for bypassing vision encoder
        image_size = GUNDAM_PRESET['image_size']
        self.empty_crop = torch.zeros(
            0, 3, image_size, image_size,
            dtype=torch.bfloat16, device=self.device
        )

        # Global image token (must match GUNDAM_PRESET['base_size'] = 1024)
        global_size = 1024  # Match other regimes to avoid shape mismatches
        self.zero_global = torch.zeros((1, 3, global_size, global_size),
                                      dtype=torch.bfloat16, device=self.device)

        # Cache spatial crop for batch reuse
        self.spatial_crop_cache = {
            bs: [[1, 1] for _ in range(bs)] for bs in [1, 2, 4, 8, 16, 32]
        }

    def _conv1d_pyramid_compress(self, embeds: torch.Tensor) -> torch.Tensor:
        """
        Apply conv1d pyramid compression to embeddings

        Args:
            embeds: Token embeddings [batch_size, seq_len=1000, hidden_dim]

        Returns:
            Compressed embeddings [batch_size, compression_target, hidden_dim]
        """
        # Conv1D expects [batch, channels, length]
        # embeds is [batch, length, channels], so transpose
        x = embeds.transpose(1, 2)  # [B, hidden_dim, 1000]

        # Apply conv pyramid
        for layer in self.conv_layers:
            x = layer(x)

        # Transpose back to [B, compression_target, hidden_dim]
        compressed = x.transpose(1, 2)

        return compressed

    def forward(
        self,
        context_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        objective: str = 'lm'
    ) -> torch.Tensor:
        """
        Forward pass with conv1d pyramid compression

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

        # 2. Apply conv1d pyramid compression
        compressed_embeds = self._conv1d_pyramid_compress(context_embeds)
        # Shape: (batch_size, compression_target, hidden_dim)

        # 3. Add separator token at end (learnable, like other compression regimes)
        separator = self.separator_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        compressed_with_sep = torch.cat([compressed_embeds, separator], dim=1)
        # Shape: (batch_size, compression_target+1, hidden_dim)

        # Apply LayerNorm to match decoder embedding scale
        compressed_with_sep = self.output_norm(compressed_with_sep)

        # 4. Build input_ids sequence with placeholders for compressed positions
        num_compressed = self.compressed_tokens  # compression_target + 1 (separator)

        if objective == 'lm':
            # [BOS] + [COMPRESSED_PLACEHOLDERS] + [CONTINUATION]
            bos_tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=self.device)
            compressed_placeholders = torch.full(
                (batch_size, num_compressed), self.compressed_token_id, dtype=torch.long, device=self.device
            )
            input_ids = torch.cat([bos_tokens, compressed_placeholders, target_tokens], dim=1)

            # Labels: mask BOS and compressed positions, only train on continuation
            labels = torch.cat([
                torch.full((batch_size, 1), -100, dtype=torch.long, device=self.device),  # BOS masked
                torch.full((batch_size, num_compressed), -100, dtype=torch.long, device=self.device),  # Compressed masked
                target_tokens  # Continuation (target for loss)
            ], dim=1)

        else:  # reconstruction
            # [BOS] + [COMPRESSED_PLACEHOLDERS] + [ORIGINAL_CONTEXT]
            original_context = target_tokens
            bos_tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=self.device)
            compressed_placeholders = torch.full(
                (batch_size, num_compressed), self.compressed_token_id, dtype=torch.long, device=self.device
            )
            input_ids = torch.cat([bos_tokens, compressed_placeholders, original_context], dim=1)

            # Labels: mask BOS and compressed positions, reconstruct original context
            labels = torch.cat([
                torch.full((batch_size, 1), -100, dtype=torch.long, device=self.device),  # BOS masked
                torch.full((batch_size, num_compressed), -100, dtype=torch.long, device=self.device),  # Compressed masked
                original_context  # Original context (reconstruction target)
            ], dim=1)

        # Move input_ids to device before embedding lookup
        input_ids = input_ids.to(self.device, non_blocking=True)

        # 5. Get initial embeddings (includes placeholders)
        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)
        # Shape: (batch_size, seq_len, hidden_dim)

        # 6. Create mask for compressed positions (True = inject compressed embedding)
        compressed_mask = torch.cat([
            torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device),  # BOS
            torch.ones(batch_size, num_compressed, dtype=torch.bool, device=self.device),  # Compressed positions
            torch.zeros(batch_size, target_tokens.shape[1], dtype=torch.bool, device=self.device)  # Target tokens
        ], dim=1)

        # 7. INJECT compressed embeddings at marked positions (same technique as other regimes!)
        inputs_embeds.masked_scatter_(
            compressed_mask.unsqueeze(-1),  # Broadcast mask to hidden_dim
            compressed_with_sep.reshape(-1, compressed_with_sep.shape[-1])  # Flatten batch+seq dims
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
        Generate text from conv1d-compressed context

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

        # Move to device
        context_tokens = context_tokens.to(self.device)

        # Get embeddings and compress
        context_embeds = self.model.model.get_input_embeddings()(context_tokens)
        compressed_embeds = self._conv1d_pyramid_compress(context_embeds)

        # Add separator
        separator = self.separator_embed.unsqueeze(0).unsqueeze(0)
        compressed_with_sep = torch.cat([compressed_embeds, separator], dim=1)

        # Apply LayerNorm to match decoder embedding scale
        compressed_with_sep = self.output_norm(compressed_with_sep)

        num_compressed = compressed_with_sep.shape[1]

        # Build input sequence: [BOS] + [COMPRESSED] + [PROMPT]
        bos_id = self.tokenizer.bos_token_id
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)

        bos_tokens = torch.tensor([[bos_id]], dtype=torch.long, device=self.device)
        compressed_placeholders = torch.full(
            (1, num_compressed), self.compressed_token_id, dtype=torch.long, device=self.device
        )
        input_ids = torch.cat([bos_tokens, compressed_placeholders, prompt_tensor], dim=1)

        # Get embeddings and inject compressed
        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)

        compressed_mask = torch.cat([
            torch.zeros(1, 1, dtype=torch.bool),  # BOS
            torch.ones(1, num_compressed, dtype=torch.bool),  # Compressed
            torch.zeros(1, input_ids.shape[1] - 1 - num_compressed, dtype=torch.bool)  # Prompt
        ], dim=1).to(self.device)

        inputs_embeds.masked_scatter_(
            compressed_mask.unsqueeze(-1),
            compressed_with_sep.reshape(-1, compressed_with_sep.shape[-1])
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
