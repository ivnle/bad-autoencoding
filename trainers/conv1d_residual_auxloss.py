"""Conv1dResidualAuxLossTrainer with auxiliary losses at intermediate stages."""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import IMAGE_TOKEN_ID, GUNDAM_PRESET
from .utils import (
    freeze_vision_encoder,
    unfreeze_decoder,
    count_parameters,
    manual_generate_with_cache
)


class ResidualConvBlock(nn.Module):
    """Residual block with two convolutions and skip connection.

    Architecture:
        Main path: Conv1d(stride=1) → GroupNorm → GELU → Conv1d(stride=2) → GroupNorm
        Skip path: AdaptiveAvgPool1d (matches main path output length)
        Output: (main + skip) → GELU

    This preserves information through the downsample step via the skip connection,
    improving gradient flow and reconstruction quality.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        device: str = 'cuda'
    ):
        """Initialize residual conv block.

        Args:
            channels: Number of input/output channels (maintained throughout)
            kernel_size: Kernel size for first conv (second conv uses kernel=3)
            device: Device to place layers on
        """
        super().__init__()

        # Main path: two convolutions with normalization
        self.conv1 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=False,
            dtype=torch.bfloat16,
            device=device
        )
        self.norm1 = nn.GroupNorm(1, channels, dtype=torch.bfloat16, device=device)
        self.gelu1 = nn.GELU()

        self.conv2 = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
            dtype=torch.bfloat16,
            device=device
        )
        self.norm2 = nn.GroupNorm(1, channels, dtype=torch.bfloat16, device=device)

        # Note: Skip connection uses F.adaptive_avg_pool1d() in forward() (no module needed)

        # Final activation after residual addition
        self.gelu_final = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor [batch, channels, length]

        Returns:
            Output tensor [batch, channels, length//2] (downsampled by 2)
        """
        # Main path
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.gelu1(out)
        out = self.conv2(out)
        out = self.norm2(out)

        # Skip connection - adaptive pooling to match main path output length
        skip = F.adaptive_avg_pool1d(x, out.shape[2])

        # Residual addition + final activation
        out = out + skip
        out = self.gelu_final(out)

        return out


class Conv1dResidualAuxLossTrainer:
    """
    Learned Compression with Auxiliary Losses at Intermediate Stages

    Extends conv1d_residual by adding reconstruction losses at each compression stage
    to address gradient dilution in deep networks.

    For K=63 (4 layers): computes losses at 500, 250, 125, and 63 tokens
    - Auxiliary losses (500, 250, 125): provide direct gradient signal to early layers
    - Final loss (63): the target compression we care about at inference

    All losses contribute to a single optimizer step via weighted combination.

    IMPORTANT: This regime ONLY supports objective='reconstruction'. The 'lm' objective
    is incompatible with the multi-stage auxiliary loss architecture because it requires
    fixed target tokens at all compression stages.
    """

    def __init__(
        self,
        model,
        tokenizer,
        compression_target: int,
        conv_kernel: int = 5,
        device: str = 'cuda',
        aux_loss_weight: float = 0.5
    ):
        """
        Initialize conv1d pyramid compression trainer with auxiliary losses

        Args:
            model: DeepSeek-OCR model
            tokenizer: Tokenizer
            compression_target: Target number of compressed tokens (e.g., 63 for 16x)
            conv_kernel: Kernel size for conv layers (default: 5)
            device: Device to use
            aux_loss_weight: Weight for auxiliary losses (0.0=only final, 1.0=only aux, 0.5=equal)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.compression_target = compression_target
        self.conv_kernel = conv_kernel
        self.device = torch.device(device)
        self.aux_loss_weight = aux_loss_weight

        # Validate parameters
        context_length = 1000  # Fixed for this dataset
        if compression_target < 1 or compression_target >= context_length:
            raise ValueError(
                f"compression_target must be in [1, {context_length}), got {compression_target}"
            )
        if conv_kernel < 1 or conv_kernel % 2 == 0:
            raise ValueError(f"conv_kernel must be odd and >= 1, got {conv_kernel}")
        if not (0.0 <= aux_loss_weight <= 1.0):
            raise ValueError(f"aux_loss_weight must be in [0, 1], got {aux_loss_weight}")

        # Get hidden dimension from model
        hidden_dim = model.config.hidden_size

        # Compute valid native targets
        valid_targets = []
        temp_length = context_length
        while temp_length > 1:
            valid_targets.append(temp_length)
            temp_length = (temp_length + 1) // 2

        # Validate compression_target is native
        if compression_target not in valid_targets:
            raise ValueError(
                f"compression_target={compression_target} is not a native target. "
                f"Valid native targets for {context_length}-token input: {valid_targets[:8]}"
            )

        # Compute number of layers
        current_length = context_length
        num_layers = 0
        while current_length > compression_target:
            current_length = (current_length + 1) // 2
            num_layers += 1

        # Build residual conv blocks
        conv_blocks = []
        for i in range(num_layers):
            conv_blocks.append(
                ResidualConvBlock(
                    channels=hidden_dim,
                    kernel_size=conv_kernel,
                    device=device
                )
            )

        self.conv_blocks = nn.ModuleList(conv_blocks)

        print(f"\n[Conv1D Residual AuxLoss] Configuration:")
        print(f"  Context length: {context_length}")
        print(f"  Compression target: {compression_target}")
        print(f"  Num residual blocks: {num_layers} (each downsamples by 2x)")
        print(f"  Auxiliary loss weight: {aux_loss_weight:.2f}")
        print(f"  Kernel size: {conv_kernel}")
        print(f"  Compression ratio: {context_length / compression_target:.2f}x")

        # Compute intermediate stages for reporting
        intermediate_lengths = []
        temp = context_length
        for _ in range(num_layers):
            temp = (temp + 1) // 2
            intermediate_lengths.append(temp)
        print(f"  Intermediate stages: {intermediate_lengths}")
        print(f"    → Auxiliary losses at: {intermediate_lengths[:-1]}")
        print(f"    → Final loss at: {intermediate_lengths[-1]}")

        # Freeze vision encoder (won't be used)
        freeze_vision_encoder(model)
        unfreeze_decoder(model)

        # Register conv pyramid with model for checkpointing and optimizer
        # Use same parameter names as conv1d_residual for checkpoint compatibility
        self.model.conv1d_residual_pyramid = self.conv_blocks

        # Create learnable separator token
        embed_std = 1 / torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        self.separator_embed = nn.Parameter(
            torch.randn(hidden_dim, device=self.device, dtype=torch.bfloat16) * embed_std
        )
        self.model.register_parameter('conv1d_residual_separator', self.separator_embed)

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
        global_size = 1024
        self.zero_global = torch.zeros((1, 3, global_size, global_size),
                                      dtype=torch.bfloat16, device=self.device)

        # Cache spatial crop for batch reuse
        self.spatial_crop_cache = {
            bs: [[1, 1] for _ in range(bs)] for bs in [1, 2, 4, 8, 16, 32]
        }

    def _compress_with_intermediates(self, embeds: torch.Tensor) -> list[torch.Tensor]:
        """
        Apply conv1d pyramid compression, capturing output at each stage.

        Args:
            embeds: Token embeddings [batch_size, seq_len=1000, hidden_dim]

        Returns:
            List of intermediate outputs, one per layer
            For K=63: [[B, 500, D], [B, 250, D], [B, 125, D], [B, 63, D]]
        """
        # Conv1D expects [batch, channels, length]
        x = embeds.transpose(1, 2)  # [B, hidden_dim, 1000]

        intermediates = []
        for block in self.conv_blocks:
            x = block(x)
            intermediates.append(x.transpose(1, 2))  # Store as [B, seq, D]

        return intermediates

    def _prepare_dummy_images(self, batch_size: int):
        """Prepare dummy images to bypass vision encoder."""
        zero_global_expanded = self.zero_global.expand(batch_size, -1, -1, -1)
        images = [(self.empty_crop, zero_global_expanded)]
        images_spatial_crop = self.spatial_crop_cache.get(batch_size, [[1, 1]] * batch_size)
        return images, images_spatial_crop

    def forward(
        self,
        context_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        objective: str = 'reconstruction'
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with auxiliary losses at intermediate compression stages.

        Args:
            context_tokens: Context token IDs (shape: [batch_size, seq_len])
            target_tokens: Target token IDs (must be reconstruction objective)
            objective: Training objective (only 'reconstruction' supported)

        Returns:
            Tuple of (weighted_loss, labels)
        """
        if objective != 'reconstruction':
            raise ValueError(
                f"Regime 'conv1d_residual_auxloss' only supports objective='reconstruction', "
                f"got '{objective}'"
            )

        batch_size = context_tokens.shape[0]

        # Move to device
        context_tokens = context_tokens.to(self.device, non_blocking=True)
        target_tokens = target_tokens.to(self.device, non_blocking=True)

        # 1. Encode context once, capture all intermediate representations
        context_embeds = self.model.model.get_input_embeddings()(context_tokens)
        intermediates = self._compress_with_intermediates(context_embeds)
        # intermediates[i].shape = [B, num_tokens_i, hidden_dim]
        # For K=63: num_tokens = [500, 250, 125, 63]

        stage_losses = []

        # 2. Four sequential forward passes through decoder (one per stage)
        for stage_idx, compressed in enumerate(intermediates):
            num_compressed_tokens = compressed.shape[1]

            # Add separator token
            separator_embed = self.separator_embed.unsqueeze(0).expand(batch_size, -1, -1)
            compressed_with_sep = torch.cat([compressed, separator_embed], dim=1)
            # Shape: [B, num_compressed_tokens + 1, D]

            num_compressed = num_compressed_tokens + 1  # +1 for separator

            # Build input_ids: [BOS] + [COMPRESSED_PLACEHOLDERS] + [TARGET]
            bos_id = self.tokenizer.bos_token_id
            bos_tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=self.device)
            compressed_placeholders = torch.full(
                (batch_size, num_compressed),
                self.compressed_token_id,
                dtype=torch.long,
                device=self.device
            )
            input_ids = torch.cat([bos_tokens, compressed_placeholders, target_tokens], dim=1)
            # Shape: [B, 1 + num_compressed + target_len]

            # Get base embeddings
            inputs_embeds = self.model.model.get_input_embeddings()(input_ids)

            # Create mask for compressed positions (True = inject compressed embedding)
            compressed_mask = torch.cat([
                torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device),  # BOS
                torch.ones(batch_size, num_compressed, dtype=torch.bool, device=self.device),  # Compressed
                torch.zeros(batch_size, target_tokens.shape[1], dtype=torch.bool, device=self.device)  # Target
            ], dim=1)

            # Inject compressed embeddings using masked_scatter
            inputs_embeds.masked_scatter_(
                compressed_mask.unsqueeze(-1),
                compressed_with_sep.reshape(-1, compressed_with_sep.shape[-1])
            )

            # Build labels: mask BOS and compressed positions, train only on target
            labels = torch.cat([
                torch.full((batch_size, 1), -100, dtype=torch.long, device=self.device),  # BOS masked
                torch.full((batch_size, num_compressed), -100, dtype=torch.long, device=self.device),  # Compressed masked
                target_tokens  # Target (reconstruction)
            ], dim=1)

            # Prepare dummy images (decoder ignores but required by API)
            images, images_spatial_crop = self._prepare_dummy_images(batch_size)

            # Forward pass for this stage
            outputs = self.model.forward(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                images=images,
                images_spatial_crop=images_spatial_crop,
                labels=labels,
                return_dict=True
            )

            stage_losses.append(outputs.loss)

        # 3. Combine losses with weighting
        # aux_weight controls contribution of auxiliary stages vs final stage
        aux_losses = stage_losses[:-1]  # First N-1 stages (e.g., 500, 250, 125)
        final_loss = stage_losses[-1]   # Last stage (e.g., 63)

        if len(aux_losses) > 0:
            avg_aux_loss = sum(aux_losses) / len(aux_losses)
            total_loss = (1 - self.aux_loss_weight) * final_loss + \
                         self.aux_loss_weight * avg_aux_loss
        else:
            # Edge case: only 1 layer (no auxiliary losses)
            total_loss = final_loss

        # Return same signature as other trainers
        return total_loss, labels

    def generate_text(
        self,
        context_tokens: torch.Tensor,
        prompt_text: str = "",
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate text from conv1d-compressed context (uses final stage only).

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

        # Get embeddings and compress (use only final stage)
        context_embeds = self.model.model.get_input_embeddings()(context_tokens)
        intermediates = self._compress_with_intermediates(context_embeds)
        compressed_embeds = intermediates[-1]  # Use final stage only

        # Add separator
        separator = self.separator_embed.unsqueeze(0).unsqueeze(0)
        compressed_with_sep = torch.cat([compressed_embeds, separator], dim=1)

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
