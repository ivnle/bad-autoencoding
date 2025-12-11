"""Conv1dResidualCompressionTrainer with residual blocks."""

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


class Conv1dResidualCompressionTrainer:
    """
    Learned Compression: 1D Convolutional Pyramid with Residual Blocks
    - Bypass vision encoder completely
    - Trainable conv1d pyramid with residual skip connections
    - Trainable language decoder
    - Context: conv1d-compressed text embeddings
    - Loss: computed only on target tokens (continuation or reconstruction)

    This trainer implements learned text compression using stacked residual blocks,
    each containing two convolutions with a skip connection. The skip connections
    preserve information through downsampling for better reconstruction quality
    and improved gradient flow during training.
    """

    def __init__(
        self,
        model,
        tokenizer,
        compression_target: int,
        conv_kernel: int = 5,
        device: str = 'cuda',
        hybrid_text_tokens: int = 0,
        train_encoder: bool = True
    ):
        """
        Initialize conv1d pyramid compression trainer

        Args:
            model: DeepSeek-OCR model
            tokenizer: Tokenizer
            compression_target: Target number of compressed tokens (e.g., 111 to match vision-small)
            conv_kernel: Kernel size for conv layers (default: 5)
            device: Device to use
            hybrid_text_tokens: Number of uncompressed text tokens to append after compressed tokens (default: 0)
                - 0: Compression-only mode (compressed tokens only)
                - >0: Hybrid mode (compressed tokens + last K explicit tokens from context)
                Note: These tokens are ALSO included in compression (redundant encoding is intentional)
                This provides both compressed and explicit representations for better quality.
            train_encoder: Make compression encoder (conv pyramid + separator) trainable (default: True)
                - True: Trainable encoder (learn compression from data)
                - False: Frozen encoder (for two-stage training or analysis)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.compression_target = compression_target
        self.conv_kernel = conv_kernel
        self.device = torch.device(device)
        self.hybrid_text_tokens = hybrid_text_tokens
        self.train_encoder = train_encoder

        # Validate parameters
        context_length = 1000  # Fixed for this dataset

        if hybrid_text_tokens < 0:
            raise ValueError(f"hybrid_text_tokens must be >= 0, got {hybrid_text_tokens}")
        if hybrid_text_tokens >= context_length:
            raise ValueError(
                f"hybrid_text_tokens ({hybrid_text_tokens}) must be < context_length ({context_length})"
            )
        if compression_target < 1 or compression_target >= context_length:
            raise ValueError(
                f"compression_target must be in [1, {context_length}), got {compression_target}"
            )
        if conv_kernel < 1 or conv_kernel % 2 == 0:
            raise ValueError(f"conv_kernel must be odd and >= 1, got {conv_kernel}")

        # Get hidden dimension from model
        hidden_dim = model.config.hidden_size

        # Build conv1d pyramid layers
        # Strategy: Use exact native compression targets
        # AdaptiveAvgPool1d in skip connections enables all native targets (even and odd)

        # Compute valid native targets
        # Native values: 500, 250, 125, 63, 32, 16, 8, ... (for input=1000)
        # Formula: ceil(context_length / 2^num_layers)
        # Note: Adaptive pooling works seamlessly for both even targets (500, 250, 125)
        # and odd targets (63, 32, 16) by adjusting window sizes as needed
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

        # Total compressed tokens = target + separator
        self.compressed_tokens = compression_target + 1  # +1 for separator

        print(f"\n[Conv1D Residual Pyramid] Configuration:")
        print(f"  Context length: {context_length}")
        print(f"  Compression target: {compression_target}")
        print(f"  Num residual blocks: {num_layers} (each downsamples by 2x)")
        print(f"  Kernel size: {conv_kernel} (first conv per block)")
        print(f"  Skip connections: AdaptiveAvgPool1d (matches main path)")
        print(f"  Native output length: {current_length}")
        print(f"  Compressed tokens (with separator): {self.compressed_tokens}")
        print(f"  Compression ratio: {context_length / compression_target:.2f}x")

        # Freeze vision encoder (won't be used)
        freeze_vision_encoder(model)
        unfreeze_decoder(model)

        # Register conv pyramid with model for checkpointing and optimizer
        # This allows gradients to flow and parameters to be saved/loaded properly
        self.model.conv1d_residual_pyramid = self.conv_blocks

        # Create learnable separator token (match other compression regimes)
        embed_std = 1 / torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        self.separator_embed = nn.Parameter(
            torch.randn(hidden_dim, device=self.device, dtype=torch.bfloat16) * embed_std
        )
        self.model.register_parameter('conv1d_residual_separator', self.separator_embed)

        # Conditionally freeze/unfreeze compression encoder
        if train_encoder:
            # Trainable mode (default): encoder parameters remain trainable
            trainable_encoder_params = (
                sum(p.numel() for p in self.conv_blocks.parameters() if p.requires_grad) +
                self.separator_embed.numel()
            )
            print(f"Trainable compression encoder: {trainable_encoder_params:,} parameters")
        else:
            # Frozen mode: freeze conv pyramid and separator for two-stage training
            for param in self.conv_blocks.parameters():
                param.requires_grad = False
            self.separator_embed.requires_grad = False

            frozen_encoder_params = (
                sum(p.numel() for p in self.conv_blocks.parameters()) +
                self.separator_embed.numel()
            )
            print(f"Frozen compression encoder: {frozen_encoder_params:,} parameters")

        # Report parameter counts
        params = count_parameters(model)
        encoder_mode = "trainable" if train_encoder else "frozen"
        print(f"Total parameters: {params['total']:,}")
        print(f"Trainable: {params['trainable']:,} ({params['trainable_pct']:.1f}%)")
        if train_encoder:
            print(f"  (includes conv pyramid + {hidden_dim} separator params)")
        print(f"Frozen: {params['frozen']:,}")
        print(f"Compression encoder mode: {encoder_mode}")

        # Log hybrid mode configuration
        if self.hybrid_text_tokens > 0:
            total_context_tokens = self.compressed_tokens + self.hybrid_text_tokens
            print(f"Hybrid mode: {self.compressed_tokens} compressed + {self.hybrid_text_tokens} text = {total_context_tokens} total context tokens")

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

    def _conv1d_residual_compress(self, embeds: torch.Tensor) -> torch.Tensor:
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

        # Apply residual conv blocks
        for block in self.conv_blocks:
            x = block(x)

        # Transpose back to [B, compression_target, hidden_dim]
        compressed = x.transpose(1, 2)

        return compressed

    def forward(
        self,
        context_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        hybrid_text: Optional[torch.Tensor] = None,
        objective: str = 'lm'
    ) -> torch.Tensor:
        """
        Forward pass with conv1d pyramid compression

        Args:
            context_tokens: Context token IDs (shape: [batch_size, seq_len])
            target_tokens: Target token IDs (continuation for lm, context for reconstruction)
            hybrid_text: Optional hybrid text token IDs (shape: [batch_size, hybrid_len])
            objective: Training objective ('lm' or 'reconstruction')

        Returns:
            Tuple of (loss, labels)
        """
        batch_size = context_tokens.shape[0]
        bos_id = self.tokenizer.bos_token_id

        # Move to device before embedding lookup
        context_tokens = context_tokens.to(self.device, non_blocking=True)
        # Keep target_tokens on CPU for concatenation (moved to device later)

        # 1. Get embeddings for context tokens
        context_embeds = self.model.model.get_input_embeddings()(context_tokens)
        # Shape: (batch_size, context_len=1000, hidden_dim)

        # 2. Apply conv1d pyramid compression
        compressed_embeds = self._conv1d_residual_compress(context_embeds)
        # Shape: (batch_size, compression_target, hidden_dim)

        # 3. Add separator token at end (learnable, like other compression regimes)
        separator = self.separator_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1)
        compressed_with_sep = torch.cat([compressed_embeds, separator], dim=1)
        # Shape: (batch_size, compression_target+1, hidden_dim)

        # 4. Build input_ids sequence with placeholders for compressed positions
        num_compressed = self.compressed_tokens  # compression_target + 1 (separator)

        if objective == 'lm':
            # [BOS] + [COMPRESSED_PLACEHOLDERS] + [HYBRID (optional)] + [CONTINUATION]
            bos_tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long)
            compressed_placeholders = torch.full(
                (batch_size, num_compressed), self.compressed_token_id, dtype=torch.long
            )

            # Build input_ids and labels incrementally
            input_ids_parts = [bos_tokens, compressed_placeholders]
            label_parts = [
                torch.full((batch_size, 1), -100, dtype=torch.long),  # BOS masked
                torch.full((batch_size, num_compressed), -100, dtype=torch.long)  # Compressed masked
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
            # [BOS] + [COMPRESSED_PLACEHOLDERS] + [HYBRID (optional)] + [ORIGINAL_CONTEXT]
            original_context = target_tokens
            bos_tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long)
            compressed_placeholders = torch.full(
                (batch_size, num_compressed), self.compressed_token_id, dtype=torch.long
            )

            # Build input_ids and labels incrementally
            input_ids_parts = [bos_tokens, compressed_placeholders]
            label_parts = [
                torch.full((batch_size, 1), -100, dtype=torch.long),  # BOS masked
                torch.full((batch_size, num_compressed), -100, dtype=torch.long)  # Compressed masked
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

        # 6. Create mask for compressed positions (True = inject compressed embedding)
        mask_parts = [
            torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device),  # BOS
            torch.ones(batch_size, num_compressed, dtype=torch.bool, device=self.device)  # Compressed positions
        ]

        # Add hybrid text mask if present
        if hybrid_text is not None and hybrid_text.numel() > 0:
            hybrid_len = hybrid_text.shape[1]
            mask_parts.append(torch.zeros(batch_size, hybrid_len, dtype=torch.bool, device=self.device))  # Hybrid (no injection)

        # Add target tokens mask
        mask_parts.append(torch.zeros(batch_size, target_tokens.shape[1], dtype=torch.bool, device=self.device))  # Target tokens

        compressed_mask = torch.cat(mask_parts, dim=1)

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
        hybrid_text_tokens: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate text from conv1d-compressed context

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

        # Move to device
        context_tokens = context_tokens.to(self.device)

        # Get embeddings and compress
        context_embeds = self.model.model.get_input_embeddings()(context_tokens)
        compressed_embeds = self._conv1d_residual_compress(context_embeds)

        # Add separator
        separator = self.separator_embed.unsqueeze(0).unsqueeze(0)
        compressed_with_sep = torch.cat([compressed_embeds, separator], dim=1)

        num_compressed = compressed_with_sep.shape[1]

        # Build input sequence: [BOS] + [COMPRESSED] + [HYBRID (optional)] + [PROMPT]
        bos_id = self.tokenizer.bos_token_id
        bos_tokens = torch.tensor([[bos_id]], dtype=torch.long)
        compressed_placeholders = torch.full(
            (1, num_compressed), self.compressed_token_id, dtype=torch.long
        )

        input_ids_parts = [bos_tokens, compressed_placeholders]

        # Add hybrid text if provided (to match training distribution)
        if hybrid_text_tokens is not None and len(hybrid_text_tokens) > 0:
            # Ensure it's a tensor on CPU
            if not isinstance(hybrid_text_tokens, torch.Tensor):
                hybrid_text_tokens = torch.tensor(hybrid_text_tokens, dtype=torch.long)
            else:
                hybrid_text_tokens = hybrid_text_tokens.cpu()  # Ensure on CPU for concatenation
            hybrid_text_tokens = hybrid_text_tokens.unsqueeze(0)  # Add batch dim
            input_ids_parts.append(hybrid_text_tokens)

        # Add prompt if specified
        if prompt_text:
            prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long)
            input_ids_parts.append(prompt_tensor)

        input_ids = torch.cat(input_ids_parts, dim=1).to(self.device)

        # Get embeddings and inject compressed
        inputs_embeds = self.model.model.get_input_embeddings()(input_ids)

        # Create mask: inject compressed embeddings only at compressed positions
        # Calculate hybrid + prompt length
        rest_len = input_ids.shape[1] - 1 - num_compressed  # Total minus BOS minus compressed

        compressed_mask = torch.cat([
            torch.zeros(1, 1, dtype=torch.bool),  # BOS
            torch.ones(1, num_compressed, dtype=torch.bool),  # Compressed
            torch.zeros(1, rest_len, dtype=torch.bool)  # Hybrid + prompt
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


