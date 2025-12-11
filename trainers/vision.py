"""VisionCompressionTrainer trainer implementation."""

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


class VisionCompressionTrainer:
    """
    Regime 1: Vision Compression
    - Frozen vision encoder (SAM + ViT + Projector)
    - Trainable language decoder
    - Context: vision-compressed (mode-dependent tokens)
    - Loss: computed only on continuation tokens
    """

    def __init__(self, model, tokenizer, vision_mode: str = 'small', device: str = 'cuda', hybrid_text_tokens: int = 0, vision_prompt: Optional[str] = None, train_encoder: bool = False):
        self.model = model
        self.tokenizer = tokenizer
        self.vision_mode = vision_mode
        self.mode_config = VISION_MODES[vision_mode]
        self.device = torch.device(device)
        self.hybrid_text_tokens = hybrid_text_tokens
        self.vision_prompt = vision_prompt
        self.train_encoder = train_encoder

        # GPU normalization constants (32.9x faster than CPU normalization)
        # Mean/std for DeepSeek-OCR preprocessing
        self._norm_mean = torch.tensor([0.5, 0.5, 0.5], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)
        self._norm_std = torch.tensor([0.5, 0.5, 0.5], device=self.device, dtype=torch.float32).view(1, 3, 1, 1)

        # Tokenize vision prompt once if provided
        if vision_prompt:
            self.vision_prompt_tokens = torch.tensor(
                tokenizer.encode(vision_prompt, add_special_tokens=False),
                dtype=torch.long
            )
            self.vision_prompt_len = len(self.vision_prompt_tokens)
        else:
            self.vision_prompt_tokens = None
            self.vision_prompt_len = 0

        # Conditionally freeze/unfreeze vision encoder
        if train_encoder:
            unfreeze_vision_encoder(model)
        else:
            freeze_vision_encoder(model)

        # Unfreeze decoder
        unfreeze_decoder(model)

        # Print parameter counts
        params = count_parameters(model)
        print(f"\n[Regime 1: Vision Compression - {vision_mode.upper()} mode]")
        print(f"Resolution: {self.mode_config['image_size']}x{self.mode_config['image_size']}")
        print(f"Vision tokens: {self.mode_config['tokens']}")
        if vision_prompt:
            print(f"Vision prompt: \"{vision_prompt}\" ({self.vision_prompt_len} tokens)")
        if hybrid_text_tokens > 0:
            total_context_tokens = self.mode_config['tokens'] + self.vision_prompt_len + hybrid_text_tokens
            print(f"Hybrid mode: {self.mode_config['tokens']} vision + {self.vision_prompt_len} prompt + {hybrid_text_tokens} text = {total_context_tokens} total context tokens")
        print(f"Total parameters: {params['total']:,}")
        print(f"Trainable: {params['trainable']:,} ({params['trainable_pct']:.1f}%)")
        print(f"Frozen: {params['frozen']:,}")

    def _normalize_images_gpu(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize uint8 image tensor on GPU (32.9x faster than CPU normalization).

        Args:
            image_tensor: uint8 tensor [B, C, H, W] on GPU

        Returns:
            Normalized bfloat16 tensor [B, C, H, W]
        """
        # uint8 -> float32 -> normalize -> bfloat16
        image_tensor = image_tensor.float() / 255.0
        image_tensor = (image_tensor - self._norm_mean) / self._norm_std
        return image_tensor.to(torch.bfloat16)

    def prepare_batch(
        self,
        image_tensor: torch.Tensor,
        continuation_tokens: torch.Tensor,
        hybrid_text: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare training batch for Regime 1

        Args:
            image_tensor: Image tensor of shape (batch_size, 3, base_size, base_size) in uint8
            continuation_tokens: Token IDs for text continuation (shape: [batch_size, seq_len])
            hybrid_text: Optional text tokens to append after vision (shape: [batch_size, hybrid_len])

        Returns:
            Tuple of (input_ids, images, images_seq_mask, labels)
        """
        # Get mode-specific configuration
        vision_token_count = self.mode_config['tokens']
        image_size = self.mode_config['image_size']
        batch_size = image_tensor.shape[0]

        # Create vision token IDs (mode-dependent count) for each sample in batch
        vision_token_ids = torch.full((batch_size, vision_token_count), IMAGE_TOKEN_ID, dtype=torch.long)

        # Build sequence: [VISION] + [PROMPT (optional)] + [HYBRID (optional)] + [CONTINUATION]
        input_ids_parts = [vision_token_ids]
        mask_parts = [torch.ones(batch_size, vision_token_count, dtype=torch.bool)]
        label_parts = [torch.full((batch_size, vision_token_count), -100, dtype=torch.long)]

        # Add vision prompt if configured
        if self.vision_prompt_tokens is not None:
            # Expand prompt tokens to batch size
            prompt_batch = self.vision_prompt_tokens.unsqueeze(0).expand(batch_size, -1)
            input_ids_parts.append(prompt_batch)
            mask_parts.append(torch.zeros(batch_size, self.vision_prompt_len, dtype=torch.bool))
            label_parts.append(torch.full((batch_size, self.vision_prompt_len), -100, dtype=torch.long))

        # Add hybrid text if present
        if hybrid_text is not None and hybrid_text.numel() > 0:
            hybrid_len = hybrid_text.shape[1]
            input_ids_parts.append(hybrid_text)
            mask_parts.append(torch.zeros(batch_size, hybrid_len, dtype=torch.bool))
            label_parts.append(torch.full((batch_size, hybrid_len), -100, dtype=torch.long))

        # Add continuation (only part that contributes to loss)
        input_ids_parts.append(continuation_tokens)
        mask_parts.append(torch.zeros(batch_size, continuation_tokens.shape[1], dtype=torch.bool))
        label_parts.append(continuation_tokens)

        # Concatenate all parts
        input_ids = torch.cat(input_ids_parts, dim=1)
        images_seq_mask = torch.cat(mask_parts, dim=1)
        labels = torch.cat(label_parts, dim=1)

        # Add BOS token at the beginning (match official implementation)
        bos_id = self.tokenizer.bos_token_id
        bos_tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long)
        bos_mask = torch.zeros(batch_size, 1, dtype=torch.bool)
        bos_labels = torch.full((batch_size, 1), -100, dtype=torch.long)

        input_ids = torch.cat([bos_tokens, input_ids], dim=1)
        images_seq_mask = torch.cat([bos_mask, images_seq_mask], dim=1)
        labels = torch.cat([bos_labels, labels], dim=1)

        # Prepare images: [(crop, global)] per sample in batch
        # For native resolution modes (crop_mode=False), we don't use cropping
        # DeepSeek's model expects one tuple per sample, not a single tuple with batched tensor

        # Move entire batch to device and normalize on GPU (32.9x faster than CPU)
        # Images are uint8 from dataset, normalize to bfloat16 on GPU
        image_tensor = image_tensor.to(self.device, non_blocking=True)
        image_tensor = self._normalize_images_gpu(image_tensor)

        # Create empty_crop once and reuse (same for all samples)
        empty_crop = torch.empty(0, 3, image_size, image_size,
                                 dtype=torch.bfloat16, device=self.device)

        images = []
        for i in range(batch_size):
            single_image = image_tensor[i:i+1]  # Slice after normalize
            images.append((empty_crop, single_image))

        return input_ids, images, images_seq_mask, labels

    def prepare_batch_reconstruction(
        self,
        image_tensor: torch.Tensor,
        context_tokens: torch.Tensor,
        hybrid_text: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare reconstruction batch for vision regime (OCR task)

        Build: [BOS] + [VISION] + [PROMPT (optional)] + [HYBRID (optional)] + [CONTEXT]
        Train on: context only (reconstruct the text from the image)

        Args:
            image_tensor: Image tensor of shape (batch_size, 3, base_size, base_size) in uint8
            context_tokens: Context token IDs to reconstruct (shape: [batch_size, seq_len])
            hybrid_text: Optional hybrid text tokens (shape: [batch_size, hybrid_len])

        Returns:
            Tuple of (input_ids, images, images_seq_mask, labels)
        """
        # Get mode-specific configuration
        vision_token_count = self.mode_config['tokens']
        image_size = self.mode_config['image_size']
        batch_size = image_tensor.shape[0]

        # Create vision token IDs (mode-dependent count) for each sample in batch
        vision_token_ids = torch.full((batch_size, vision_token_count), IMAGE_TOKEN_ID, dtype=torch.long)

        # Build sequence: [VISION] + [PROMPT (optional)] + [HYBRID (optional)] + [CONTEXT]
        input_ids_parts = [vision_token_ids]
        mask_parts = [torch.ones(batch_size, vision_token_count, dtype=torch.bool)]
        label_parts = [torch.full((batch_size, vision_token_count), -100, dtype=torch.long)]

        # Add vision prompt if configured
        if self.vision_prompt_tokens is not None:
            # Expand prompt tokens to batch size
            prompt_batch = self.vision_prompt_tokens.unsqueeze(0).expand(batch_size, -1)
            input_ids_parts.append(prompt_batch)
            mask_parts.append(torch.zeros(batch_size, self.vision_prompt_len, dtype=torch.bool))
            label_parts.append(torch.full((batch_size, self.vision_prompt_len), -100, dtype=torch.long))

        # Add hybrid text if present
        if hybrid_text is not None and hybrid_text.numel() > 0:
            hybrid_len = hybrid_text.shape[1]
            input_ids_parts.append(hybrid_text)
            mask_parts.append(torch.zeros(batch_size, hybrid_len, dtype=torch.bool))
            label_parts.append(torch.full((batch_size, hybrid_len), -100, dtype=torch.long))

        # Add context (reconstruction target - only part that contributes to loss)
        input_ids_parts.append(context_tokens)
        mask_parts.append(torch.zeros(batch_size, context_tokens.shape[1], dtype=torch.bool))
        label_parts.append(context_tokens)  # Train on context tokens

        # Concatenate all parts
        input_ids = torch.cat(input_ids_parts, dim=1)
        images_seq_mask = torch.cat(mask_parts, dim=1)
        labels = torch.cat(label_parts, dim=1)

        # Add BOS token at the beginning (match official implementation)
        bos_id = self.tokenizer.bos_token_id
        bos_tokens = torch.full((batch_size, 1), bos_id, dtype=torch.long)
        bos_mask = torch.zeros(batch_size, 1, dtype=torch.bool)
        bos_labels = torch.full((batch_size, 1), -100, dtype=torch.long)

        input_ids = torch.cat([bos_tokens, input_ids], dim=1)
        images_seq_mask = torch.cat([bos_mask, images_seq_mask], dim=1)
        labels = torch.cat([bos_labels, labels], dim=1)

        # Prepare images: [(crop, global)] per sample in batch
        # For native resolution modes (crop_mode=False), we don't use cropping

        # Move entire batch to device and normalize on GPU (32.9x faster than CPU)
        # Images are uint8 from dataset, normalize to bfloat16 on GPU
        image_tensor = image_tensor.to(self.device, non_blocking=True)
        image_tensor = self._normalize_images_gpu(image_tensor)

        # Create empty_crop once and reuse (same for all samples)
        empty_crop = torch.empty(0, 3, image_size, image_size,
                                 dtype=torch.bfloat16, device=self.device)

        images = []
        for i in range(batch_size):
            single_image = image_tensor[i:i+1]  # Slice after normalize
            images.append((empty_crop, single_image))

        return input_ids, images, images_seq_mask, labels

    def forward(
        self,
        image_tensor: torch.Tensor,
        target_tokens: torch.Tensor,
        hybrid_text: Optional[torch.Tensor] = None,
        objective: str = 'lm'
    ) -> torch.Tensor:
        """
        Forward pass for Regime 1

        Args:
            image_tensor: Image tensor
            target_tokens: Target token IDs (continuation for lm, context for reconstruction)
            hybrid_text: Optional hybrid text tokens (for hybrid mode)
            objective: Training objective ('lm' or 'reconstruction')

        Returns:
            Loss tensor
        """
        # Prepare batch based on objective
        if objective == 'lm':
            # LM objective: predict continuation
            continuation_tokens = target_tokens
            input_ids, images, images_seq_mask, labels = self.prepare_batch(
                image_tensor, continuation_tokens, hybrid_text
            )
        else:  # reconstruction
            # Reconstruction objective: predict context (OCR task)
            context_tokens = target_tokens
            input_ids, images, images_seq_mask, labels = self.prepare_batch_reconstruction(
                image_tensor, context_tokens, hybrid_text
            )

        # Move to device (batch dimension already exists)
        input_ids = input_ids.to(self.device)
        images_seq_mask = images_seq_mask.to(self.device)
        labels = labels.to(self.device)

        # Create spatial crop list for each sample in batch
        batch_size = input_ids.shape[0]
        images_spatial_crop = [[1, 1]] * batch_size

        # Forward pass
        outputs = self.model.forward(
            input_ids=input_ids,
            images=images,
            images_seq_mask=images_seq_mask,
            images_spatial_crop=images_spatial_crop,
            labels=labels,
            return_dict=True
        )

        return outputs.loss, labels

    def generate_text(
        self,
        image_tensor: torch.Tensor,
        prompt_text: str = "",
        hybrid_text_tokens: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate text from image (optionally with text prompt and/or hybrid text)

        Args:
            image_tensor: Image tensor of shape (3, base_size, base_size) in uint8
            prompt_text: Optional additional text prompt (appended after everything)
            hybrid_text_tokens: Optional hybrid text tokens (shape: [seq_len]) to match training distribution
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
            **kwargs: Additional generation parameters

        Returns:
            Generated text (decoded)
        """
        # Prepare input (similar to prepare_batch but without continuation)
        vision_token_count = self.mode_config['tokens']
        image_size = self.mode_config['image_size']

        # Create vision token IDs: [VISION_TOKENS]
        vision_token_ids = torch.full((vision_token_count,), IMAGE_TOKEN_ID, dtype=torch.long)

        # Build input sequence: [VISION] + [VISION_PROMPT (optional)] + [HYBRID_TEXT (optional)] + [ADDITIONAL_PROMPT (optional)]
        input_parts = [vision_token_ids]
        mask_parts = [torch.ones(vision_token_count, dtype=torch.bool)]

        # Add vision prompt if configured
        if self.vision_prompt_tokens is not None:
            input_parts.append(self.vision_prompt_tokens)
            mask_parts.append(torch.zeros(self.vision_prompt_len, dtype=torch.bool))

        # Add hybrid text tokens if provided (to match training distribution)
        if hybrid_text_tokens is not None and len(hybrid_text_tokens) > 0:
            # Ensure it's on CPU and convert to tensor if needed
            if not isinstance(hybrid_text_tokens, torch.Tensor):
                hybrid_text_tokens = torch.tensor(hybrid_text_tokens, dtype=torch.long)
            input_parts.append(hybrid_text_tokens.cpu())
            mask_parts.append(torch.zeros(len(hybrid_text_tokens), dtype=torch.bool))

        # Add additional user-provided prompt if specified
        if prompt_text:
            prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long)
            input_parts.append(prompt_tokens)
            mask_parts.append(torch.zeros(len(prompt_tokens), dtype=torch.bool))

        # Concatenate all parts
        input_ids = torch.cat(input_parts)
        images_seq_mask = torch.cat(mask_parts)

        # Add BOS token at the beginning (match official implementation)
        bos_id = self.tokenizer.bos_token_id
        input_ids = torch.cat([torch.tensor([bos_id], dtype=torch.long), input_ids])
        images_seq_mask = torch.cat([torch.tensor([False], dtype=torch.bool), images_seq_mask])

        # Prepare images (mirror prepare_batch structure)
        # Images are uint8, normalize on GPU (32.9x faster than CPU)
        image_tensor = image_tensor.to(self.device)
        image_tensor = self._normalize_images_gpu(image_tensor.unsqueeze(0)).squeeze(0)
        # Match official implementation: use dimension 1 instead of 0 for crop tensor
        empty_crop = torch.zeros((1, 3, image_size, image_size),
                                 dtype=torch.bfloat16, device=self.device)
        # Single image: create tuple list with one element
        images = [(empty_crop, image_tensor.unsqueeze(0))]
        images_spatial_crop = [[1, 1]]

        # Move to device and add batch dimension
        input_ids = input_ids.unsqueeze(0).to(self.device)
        images_seq_mask = images_seq_mask.unsqueeze(0).to(self.device)

        # Get initial embeddings (standard vocabulary embeddings)
        # Vision tokens will be replaced by the model's forward() method
        initial_embeds = self.model.model.get_input_embeddings()(input_ids)

        # Generate using manual decode with KV-cache
        generated_ids = manual_generate_with_cache(
            model=self.model,
            tokenizer=self.tokenizer,
            initial_ids=input_ids,
            initial_embeds=initial_embeds,
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

