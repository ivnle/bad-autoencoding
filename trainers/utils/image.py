"""Image processing and transformation utilities."""

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class BasicImageTransform:
    """Image transformation matching DeepSeek-OCR preprocessing"""

    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.normalize = normalize

    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Transform PIL Image to tensor using optimized torchvision operations.

        Optimization: Uses torchvision.transforms.functional which has fewer
        memory copies and better vectorization than PIL → numpy → torch pipeline.

        Args:
            image: PIL Image in RGB format

        Returns:
            Tensor of shape (3, H, W) in range [-1, 1] if normalized,
            otherwise [0, 1]. Returns float32.
        """
        # Convert PIL to tensor [C, H, W] in range [0, 255] (uint8)
        # This is more efficient than PIL → numpy → torch (fewer copies)
        img_tensor = transforms.functional.pil_to_tensor(image)

        # Convert to float32 and scale to [0, 1] in one operation
        img_tensor = img_tensor.float() / 255.0

        if self.normalize:
            # Apply normalization: (x - mean) / std
            # torchvision expects mean/std as lists, not numpy arrays
            mean = self.mean.tolist() if hasattr(self.mean, 'tolist') else self.mean
            std = self.std.tolist() if hasattr(self.std, 'tolist') else self.std
            img_tensor = transforms.functional.normalize(img_tensor, mean, std)

        return img_tensor


def create_dummy_image(size: int = 1024, mode: str = 'constant') -> torch.Tensor:
    """
    Create dummy image tensor for testing

    Args:
        size: Image size (will be size x size)
        mode: 'constant' (all 0.5), 'random', or 'gradient'

    Returns:
        Image tensor of shape (3, size, size) in bfloat16
    """
    if mode == 'constant':
        # Constant gray image
        img_array = np.ones((size, size, 3), dtype=np.float32) * 127.5
    elif mode == 'random':
        # Random noise
        img_array = np.random.randint(0, 256, (size, size, 3), dtype=np.uint8).astype(np.float32)
    elif mode == 'gradient':
        # Gradient pattern
        grad = np.linspace(0, 255, size)
        img_array = np.stack([
            np.tile(grad[:, None], (1, size)),  # Red: horizontal gradient
            np.tile(grad[None, :], (size, 1)),  # Green: vertical gradient
            np.ones((size, size)) * 127.5,       # Blue: constant
        ], axis=-1).astype(np.float32)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Convert to PIL then apply transform
    pil_image = Image.fromarray(img_array.astype(np.uint8), mode='RGB')
    transform = BasicImageTransform()
    img_tensor = transform(pil_image)

    return img_tensor.to(torch.bfloat16)
