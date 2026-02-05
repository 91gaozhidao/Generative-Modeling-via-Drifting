"""
Visualization Utilities for Drifting Field Generative Models

This module provides utilities for:
- Converting latent tensors to images using VAE decoder
- Creating image grids for visualization
- Saving generated samples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
import numpy as np

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def latent_to_image(
    latents: torch.Tensor,
    vae: Optional[nn.Module] = None,
    scale_factor: float = 0.18215,
) -> torch.Tensor:
    """
    Convert latent tensors to RGB images.
    
    Args:
        latents: Latent tensors of shape (B, C, H, W)
        vae: Optional VAE decoder. If None, returns normalized latents
        scale_factor: VAE latent scale factor (default for SD)
        
    Returns:
        Image tensors of shape (B, 3, H', W') in range [0, 1]
    """
    if vae is not None:
        # Decode with VAE
        with torch.no_grad():
            # Scale latents
            latents = latents / scale_factor
            
            # Decode
            images = vae.decode(latents).sample
            
            # Normalize to [0, 1]
            images = (images / 2 + 0.5).clamp(0, 1)
    else:
        # Without VAE, just normalize the latents for visualization
        # Take first 3 channels if more exist
        if latents.shape[1] > 3:
            latents = latents[:, :3]
        elif latents.shape[1] < 3:
            # Repeat to get 3 channels
            latents = latents.repeat(1, 3, 1, 1)[:, :3]
        
        # Normalize to [0, 1]
        images = (latents - latents.min()) / (latents.max() - latents.min() + 1e-8)
    
    return images


def tensor_to_pil(
    tensor: torch.Tensor,
    normalize: bool = False,
) -> Union["Image.Image", List["Image.Image"]]:
    """
    Convert tensor to PIL Image(s).
    
    Args:
        tensor: Image tensor of shape (B, C, H, W) or (C, H, W)
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        PIL Image or list of PIL Images
    """
    if not HAS_PIL:
        raise ImportError("PIL is required for tensor_to_pil. Install with: pip install Pillow")
    
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    if normalize:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min() + 1e-8)
    
    # Clamp and convert to uint8
    tensor = tensor.clamp(0, 1)
    tensor = (tensor * 255).byte()
    
    # Convert to numpy and PIL
    images = []
    for i in range(tensor.shape[0]):
        img_np = tensor[i].permute(1, 2, 0).cpu().numpy()
        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)
        images.append(Image.fromarray(img_np))
    
    return images[0] if len(images) == 1 else images


def make_grid(
    images: torch.Tensor,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Make a grid of images.
    
    Args:
        images: Tensor of shape (B, C, H, W)
        nrow: Number of images per row
        padding: Padding between images
        normalize: Whether to normalize each image
        pad_value: Value for padding
        
    Returns:
        Grid tensor of shape (C, H', W')
    """
    if normalize:
        # Normalize each image independently
        images = images.clone()
        for i in range(images.shape[0]):
            img = images[i]
            images[i] = (img - img.min()) / (img.max() - img.min() + 1e-8)
    
    batch_size, channels, height, width = images.shape
    
    # Calculate grid dimensions
    ncol = (batch_size + nrow - 1) // nrow
    
    # Create output tensor
    grid_height = height * ncol + padding * (ncol + 1)
    grid_width = width * nrow + padding * (nrow + 1)
    grid = torch.full((channels, grid_height, grid_width), pad_value, device=images.device)
    
    # Fill in the images
    for idx in range(batch_size):
        row = idx // nrow
        col = idx % nrow
        
        y_start = padding + row * (height + padding)
        x_start = padding + col * (width + padding)
        
        grid[:, y_start:y_start + height, x_start:x_start + width] = images[idx]
    
    return grid


def save_image(
    tensor: torch.Tensor,
    path: str,
    nrow: int = 8,
    normalize: bool = True,
) -> None:
    """
    Save a tensor as an image.
    
    Args:
        tensor: Image tensor of shape (B, C, H, W) or (C, H, W)
        path: Path to save the image
        nrow: Number of images per row (for batched tensors)
        normalize: Whether to normalize the image
    """
    if not HAS_PIL:
        raise ImportError("PIL is required for save_image. Install with: pip install Pillow")
    
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    # Make grid
    grid = make_grid(tensor, nrow=nrow, normalize=normalize)
    
    # Convert to PIL and save
    pil_image = tensor_to_pil(grid.unsqueeze(0))
    # tensor_to_pil returns a single Image when batch size is 1
    if isinstance(pil_image, list):
        pil_image = pil_image[0]
    pil_image.save(path)


def generate_and_visualize(
    model: nn.Module,
    vae: Optional[nn.Module] = None,
    labels: Optional[torch.Tensor] = None,
    num_samples: int = 16,
    cfg_scale: float = 1.5,
    device: str = "cpu",
    nrow: int = 4,
    save_path: Optional[str] = None,
) -> torch.Tensor:
    """
    Generate samples and create a visualization grid.
    
    Args:
        model: DriftingDiT model
        vae: Optional VAE decoder
        labels: Class labels (if None, uses random labels)
        num_samples: Number of samples to generate
        cfg_scale: CFG scale for generation
        device: Device to run on
        nrow: Number of images per row
        save_path: Optional path to save the grid
        
    Returns:
        Image grid tensor
    """
    model.eval()
    
    # Generate labels if not provided
    if labels is None:
        labels = torch.randint(0, model.num_classes, (num_samples,), device=device)
    else:
        labels = labels.to(device)
        num_samples = labels.shape[0]
    
    # Generate samples
    with torch.no_grad():
        latents = model.generate(
            batch_size=num_samples,
            y=labels,
            cfg_scale=cfg_scale,
            device=device,
        )
        
        # Convert to images
        images = latent_to_image(latents, vae=vae)
    
    # Make grid
    grid = make_grid(images, nrow=nrow, normalize=True)
    
    # Save if path provided
    if save_path is not None:
        save_image(grid.unsqueeze(0), save_path, normalize=False)
    
    return grid


# ImageNet class name mapping (subset for common classes)
IMAGENET_CLASSES = {
    0: "tench",
    1: "goldfish",
    2: "great white shark",
    3: "tiger shark",
    4: "hammerhead shark",
    5: "electric ray",
    6: "stingray",
    7: "cock",
    8: "hen",
    9: "ostrich",
    # ... many more classes
    263: "Pembroke Welsh Corgi",
    264: "Cardigan Welsh Corgi",
    # ... many more classes
    409: "analog clock",
    # ... many more classes
    985: "daisy",
    986: "yellow lady's slipper",
    987: "corn",
    988: "acorn",
    989: "hip (rose hip)",
    990: "buckeye",
    991: "coral fungus",
    992: "agaric",
    993: "gyromitra",
    994: "stinkhorn",
    995: "earthstar",
    996: "hen-of-the-woods",
    997: "bolete",
    998: "ear (corn)",
    999: "toilet tissue",
}


def get_class_name(class_id: int) -> str:
    """Get human-readable class name for ImageNet class ID."""
    return IMAGENET_CLASSES.get(class_id, f"class_{class_id}")
