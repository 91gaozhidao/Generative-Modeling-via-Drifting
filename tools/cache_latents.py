"""
Latent Pre-caching Script for Drifting Field Generative Models

This script pre-processes image datasets by encoding them to latent space using
a VAE encoder (SD-VAE-ft-mse), as described in the paper's Appendix A.8.

Key benefits:
- Eliminates VAE encoding during training (90%+ I/O reduction)
- Allows training to work entirely in latent space
- Significantly reduces GPU memory usage

Usage:
    # For ImageNet-style folder structure
    python tools/cache_latents.py --data_dir /path/to/imagenet/train --output_dir ./cached_latents

    # For dummy dataset (for testing)
    python tools/cache_latents.py --dummy --num_dummy_samples 1000 --output_dir ./cached_latents
"""

import argparse
import os
from pathlib import Path
from typing import Optional, Tuple, List
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

try:
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# VAE latent scale factor (standard for Stable Diffusion)
VAE_SCALE_FACTOR = 0.18215


def load_vae(device: str = "cpu") -> Optional[nn.Module]:
    """
    Load the VAE encoder from stabilityai/sd-vae-ft-mse.
    
    Args:
        device: Device to load the VAE on
        
    Returns:
        VAE model or None if diffusers not available
    """
    try:
        from diffusers import AutoencoderKL
        
        print("Loading VAE from stabilityai/sd-vae-ft-mse...")
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float32,
        )
        vae = vae.to(device)
        vae.eval()
        
        # Freeze VAE parameters
        for param in vae.parameters():
            param.requires_grad = False
        
        print("VAE loaded successfully!")
        return vae
        
    except ImportError:
        print("Error: diffusers not installed.")
        print("Install with: pip install diffusers transformers")
        return None
    except Exception as e:
        print(f"Error loading VAE: {e}")
        return None


def get_image_transform(image_size: int = 256) -> "transforms.Compose":
    """
    Get image preprocessing transform.
    
    Args:
        image_size: Target image size (default: 256)
        
    Returns:
        Torchvision transform
    """
    if not HAS_TORCHVISION:
        raise ImportError("torchvision is required. Install with: pip install torchvision")
    
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1] normalization
    ])


class DummyImageDataset(Dataset):
    """
    Dummy dataset for testing the pipeline without real images.
    
    Generates random RGB images with random class labels.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_classes: int = 1000,
        image_size: int = 256,
        seed: int = 42,
    ):
        """
        Args:
            num_samples: Number of dummy samples to generate
            num_classes: Number of classes for labels
            image_size: Size of generated images
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        
        # Generate fixed random labels
        rng = np.random.RandomState(seed)
        self.labels = rng.randint(0, num_classes, size=num_samples)
        
        # Store the seed for reproducible image generation
        self.seed = seed
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Generate a random RGB image tensor.
        
        Returns:
            Tuple of (image_tensor, label) where image_tensor is (3, H, W) in [-1, 1]
        """
        # Use deterministic seed based on index for reproducibility
        rng = np.random.RandomState(self.seed + idx)
        
        # Generate random RGB values in [-1, 1]
        image = rng.randn(3, self.image_size, self.image_size).astype(np.float32)
        image = np.clip(image / 3, -1, 1)  # Scale and clip to [-1, 1]
        
        return torch.from_numpy(image), int(self.labels[idx])


class DummyLatentDataset(Dataset):
    """
    Dummy dataset that directly generates random latent tensors.
    
    This bypasses the need for VAE encoding and is useful for testing
    the training pipeline without requiring the diffusers library.
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_classes: int = 1000,
        latent_shape: Tuple[int, ...] = (4, 32, 32),
        seed: int = 42,
    ):
        """
        Args:
            num_samples: Number of dummy samples to generate
            num_classes: Number of classes for labels
            latent_shape: Shape of latent tensors (C, H, W)
            seed: Random seed for reproducibility
        """
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.latent_shape = latent_shape
        self.seed = seed
        
        # Generate fixed random labels
        rng = np.random.RandomState(seed)
        self.labels = rng.randint(0, num_classes, size=num_samples)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Generate a random latent tensor.
        
        Returns:
            Tuple of (latent_tensor, label)
        """
        # Use deterministic seed for reproducibility
        rng = np.random.RandomState(self.seed + idx)
        latent = rng.randn(*self.latent_shape).astype(np.float32)
        
        return torch.from_numpy(latent), int(self.labels[idx])


def cache_dummy_latents(
    num_samples: int,
    num_classes: int,
    output_dir: str,
    latent_shape: Tuple[int, ...] = (4, 32, 32),
    seed: int = 42,
    save_individually: bool = False,
) -> dict:
    """
    Generate and cache dummy random latent tensors directly.
    
    This function bypasses VAE encoding and generates random latent
    tensors for testing the training pipeline.
    
    Args:
        num_samples: Number of samples to generate
        num_classes: Number of classes
        output_dir: Directory to save cached latents
        latent_shape: Shape of each latent tensor (C, H, W)
        seed: Random seed for reproducibility
        save_individually: Whether to save each sample as a separate file
        
    Returns:
        Dictionary with metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_samples} dummy latents to {output_dir}...")
    
    rng = np.random.RandomState(seed)
    labels = rng.randint(0, num_classes, size=num_samples)
    
    if save_individually:
        for i in tqdm(range(num_samples), desc="Generating"):
            sample_rng = np.random.RandomState(seed + i)
            latent = torch.from_numpy(
                sample_rng.randn(*latent_shape).astype(np.float32)
            )
            sample_path = output_dir / f"sample_{i:08d}.pt"
            torch.save({
                'latent': latent,
                'label': int(labels[i]),
            }, sample_path)
    else:
        # Generate all latents at once
        all_latents = []
        for i in tqdm(range(num_samples), desc="Generating"):
            sample_rng = np.random.RandomState(seed + i)
            latent = sample_rng.randn(*latent_shape).astype(np.float32)
            all_latents.append(latent)
        
        all_latents = torch.from_numpy(np.stack(all_latents))
        all_labels = torch.tensor(labels, dtype=torch.long)
        
        # Save latents and labels
        torch.save(all_latents, output_dir / "latents.pt")
        torch.save(all_labels, output_dir / "labels.pt")
        
        print(f"Saved {num_samples} samples:")
        print(f"  - Latents: {output_dir / 'latents.pt'} (shape: {all_latents.shape})")
        print(f"  - Labels: {output_dir / 'labels.pt'} (shape: {all_labels.shape})")
    
    # Save metadata
    metadata = {
        'num_samples': num_samples,
        'num_classes': num_classes,
        'latent_shape': list(latent_shape),
        'scale_factor': VAE_SCALE_FACTOR,
        'use_mean': True,
        'save_individually': save_individually,
        'is_dummy': True,
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to {output_dir / 'metadata.json'}")
    
    return metadata


@torch.no_grad()
def encode_images(
    vae: nn.Module,
    images: torch.Tensor,
    use_mean: bool = True,
) -> torch.Tensor:
    """
    Encode images to latent space using VAE.
    
    Args:
        vae: VAE encoder model
        images: Input images of shape (B, 3, H, W) in range [-1, 1]
        use_mean: If True, use the mean of the latent distribution.
                  If False, sample from the distribution.
        
    Returns:
        Latent tensors of shape (B, 4, H//8, W//8), scaled by VAE_SCALE_FACTOR
    """
    # Encode to latent distribution
    latent_dist = vae.encode(images).latent_dist
    
    if use_mean:
        latents = latent_dist.mean
    else:
        latents = latent_dist.sample()
    
    # Apply scale factor to normalize variance
    latents = latents * VAE_SCALE_FACTOR
    
    return latents


def cache_dataset(
    dataloader: DataLoader,
    vae: nn.Module,
    output_dir: str,
    device: str = "cpu",
    use_mean: bool = True,
    save_individually: bool = False,
) -> dict:
    """
    Cache an entire dataset as latent tensors.
    
    Args:
        dataloader: DataLoader providing (images, labels)
        vae: VAE encoder model
        output_dir: Directory to save cached latents
        device: Device to run encoding on
        use_mean: Whether to use mean or sample from latent distribution
        save_individually: If True, save each sample as a separate file.
                          If False, save batched tensors.
        
    Returns:
        Dictionary with caching statistics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_latents = []
    all_labels = []
    
    num_samples = 0
    
    print(f"Caching latents to {output_dir}...")
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Encoding")):
        images = images.to(device)
        
        # Encode to latents
        latents = encode_images(vae, images, use_mean=use_mean)
        
        if save_individually:
            # Save each sample as individual file
            for i, (latent, label) in enumerate(zip(latents, labels)):
                sample_idx = batch_idx * dataloader.batch_size + i
                sample_path = output_dir / f"sample_{sample_idx:08d}.pt"
                torch.save({
                    'latent': latent.cpu(),
                    'label': label.item() if isinstance(label, torch.Tensor) else label,
                }, sample_path)
                num_samples += 1
        else:
            # Accumulate for batched saving
            all_latents.append(latents.cpu())
            all_labels.extend(
                [l.item() if isinstance(l, torch.Tensor) else l for l in labels]
            )
            num_samples += len(labels)
    
    if not save_individually:
        # Concatenate and save as single files
        all_latents = torch.cat(all_latents, dim=0)
        all_labels = torch.tensor(all_labels, dtype=torch.long)
        
        # Save latents
        torch.save(all_latents, output_dir / "latents.pt")
        
        # Save labels
        torch.save(all_labels, output_dir / "labels.pt")
        
        print(f"Saved {num_samples} samples:")
        print(f"  - Latents: {output_dir / 'latents.pt'} (shape: {all_latents.shape})")
        print(f"  - Labels: {output_dir / 'labels.pt'} (shape: {all_labels.shape})")
    else:
        print(f"Saved {num_samples} individual sample files to {output_dir}")
    
    # Save metadata
    metadata = {
        'num_samples': num_samples,
        'latent_shape': list(latents.shape[1:]),
        'scale_factor': VAE_SCALE_FACTOR,
        'use_mean': use_mean,
        'save_individually': save_individually,
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to {output_dir / 'metadata.json'}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Cache image dataset as VAE latents for efficient training"
    )
    
    # Data source
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to image dataset (ImageNet-style folder structure)",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Generate dummy dataset for testing",
    )
    parser.add_argument(
        "--num_dummy_samples",
        type=int,
        default=1000,
        help="Number of dummy samples to generate (default: 1000)",
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save cached latents",
    )
    
    # Processing options
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Input image size (default: 256)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run encoding on",
    )
    parser.add_argument(
        "--use_sample",
        action="store_true",
        help="Sample from latent distribution instead of using mean",
    )
    parser.add_argument(
        "--save_individually",
        action="store_true",
        help="Save each sample as a separate file",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        help="Number of classes (default: 1000 for ImageNet)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dummy dataset (default: 42)",
    )
    parser.add_argument(
        "--latent_size",
        type=int,
        default=32,
        help="Latent spatial size (default: 32 for 256px images with 8x downsampling)",
    )
    parser.add_argument(
        "--latent_channels",
        type=int,
        default=4,
        help="Number of latent channels (default: 4 for SD-VAE)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.dummy and args.data_dir is None:
        parser.error("Either --data_dir or --dummy must be specified")
    
    # For dummy data, we can skip VAE and generate random latents directly
    if args.dummy:
        print(f"Creating dummy latent dataset with {args.num_dummy_samples} samples...")
        latent_shape = (args.latent_channels, args.latent_size, args.latent_size)
        metadata = cache_dummy_latents(
            num_samples=args.num_dummy_samples,
            num_classes=args.num_classes,
            output_dir=args.output_dir,
            latent_shape=latent_shape,
            seed=args.seed,
            save_individually=args.save_individually,
        )
        print("\nCaching complete!")
        print(f"Total samples: {metadata['num_samples']}")
        print(f"Latent shape per sample: {metadata['latent_shape']}")
        return
    
    # For real data, we need VAE
    vae = load_vae(args.device)
    if vae is None:
        print("Failed to load VAE. Exiting.")
        return
    
    # Create dataset for real images
    if not HAS_TORCHVISION:
        print("Error: torchvision is required for loading image datasets.")
        print("Install with: pip install torchvision")
        return
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        return
    
    print(f"Loading dataset from {args.data_dir}...")
    transform = get_image_transform(args.image_size)
    dataset = ImageFolder(args.data_dir, transform=transform)
    print(f"Found {len(dataset)} images in {len(dataset.classes)} classes")
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Preserve order for reproducibility
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )
    
    # Cache latents
    metadata = cache_dataset(
        dataloader=dataloader,
        vae=vae,
        output_dir=args.output_dir,
        device=args.device,
        use_mean=not args.use_sample,
        save_individually=args.save_individually,
    )
    
    print("\nCaching complete!")
    print(f"Total samples: {metadata['num_samples']}")
    print(f"Latent shape per sample: {metadata['latent_shape']}")
    print(f"Scale factor: {metadata['scale_factor']}")


if __name__ == "__main__":
    main()
