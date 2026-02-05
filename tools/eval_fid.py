"""
FID Evaluation Script for Drifting Field Generative Models

This script evaluates the model quality using Fréchet Inception Distance (FID),
the standard metric for generative model evaluation on ImageNet.

Key features:
- Batch generation of 50,000 samples for statistically significant FID
- Uses clean-fid library (standard ImageNet FID computation tool)
- Supports reference image folders or built-in dataset statistics

Usage:
    # Evaluate against ImageNet validation set
    python tools/eval_fid.py --checkpoint ./outputs/checkpoints/best_model.pt \\
        --imagenet_val /path/to/imagenet/val

    # Evaluate against built-in dataset statistics
    python tools/eval_fid.py --checkpoint ./outputs/checkpoints/best_model.pt \\
        --dataset_name imagenet

    # Quick test with fewer samples
    python tools/eval_fid.py --checkpoint ./outputs/checkpoints/best_model.pt \\
        --num_samples 1000 --imagenet_val /path/to/imagenet/val
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import json
import time
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from drifting.models.drifting_dit import (
    DriftingDiT,
    DriftingDiTSmall,
    DriftingDiTBase,
    DriftingDiTLarge,
)

# Model size configurations
MODEL_CONFIGS = {
    "small": DriftingDiTSmall,
    "base": DriftingDiTBase,
    "large": DriftingDiTLarge,
}


def load_vae(device: str = "cpu") -> Optional[nn.Module]:
    """
    Load the VAE decoder for converting latents to images.
    
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


def load_model(
    checkpoint_path: str,
    model_size: str = "small",
    device: str = "cpu",
) -> nn.Module:
    """
    Load trained DriftingDiT model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_size: Model size (small, base, large)
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    ModelClass = MODEL_CONFIGS.get(model_size, DriftingDiTSmall)
    
    model = ModelClass(
        img_size=32,  # Standard latent size for 256px images
        patch_size=2,
        in_chans=4,
        num_classes=1000,
        num_style_tokens=32,
    )
    
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Try different checkpoint formats
    if "model_ema" in checkpoint:
        # Prefer EMA weights for evaluation
        model.load_state_dict(checkpoint["model_ema"])
        print("Loaded EMA model weights")
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        print("Loaded model weights")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded state dict directly")
    
    model = model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def decode_latents(
    latents: torch.Tensor,
    vae: nn.Module,
) -> torch.Tensor:
    """
    Decode latent tensors to images using VAE.
    
    Args:
        latents: Latent tensors of shape (B, 4, H, W)
        vae: VAE decoder
        
    Returns:
        Images of shape (B, 3, H*8, W*8) in range [0, 255] as uint8
    """
    # VAE scale factor
    latents = latents / 0.18215
    
    # Decode
    images = vae.decode(latents).sample
    
    # Convert to [0, 255] uint8
    images = (images + 1) * 127.5
    images = images.clamp(0, 255).to(torch.uint8)
    
    return images


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    vae: nn.Module,
    num_samples: int,
    batch_size: int,
    output_dir: str,
    cfg_scale: float = 1.5,
    device: str = "cuda",
    seed: Optional[int] = None,
) -> int:
    """
    Generate samples and save to disk for FID computation.
    
    Args:
        model: DriftingDiT model
        vae: VAE decoder
        num_samples: Total number of samples to generate
        batch_size: Batch size for generation
        output_dir: Directory to save generated images
        cfg_scale: CFG scale for generation
        device: Device to generate on
        seed: Random seed for reproducibility
        
    Returns:
        Number of images generated
    """
    try:
        from PIL import Image
    except ImportError:
        print("Error: PIL not installed. Install with: pip install Pillow")
        return 0
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    model.eval()
    num_generated = 0
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    print(f"Generating {num_samples} samples...")
    
    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        # Determine batch size for this iteration
        current_batch_size = min(batch_size, num_samples - num_generated)
        
        # Sample random class labels (uniform over ImageNet classes)
        labels = torch.randint(0, 1000, (current_batch_size,), device=device)
        
        # Sample noise
        z = torch.randn(
            current_batch_size,
            model.in_chans,
            model.img_size,
            model.img_size,
            device=device,
        )
        
        # One-step generation
        latents = model(z, labels, cfg_scale=cfg_scale)
        
        # Decode to images
        images = decode_latents(latents, vae)
        
        # Save images
        for i, img_tensor in enumerate(images):
            img_idx = num_generated + i
            
            # Convert to PIL Image
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img = Image.fromarray(img_np)
            
            # Save with zero-padded filename
            img_path = output_dir / f"{img_idx:08d}.png"
            img.save(img_path)
        
        num_generated += current_batch_size
    
    print(f"Generated {num_generated} images to {output_dir}")
    return num_generated


def compute_fid(
    generated_dir: str,
    reference_dir: Optional[str] = None,
    dataset_name: Optional[str] = None,
    dataset_res: int = 256,
    dataset_split: str = "train",
    batch_size: int = 64,
    device: str = "cuda",
) -> float:
    """
    Compute FID between generated images and reference.
    
    Args:
        generated_dir: Directory containing generated images
        reference_dir: Directory containing reference images (e.g., ImageNet val)
        dataset_name: Name of built-in dataset for clean-fid (e.g., "imagenet")
        dataset_res: Resolution for dataset statistics (default: 256)
        dataset_split: Dataset split for statistics (default: "train")
        batch_size: Batch size for feature extraction
        device: Device for computation
        
    Returns:
        FID score
    """
    try:
        from cleanfid import fid
    except ImportError:
        print("Error: clean-fid not installed.")
        print("Install with: pip install clean-fid")
        return -1.0
    
    print("\nComputing FID score...")
    
    if reference_dir is not None:
        # Compute FID against reference directory (most common case)
        print(f"Reference directory: {reference_dir}")
        fid_score = fid.compute_fid(
            fdir1=generated_dir,
            fdir2=reference_dir,
            mode="clean",
            batch_size=batch_size,
            device=torch.device(device),
            verbose=True,
        )
    elif dataset_name is not None:
        # Use built-in dataset statistics from clean-fid
        # Common supported datasets: FFHQ, cifar10, stl10
        # Note: Some datasets may require pre-computed statistics
        print(f"Using built-in statistics for: {dataset_name}")
        try:
            fid_score = fid.compute_fid(
                fdir1=generated_dir,
                dataset_name=dataset_name,
                dataset_res=dataset_res,
                dataset_split=dataset_split,
                mode="clean",
                batch_size=batch_size,
                device=torch.device(device),
                verbose=True,
            )
        except Exception as e:
            print(f"Error computing FID with dataset '{dataset_name}': {e}")
            print("This dataset may not have pre-computed statistics available.")
            print("Consider using --imagenet_val with a reference directory instead.")
            return -1.0
    else:
        print("Error: Must specify either reference_dir or dataset_name")
        print("Examples:")
        print("  --imagenet_val /path/to/imagenet/val")
        print("  --dataset_name FFHQ")
        return -1.0
    
    return fid_score


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Drifting Field models using FID"
    )
    
    # Model arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["small", "base", "large"],
        default="small",
        help="Model size (default: small)",
    )
    
    # Generation arguments
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50000,
        help="Number of samples to generate (default: 50000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for generation (default: 64)",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.5,
        help="CFG scale for generation (default: 1.5)",
    )
    
    # Reference arguments (one of these is required for FID computation)
    parser.add_argument(
        "--imagenet_val",
        type=str,
        default=None,
        help="Path to ImageNet validation set directory",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "Built-in dataset name for clean-fid statistics. "
            "Common options: 'FFHQ', 'cifar10', 'stl10'. "
            "Note: ImageNet statistics may need to be computed first. "
            "See clean-fid documentation for available datasets."
        ),
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./fid_samples",
        help="Directory to save generated samples (default: ./fid_samples)",
    )
    parser.add_argument(
        "--keep_samples",
        action="store_true",
        help="Keep generated samples after FID computation",
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.imagenet_val is None and args.dataset_name is None:
        print("Warning: No reference specified. Will generate samples only.")
        print("To compute FID, specify one of: --imagenet_val or --dataset_name")
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    print("=" * 70)
    print("Drifting Field Model - FID Evaluation")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Model size: {args.model_size}")
    print(f"  Num samples: {args.num_samples}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  CFG scale: {args.cfg_scale}")
    print(f"  Device: {args.device}")
    print()
    
    # Load model
    model = load_model(
        args.checkpoint,
        model_size=args.model_size,
        device=args.device,
    )
    
    # Load VAE
    vae = load_vae(args.device)
    if vae is None:
        print("Error: Failed to load VAE. Exiting.")
        return
    
    # Generate samples
    start_time = time.time()
    num_generated = generate_samples(
        model=model,
        vae=vae,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        cfg_scale=args.cfg_scale,
        device=args.device,
        seed=args.seed,
    )
    generation_time = time.time() - start_time
    
    print(f"\nGeneration complete!")
    print(f"  Generated {num_generated} images in {generation_time:.1f}s")
    print(f"  Rate: {num_generated / generation_time:.1f} images/sec")
    
    # Compute FID if reference is specified
    if args.imagenet_val is not None or args.dataset_name is not None:
        fid_score = compute_fid(
            generated_dir=args.output_dir,
            reference_dir=args.imagenet_val,
            dataset_name=args.dataset_name,
            batch_size=args.batch_size,
            device=args.device,
        )
        
        print()
        print("=" * 70)
        print(f"FID Score: {fid_score:.2f}")
        print("=" * 70)
        
        # Save results
        results = {
            "fid_score": fid_score,
            "num_samples": num_generated,
            "cfg_scale": args.cfg_scale,
            "model_size": args.model_size,
            "checkpoint": args.checkpoint,
            "generation_time_seconds": generation_time,
        }
        
        results_path = Path(args.output_dir).parent / "fid_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {results_path}")
    
    # Clean up generated samples if not keeping
    if not args.keep_samples:
        print(f"\nCleaning up generated samples...")
        shutil.rmtree(args.output_dir)
        print(f"Removed: {args.output_dir}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
