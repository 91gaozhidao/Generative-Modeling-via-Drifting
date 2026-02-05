"""
Generate Images with Drifting Field Generative Models

This script implements the inference pipeline for the Drifting Field method,
featuring one-step (1-NFE) generation - the main advantage over diffusion models.

Usage:
    python generate.py --output sample_output.png
    python generate.py --classes 1 263 --cfg_scale 1.5 --output samples.png
"""

import argparse
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union

from drifting.models.drifting_dit import DriftingDiT, DriftingDiTSmall, DriftingDiTBase
from drifting.utils.visualization import (
    latent_to_image,
    make_grid,
    save_image,
    tensor_to_pil,
    get_class_name,
    HAS_PIL,
)


def load_vae(device: str = "cpu") -> Optional[nn.Module]:
    """
    Load the VAE decoder for converting latents to images.
    
    Tries to load from diffusers if available.
    
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
        print("Warning: diffusers not installed. Generating without VAE decoder.")
        print("Install with: pip install diffusers transformers")
        return None
    except Exception as e:
        print(f"Warning: Could not load VAE: {e}")
        print("Generating without VAE decoder.")
        return None


def generate_images(
    model: nn.Module,
    vae: Optional[nn.Module],
    labels: torch.Tensor,
    cfg_scale: float = 1.5,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate images using the Drifting Field model.
    
    This is the main inference function - just ONE forward pass!
    
    Args:
        model: DriftingDiT model
        vae: Optional VAE decoder
        labels: Class labels of shape (B,)
        cfg_scale: CFG scale for conditioning strength (typically 1.0-7.5)
        device: Device to run on
        
    Returns:
        Generated images of shape (B, 3, H, W) in range [0, 1]
    """
    model.eval()
    labels = labels.to(device)
    batch_size = labels.shape[0]
    
    with torch.no_grad():
        # Step 1: Sample noise
        z = torch.randn(
            batch_size,
            model.in_chans,
            model.img_size,
            model.img_size,
            device=device,
        )
        
        # Step 2: One-step generation (1-NFE inference!)
        latents = model(z, labels, cfg_scale=cfg_scale)
        
        # Step 3: Decode latents to images
        images = latent_to_image(latents, vae=vae)
    
    return images


def demo(
    model_path: Optional[str] = None,
    output_path: str = "sample_output.png",
    device: str = "cpu",
    load_vae_decoder: bool = True,
    model_size: str = "small",
) -> None:
    """
    Demo function to generate sample images.
    
    Generates a "Goldfish" (Class 1) and a "Pembroke Welsh Corgi" (Class 263)
    image to demonstrate the model's capabilities.
    
    Args:
        model_path: Path to pretrained weights (if None, uses random weights)
        output_path: Path to save the output image
        device: Device to run on
        load_vae_decoder: Whether to load VAE decoder
        model_size: Model size ("small", "base", "large")
    """
    print(f"Running demo on device: {device}")
    print(f"Model size: {model_size}")
    
    # Initialize model
    model_classes = {
        "small": DriftingDiTSmall,
        "base": DriftingDiTBase,
    }
    
    ModelClass = model_classes.get(model_size, DriftingDiTSmall)
    
    model = ModelClass(
        img_size=32,
        patch_size=2,
        in_chans=4,
        num_classes=1000,
        num_style_tokens=32,
    ).to(device)
    
    # Load pretrained weights if available
    if model_path is not None:
        print(f"Loading model from {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            elif "model_ema" in checkpoint:
                model.load_state_dict(checkpoint["model_ema"])
            else:
                model.load_state_dict(checkpoint)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load model weights: {e}")
            print("Using random initialization.")
    else:
        print("No model path provided. Using random initialization.")
        print("Note: Random weights will produce noise-like outputs.")
    
    # Load VAE if requested
    vae = None
    if load_vae_decoder:
        vae = load_vae(device)
    
    # Define classes to generate
    # Class 1: Goldfish
    # Class 263: Pembroke Welsh Corgi
    class_ids = [1, 263]
    labels = torch.tensor(class_ids, device=device)
    
    print(f"\nGenerating images for classes:")
    for cid in class_ids:
        print(f"  - Class {cid}: {get_class_name(cid)}")
    
    # Generate images
    print("\nGenerating images (1-NFE inference)...")
    images = generate_images(
        model=model,
        vae=vae,
        labels=labels,
        cfg_scale=1.5,
        device=device,
    )
    
    # Create grid and save
    grid = make_grid(images, nrow=2, normalize=True, padding=4)
    
    if HAS_PIL:
        save_image(grid.unsqueeze(0), output_path, normalize=False)
        print(f"\nSaved output to: {output_path}")
    else:
        print("\nPIL not available. Cannot save image.")
        print("Install with: pip install Pillow")
    
    print("Demo complete!")


def main():
    """Main entry point for the generate script."""
    parser = argparse.ArgumentParser(
        description="Generate images with Drifting Field models"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sample_output.png",
        help="Output path for generated images",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[1, 263],
        help="Class IDs to generate (default: 1=goldfish, 263=Pembroke)",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=1.5,
        help="CFG scale for conditioning strength (default: 1.5)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples per class (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--no_vae",
        action="store_true",
        help="Don't load VAE decoder",
    )
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["small", "base"],
        default="small",
        help="Model size to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    # Set seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Run demo if no specific classes provided beyond defaults
    if args.classes == [1, 263] and args.num_samples is None:
        demo(
            model_path=args.model_path,
            output_path=args.output,
            device=args.device,
            load_vae_decoder=not args.no_vae,
            model_size=args.model_size,
        )
    else:
        # Custom generation
        print(f"Running custom generation on device: {args.device}")
        
        # Initialize model
        model_classes = {
            "small": DriftingDiTSmall,
            "base": DriftingDiTBase,
        }
        ModelClass = model_classes.get(args.model_size, DriftingDiTSmall)
        
        model = ModelClass(
            img_size=32,
            patch_size=2,
            in_chans=4,
            num_classes=1000,
            num_style_tokens=32,
        ).to(args.device)
        
        # Load weights
        if args.model_path:
            checkpoint = torch.load(args.model_path, map_location=args.device)
            if "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            else:
                model.load_state_dict(checkpoint)
        
        # Load VAE
        vae = load_vae(args.device) if not args.no_vae else None
        
        # Create labels
        num_samples = args.num_samples or 1
        all_labels = []
        for cid in args.classes:
            all_labels.extend([cid] * num_samples)
        labels = torch.tensor(all_labels, device=args.device)
        
        # Generate
        images = generate_images(
            model=model,
            vae=vae,
            labels=labels,
            cfg_scale=args.cfg_scale,
            device=args.device,
        )
        
        # Save
        nrow = min(8, len(images))
        save_image(images, args.output, nrow=nrow, normalize=True)
        print(f"Saved {len(images)} images to: {args.output}")


if __name__ == "__main__":
    main()
