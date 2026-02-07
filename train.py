"""
Main Training Script for Drifting Field Generative Models

This is the command center for training the Drifting Model on ImageNet
(or other datasets). It orchestrates all components:
- DriftingDiT model
- DriftingTrainer with sample queue
- LatentDataset for efficient data loading
- Checkpointing and visualization

Usage:
    # Train on pre-cached latents
    python train.py --data_dir ./cached_latents --output_dir ./outputs
    
    # Train on dummy data for testing
    python train.py --dummy --output_dir ./outputs
    
    # Resume from checkpoint
    python train.py --data_dir ./cached_latents --resume ./outputs/checkpoint.pt
"""

import argparse
import os
from pathlib import Path
from typing import Optional
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from drifting.models.drifting_dit import (
    DriftingDiT,
    DriftingDiTSmall,
    DriftingDiTBase,
    DriftingDiTLarge,
)
from drifting.models.drifting_loss import DriftingLoss
from drifting.training import DriftingTrainer, create_trainer
from drifting.data.dataset import (
    LatentDataset,
    DummyLatentDataset,
    create_dataloader,
    create_dummy_dataloader,
)
from drifting.utils.visualization import (
    save_image,
    make_grid,
    latent_to_image,
    get_class_name,
    HAS_PIL,
)


# Model size configurations
MODEL_CONFIGS = {
    "small": DriftingDiTSmall,
    "base": DriftingDiTBase,
    "large": DriftingDiTLarge,
}


def load_vae(device: str = "cpu") -> Optional[nn.Module]:
    """Load VAE for visualization during training."""
    try:
        from diffusers import AutoencoderKL
        
        print("Loading VAE for visualization...")
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float32,
        )
        vae = vae.to(device)
        vae.eval()
        
        for param in vae.parameters():
            param.requires_grad = False
        
        return vae
    except Exception as e:
        print(f"Warning: Could not load VAE: {e}")
        print("Visualization will use raw latents.")
        return None


def generate_samples(
    trainer: DriftingTrainer,
    vae: Optional[nn.Module],
    output_path: str,
    class_ids: list = None,
    cfg_scale: float = 1.5,
    num_samples_per_class: int = 2,
    device: str = "cpu",
) -> None:
    """
    Generate and save sample images for monitoring training progress.
    
    Args:
        trainer: DriftingTrainer instance
        vae: Optional VAE decoder
        output_path: Path to save the sample grid
        class_ids: List of class IDs to generate (default: [1, 263])
        cfg_scale: CFG scale for generation
        num_samples_per_class: Number of samples per class
        device: Device to generate on
    """
    if class_ids is None:
        # Default classes: Goldfish and Pembroke Welsh Corgi
        class_ids = [1, 263]
    
    # Create labels
    labels = []
    for cid in class_ids:
        labels.extend([cid] * num_samples_per_class)
    labels = torch.tensor(labels, device=device)
    
    print(f"Generating samples for classes: {[get_class_name(c) for c in class_ids]}")
    
    # Generate latents
    latents = trainer.generate(
        batch_size=len(labels),
        labels=labels,
        cfg_scale=cfg_scale,
        use_ema=True,
    )
    
    # Convert to images
    images = latent_to_image(latents, vae=vae)
    
    # Create grid and save
    nrow = min(num_samples_per_class * len(class_ids), 8)
    grid = make_grid(images, nrow=nrow, normalize=True, padding=4)
    
    if HAS_PIL:
        save_image(grid.unsqueeze(0), output_path, normalize=False)
        print(f"Saved samples to: {output_path}")
    else:
        print("Warning: PIL not available, cannot save images")


def train(
    args: argparse.Namespace,
) -> None:
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    # Set up device
    device = args.device
    print(f"Using device: {device}")
    
    # Enable TF32 for faster matrix multiplications on A100/Ampere GPUs
    torch.set_float32_matmul_precision('high')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Configuration saved to {output_dir / 'config.json'}")
    
    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Random seed: {args.seed}")
    
    # Create dataset and dataloader
    print("\nSetting up data...")
    if args.dummy:
        print(f"Using dummy dataset with {args.num_dummy_samples} samples")
        dataloader = create_dummy_dataloader(
            num_samples=args.num_dummy_samples,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            latent_shape=(args.in_channels, args.latent_size, args.latent_size),
            seed=args.seed or 42,
        )
        num_samples = args.num_dummy_samples
    else:
        print(f"Loading dataset from {args.data_dir}")
        dataloader = create_dataloader(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=device == "cuda",
        )
        num_samples = len(dataloader.dataset)
    
    print(f"Dataset size: {num_samples} samples")
    print(f"Batch size: {args.batch_size}")
    print(f"Batches per epoch: {len(dataloader)}")
    
    # Create model
    print(f"\nCreating {args.model_size} model...")
    ModelClass = MODEL_CONFIGS.get(args.model_size, DriftingDiTSmall)
    
    model = ModelClass(
        img_size=args.latent_size,
        patch_size=args.patch_size,
        in_chans=args.in_channels,
        num_classes=args.num_classes,
        num_style_tokens=args.num_style_tokens,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # Create trainer
    print("\nSetting up trainer...")
    trainer = create_trainer(
        model=model,
        feature_extractor=args.feature_extractor,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        total_steps=args.epochs * len(dataloader),
        device=device,
        queue_size=args.queue_size,
        num_classes=args.num_classes,
        cfg_scale_range=(args.cfg_min, args.cfg_max),
        uncond_prob=args.uncond_prob,
        latent_shape=(args.in_channels, args.latent_size, args.latent_size),
        ema_decay=args.ema_decay,
        mae_checkpoint=args.mae_checkpoint,
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        start_epoch = trainer.epoch
        print(f"Resuming from epoch {start_epoch}")
    
    # Load VAE for visualization
    vae = None
    generate_samples_flag = not args.no_generate_samples
    if generate_samples_flag:
        vae = load_vae(device)
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*60}\n")
    
    best_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train one epoch
        metrics = trainer.train_epoch(
            dataloader=dataloader,
            use_cfg_training=not args.no_cfg_training,
            log_interval=args.log_interval,
        )
        
        epoch_time = time.time() - epoch_start
        
        # Print metrics
        print(f"\nEpoch {epoch + 1}/{args.epochs} completed in {epoch_time:.1f}s")
        print(f"  Average Loss: {metrics['avg_loss']:.4f}")
        print(f"  Training Steps: {metrics['num_steps']}")
        print(f"  Global Step: {trainer.global_step}")
        
        # Generate samples at end of epoch
        if generate_samples_flag and (epoch + 1) % args.sample_interval == 0:
            sample_path = samples_dir / f"epoch_{epoch + 1:04d}.png"
            try:
                generate_samples(
                    trainer=trainer,
                    vae=vae,
                    output_path=str(sample_path),
                    class_ids=args.sample_classes,
                    cfg_scale=args.sample_cfg_scale,
                    num_samples_per_class=args.samples_per_class,
                    device=device,
                )
            except Exception as e:
                print(f"Warning: Failed to generate samples: {e}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = checkpoints_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt"
            trainer.save_checkpoint(str(checkpoint_path))
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if metrics['avg_loss'] < best_loss:
            best_loss = metrics['avg_loss']
            best_path = checkpoints_dir / "best_model.pt"
            trainer.save_checkpoint(str(best_path))
            print(f"New best model saved (loss: {best_loss:.4f})")
        
        # Save latest checkpoint
        latest_path = checkpoints_dir / "latest.pt"
        trainer.save_checkpoint(str(latest_path))
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoints_dir}")
    if generate_samples_flag:
        print(f"Samples saved to: {samples_dir}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train Drifting Field Generative Models"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing cached latents",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use dummy dataset for testing",
    )
    parser.add_argument(
        "--num_dummy_samples",
        type=int,
        default=1000,
        help="Number of dummy samples (default: 1000)",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and samples",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_size",
        type=str,
        choices=["small", "base", "large"],
        default="small",
        help="Model size (default: small)",
    )
    parser.add_argument(
        "--latent_size",
        type=int,
        default=32,
        help="Latent spatial size (default: 32 for 256px images)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=2,
        help="Patch size (default: 2)",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=4,
        help="Number of latent channels (default: 4 for SD-VAE)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        help="Number of classes (default: 1000 for ImageNet)",
    )
    parser.add_argument(
        "--num_style_tokens",
        type=int,
        default=32,
        help="Number of style tokens (default: 32)",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size (default: 64)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=5000,
        help="Learning rate warmup steps (default: 5000)",
    )
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.9999,
        help="EMA decay rate (default: 0.9999)",
    )
    
    # CFG arguments
    parser.add_argument(
        "--no_cfg_training",
        action="store_true",
        help="Disable CFG-aware training (default: CFG training is enabled)",
    )
    parser.add_argument(
        "--cfg_min",
        type=float,
        default=1.0,
        help="Minimum CFG scale during training (default: 1.0)",
    )
    parser.add_argument(
        "--cfg_max",
        type=float,
        default=7.5,
        help="Maximum CFG scale during training (default: 7.5)",
    )
    parser.add_argument(
        "--uncond_prob",
        type=float,
        default=0.1,
        help="Probability of unconditional training (default: 0.1)",
    )
    
    # Loss arguments
    parser.add_argument(
        "--feature_extractor",
        type=str,
        default="latent",
        help="Feature extractor type (default: latent)",
    )
    parser.add_argument(
        "--mae_checkpoint",
        type=str,
        default=None,
        help="Path to pretrained MAE encoder checkpoint for feature extractor",
    )
    
    # Queue arguments
    parser.add_argument(
        "--queue_size",
        type=int,
        default=128,
        help="Sample queue size per class (default: 128)",
    )
    
    # Checkpoint arguments
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (default: 5)",
    )
    
    # Visualization arguments
    parser.add_argument(
        "--no_generate_samples",
        action="store_true",
        help="Disable sample generation during training",
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=1,
        help="Generate samples every N epochs (default: 1)",
    )
    parser.add_argument(
        "--sample_classes",
        type=int,
        nargs="+",
        default=[1, 263],
        help="Class IDs for sample generation (default: 1=goldfish, 263=corgi)",
    )
    parser.add_argument(
        "--sample_cfg_scale",
        type=float,
        default=1.5,
        help="CFG scale for sample generation (default: 1.5)",
    )
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=2,
        help="Number of samples per class (default: 2)",
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers (default: 4)",
    )
    
    # Misc arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Log interval in steps (default: 100)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.dummy and args.data_dir is None:
        parser.error("Either --data_dir or --dummy must be specified")
    
    # Run training
    train(args)


if __name__ == "__main__":
    main()
