"""
MAE Pre-training Script for Latent Feature Extractor

This script pre-trains the feature extractor using Masked Autoencoding (MAE)
on VAE latent representations, as described in the paper's Appendix A.3:

> "We use a ResNet-style network... pre-trained with Masked Autoencoding (MAE)... 
> on the VAE latent space."
> "We found it crucial to use a feature extractor... raw pixels/latents failed."

The pretrained encoder provides a structured feature space for computing
the drifting field, which is essential for achieving SOTA performance.

Usage:
    # Train MAE on pre-cached latents
    python tools/train_mae.py --data_dir ./data/latents --output_dir ./weights/mae
    
    # Train on dummy data for testing
    python tools/train_mae.py --dummy --output_dir ./weights/mae
    
    # Resume from checkpoint
    python tools/train_mae.py --data_dir ./data/latents --resume ./weights/mae/checkpoint.pt
"""

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from drifting.models.mae import LatentMAE, create_mae
from drifting.data.dataset import (
    LatentDataset,
    DummyLatentDataset,
    create_dataloader,
    create_dummy_dataloader,
)


def train_epoch(
    model: LatentMAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    log_interval: int = 100,
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: LatentMAE model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        log_interval: Logging interval
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (latents, _) in enumerate(pbar):
        latents = latents.to(device)
        
        # Forward pass
        loss, _ = model(latents)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % log_interval == 0:
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / max(num_batches, 1)
    
    return {
        'avg_loss': avg_loss,
        'num_batches': num_batches,
    }


@torch.no_grad()
def validate(
    model: LatentMAE,
    dataloader: DataLoader,
    device: str,
) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        model: LatentMAE model
        dataloader: Validation data loader
        device: Device to validate on
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    for latents, _ in dataloader:
        latents = latents.to(device)
        
        loss, _ = model(latents)
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    
    return {
        'val_loss': avg_loss,
        'num_batches': num_batches,
    }


def train_cls_step(
    model: LatentMAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    max_iters: int = 3000,
    log_interval: int = 100,
) -> Dict[str, float]:
    """
    Classification fine-tuning loop (Paper Appendix A.3, Table 3).
    
    Fine-tunes all parameters with CrossEntropy loss for a fixed number
    of iterations (default 3k as per paper).
    
    Args:
        model: LatentMAE model (with classifier_head)
        dataloader: Training data loader (must yield (latents, labels))
        optimizer: Optimizer
        device: Device to train on
        max_iters: Maximum training iterations (default: 3000 per paper)
        log_interval: Logging interval
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    cur_iter = 0
    
    pbar = tqdm(total=max_iters, desc="Classification Fine-tuning")
    
    while cur_iter < max_iters:
        for latents, labels in dataloader:
            if cur_iter >= max_iters:
                break
            
            latents = latents.to(device)
            labels = labels.to(device)
            
            # Forward through encoder + classification head
            logits = model.forward_cls(latents)
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.shape[0]
            cur_iter += 1
            
            if cur_iter % log_interval == 0:
                acc = total_correct / max(total_samples, 1)
                avg = total_loss / cur_iter
                pbar.set_postfix({'loss': f"{avg:.4f}", 'acc': f"{acc:.4f}"})
                pbar.update(log_interval)
    
    pbar.close()
    
    avg_loss = total_loss / max(cur_iter, 1)
    accuracy = total_correct / max(total_samples, 1)
    
    return {
        'avg_loss': avg_loss,
        'accuracy': accuracy,
        'num_iters': cur_iter,
    }


def save_checkpoint(
    model: LatentMAE,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    best_loss: float,
    output_path: str,
) -> None:
    """
    Save training checkpoint.
    
    Args:
        model: LatentMAE model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        epoch: Current epoch
        best_loss: Best validation loss
        output_path: Path to save checkpoint
    """
    checkpoint = {
        'model': model.state_dict(),
        'encoder_state_dict': model.get_encoder_state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_loss': best_loss,
    }
    
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    
    torch.save(checkpoint, output_path)


def save_encoder(
    model: LatentMAE,
    output_path: str,
) -> None:
    """
    Save encoder weights only (for loading into LatentFeatureExtractor).
    
    Args:
        model: LatentMAE model
        output_path: Path to save encoder weights
    """
    encoder_state = model.get_encoder_state_dict()
    torch.save({
        'encoder_state_dict': encoder_state,
        'state_dict': encoder_state,  # Alternative key for compatibility
    }, output_path)
    print(f"Saved encoder weights to {output_path}")


def load_checkpoint(
    model: LatentMAE,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    checkpoint_path: str,
    device: str,
) -> tuple:
    """
    Load training checkpoint.
    
    Args:
        model: LatentMAE model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        checkpoint_path: Path to checkpoint
        device: Device to load checkpoint to
        
    Returns:
        Tuple of (start_epoch, best_loss)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    start_epoch = checkpoint.get('epoch', 0) + 1
    best_loss = checkpoint.get('best_loss', float('inf'))
    
    return start_epoch, best_loss


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train feature extractor using MAE on VAE latents"
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
        default="./weights/mae",
        help="Output directory for checkpoints",
    )
    
    # Model arguments
    parser.add_argument(
        "--hidden_channels",
        type=int,
        default=64,
        help="Hidden dimension for encoder (default: 64)",
    )
    parser.add_argument(
        "--num_stages",
        type=int,
        default=4,
        help="Number of encoder stages (default: 4)",
    )
    parser.add_argument(
        "--patch_size",
        type=int,
        default=4,
        help="Patch size for masking (default: 4)",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.75,
        help="Ratio of patches to mask (default: 0.75)",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=4,
        help="Number of latent channels (default: 4 for SD-VAE)",
    )
    parser.add_argument(
        "--latent_size",
        type=int,
        default=32,
        help="Latent spatial size (default: 32)",
    )
    parser.add_argument(
        "--blocks_per_stage",
        type=int,
        nargs='+',
        default=[3, 4, 6, 3],
        help="Number of BasicBlocks per stage (default: 3 4 6 3 for L/2 alignment)",
    )
    parser.add_argument(
        "--finetune_cls",
        action="store_true",
        help="Enable classification fine-tuning mode (Paper Appendix A.3). "
             "Requires --resume to load pre-trained MAE weights.",
    )
    parser.add_argument(
        "--cls_iters",
        type=int,
        default=3000,
        help="Number of classification fine-tuning iterations (default: 3000 per paper)",
    )
    parser.add_argument(
        "--cls_lr",
        type=float,
        default=1e-3,
        help="Learning rate for classification fine-tuning (default: 1e-3)",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size (default: 128)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.5e-4,
        help="Learning rate (default: 1.5e-4)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay (default: 0.05)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Warmup epochs (default: 5)",
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
        help="Log interval in batches (default: 100)",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        help="Number of classes for classification head and dummy dataset (default: 1000)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.dummy and args.data_dir is None:
        parser.error("Either --data_dir or --dummy must be specified")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2, default=str)
    print(f"Configuration saved to {output_dir / 'config.json'}")
    
    # Setup device
    device = args.device
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    print("\nSetting up data...")
    if args.dummy:
        print(f"Using dummy dataset with {args.num_dummy_samples} samples")
        dataloader = create_dummy_dataloader(
            num_samples=args.num_dummy_samples,
            num_classes=args.num_classes,
            batch_size=args.batch_size,
            latent_shape=(args.in_channels, args.latent_size, args.latent_size),
            seed=args.seed,
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
    print("\nCreating MAE model...")
    model = create_mae(
        in_channels=args.in_channels,
        hidden_channels=args.hidden_channels,
        num_stages=args.num_stages,
        patch_size=args.patch_size,
        mask_ratio=args.mask_ratio,
        input_size=args.latent_size,
        blocks_per_stage=args.blocks_per_stage,
        num_classes=args.num_classes,
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    
    # Create scheduler with warmup
    total_steps = args.epochs * len(dataloader)
    warmup_steps = args.warmup_epochs * len(dataloader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * math.pi)).item())
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        start_epoch, best_loss = load_checkpoint(
            model, optimizer, scheduler, args.resume, device
        )
        print(f"Resuming from epoch {start_epoch}, best loss: {best_loss:.4f}")
    
    # Classification fine-tuning mode (Paper Appendix A.3, Table 3)
    if args.finetune_cls:
        if args.resume is None:
            parser.error("--finetune_cls requires --resume to load pre-trained MAE weights")
        
        print(f"\n{'='*60}")
        print(f"Classification Fine-tuning for {args.cls_iters} iterations")
        print(f"{'='*60}\n")
        
        # Create a separate optimizer for fine-tuning (fine-tune all per paper)
        cls_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.cls_lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
        
        cls_metrics = train_cls_step(
            model=model,
            dataloader=dataloader,
            optimizer=cls_optimizer,
            device=device,
            max_iters=args.cls_iters,
            log_interval=args.log_interval,
        )
        
        print(f"\nClassification fine-tuning complete!")
        print(f"  Final Loss: {cls_metrics['avg_loss']:.4f}")
        print(f"  Accuracy: {cls_metrics['accuracy']:.4f}")
        
        # Save fine-tuned model
        ft_path = output_dir / "finetuned_cls.pt"
        save_checkpoint(
            model, cls_optimizer, None, 0, cls_metrics['avg_loss'],
            str(ft_path)
        )
        
        # Save encoder with classification fine-tuning
        ft_encoder_path = output_dir / "finetuned_encoder.pt"
        save_encoder(model, str(ft_encoder_path))
        
        print(f"Fine-tuned encoder saved to {ft_encoder_path}")
        print(f"{'='*60}\n")
        return
    
    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting MAE pre-training for {args.epochs} epochs")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train one epoch
        train_metrics = train_epoch(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            epoch=epoch + 1,
            log_interval=args.log_interval,
        )
        
        # Step scheduler
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Print metrics
        print(f"\nEpoch {epoch + 1}/{args.epochs} completed in {epoch_time:.1f}s")
        print(f"  Average Loss: {train_metrics['avg_loss']:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt"
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, best_loss,
                str(checkpoint_path)
            )
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if train_metrics['avg_loss'] < best_loss:
            best_loss = train_metrics['avg_loss']
            
            # Save full model checkpoint
            best_path = output_dir / "best_model.pt"
            save_checkpoint(
                model, optimizer, scheduler, epoch + 1, best_loss,
                str(best_path)
            )
            
            # Save encoder only (for loading into LatentFeatureExtractor)
            encoder_path = output_dir / "best_encoder.pt"
            save_encoder(model, str(encoder_path))
            
            print(f"New best model saved (loss: {best_loss:.4f})")
        
        # Save latest checkpoint
        latest_path = output_dir / "latest.pt"
        save_checkpoint(
            model, optimizer, scheduler, epoch + 1, best_loss,
            str(latest_path)
        )
    
    # Save final encoder
    final_encoder_path = output_dir / "final_encoder.pt"
    save_encoder(model, str(final_encoder_path))
    
    print(f"\n{'='*60}")
    print("MAE pre-training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Encoder weights saved to: {output_dir / 'best_encoder.pt'}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
