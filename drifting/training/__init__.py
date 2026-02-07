"""
Training Loop for Drifting Field Generative Models

This module implements the complete training loop with:
- Sample queue management
- Stop-gradient for fixed-point iteration
- Classifier-Free Guidance (CFG) at training time
- Mixed unconditional samples for CFG

The key insight is that unlike Diffusion where CFG happens at inference,
Drifting Models bake CFG into the training objective.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
import os
import warnings
from tqdm import tqdm

from ..models.drifting_dit import DriftingDiT
from ..models.drifting_loss import DriftingLoss
from ..data.sample_queue import SampleQueue


class DriftingTrainer:
    """
    Trainer for Drifting Field Generative Models.
    
    Implements the complete training loop including:
    - Sample queue for positive attractors
    - Stop-gradient for fixed-point iteration
    - Training-time CFG with unconditional samples
    
    Args:
        model: DriftingDiT generator model
        loss_fn: DriftingLoss module
        optimizer: Optimizer
        scheduler: Optional learning rate scheduler
        device: Device to train on
        queue_size: Size of sample queue per class
        num_classes: Number of classes
        cfg_scale_range: Range of CFG scales to sample from [min, max]
        uncond_prob: Probability of using unconditional training
        latent_shape: Shape of latent samples (C, H, W)
        ema_decay: EMA decay for model weights (0 to disable)
    """
    
    def __init__(
        self,
        model: DriftingDiT,
        loss_fn: DriftingLoss,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        queue_size: int = 128,
        num_classes: int = 1000,
        cfg_scale_range: tuple = (1.0, 7.5),
        uncond_prob: float = 0.1,
        latent_shape: tuple = (4, 32, 32),
        ema_decay: float = 0.9999,
    ):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.cfg_scale_range = cfg_scale_range
        self.uncond_prob = uncond_prob
        self.latent_shape = latent_shape
        
        # Sample queue
        self.sample_queue = SampleQueue(
            queue_size=queue_size,
            num_classes=num_classes,
            latent_shape=latent_shape,
            device=device,
        )
        
        # EMA model
        self.ema_decay = ema_decay
        if ema_decay > 0:
            self.model_ema = self._create_ema_model()
        else:
            self.model_ema = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
    
    def _create_ema_model(self) -> nn.Module:
        """Create EMA copy of the model."""
        import copy
        ema_model = copy.deepcopy(self.model)
        for param in ema_model.parameters():
            param.requires_grad = False
        return ema_model
    
    @torch.no_grad()
    def _update_ema(self):
        """Update EMA model weights."""
        if self.model_ema is None:
            return
        
        for ema_param, model_param in zip(
            self.model_ema.parameters(),
            self.model.parameters()
        ):
            ema_param.data.mul_(self.ema_decay).add_(
                model_param.data, alpha=1 - self.ema_decay
            )
    
    def train_step(
        self,
        x_data: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Single training step.
        
        Args:
            x_data: Real latent samples of shape (B, C, H, W)
            labels: Class labels of shape (B,)
            
        Returns:
            Dictionary of loss values
        """
        batch_size = x_data.shape[0]
        
        # Apply VAE latent scaling factor (SD standard)
        x_data = x_data * 0.18215
        
        # Add samples to queue
        self.sample_queue.add(x_data, labels)
        
        # Skip if queue not ready
        if not self.sample_queue.is_ready():
            return {'loss': 0.0, 'skip': True}
        
        # Randomly drop labels for unconditional training (CFG)
        uncond_mask = torch.rand(batch_size, device=self.device) < self.uncond_prob
        labels_train = labels.clone()
        labels_train[uncond_mask] = self.sample_queue.num_classes  # Unconditional label
        
        # Sample CFG scale
        cfg_scale = torch.empty(1).uniform_(*self.cfg_scale_range).item()
        
        # Sample noise and generate with BF16 mixed precision
        z = torch.randn(batch_size, *self.latent_shape, device=self.device)
        
        # Use BF16 autocast for A100/Ampere GPUs (CUDA) or skip for CPU/other devices
        autocast_enabled = self.device == "cuda"
        autocast_device = self.device if self.device in ["cuda", "cpu"] else "cpu"
        with torch.amp.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=autocast_enabled):
            x_gen = self.model(z, labels_train, cfg_scale=cfg_scale)
            
            # Get positive samples from queue
            x_pos = self.sample_queue.sample(labels, num_samples=1)
            
            # Compute drifting loss
            loss = self.loss_fn(x_gen, x_pos)
        
        # Add unconditional samples to queue
        if uncond_mask.any():
            self.sample_queue.add_unconditional(x_data[uncond_mask])
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping and norm calculation
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        grad_norm_value = grad_norm.item()
        
        # Warn if gradient norm is zero (potential training issue)
        if grad_norm_value == 0.0:
            warnings.warn("Gradient norm is 0.0 - model may not be learning!")
        
        self.optimizer.step()
        
        # Update EMA
        self._update_ema()
        
        self.global_step += 1
        
        return {
            'loss': loss.item(),
            'cfg_scale': cfg_scale,
            'uncond_ratio': uncond_mask.float().mean().item(),
            'grad_norm': grad_norm_value,
        }
    
    def train_step_with_cfg(
        self,
        x_data: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Training step with explicit CFG handling.
        
        This version computes separate conditional and unconditional losses
        and combines them based on the CFG scale.
        
        Args:
            x_data: Real latent samples of shape (B, C, H, W)
            labels: Class labels of shape (B,)
            
        Returns:
            Dictionary of loss values
        """
        batch_size = x_data.shape[0]
        
        # Apply VAE latent scaling factor (SD standard)
        x_data = x_data * 0.18215
        
        # Add samples to queue
        self.sample_queue.add(x_data, labels)
        
        if not self.sample_queue.is_ready():
            return {'loss': 0.0, 'skip': True}
        
        # Sample CFG scale
        cfg_scale = torch.empty(1).uniform_(*self.cfg_scale_range).item()
        
        # Sample noise
        z = torch.randn(batch_size, *self.latent_shape, device=self.device)
        
        # Use BF16 autocast for A100/Ampere GPUs (CUDA) or skip for CPU/other devices
        autocast_enabled = self.device == "cuda"
        autocast_device = self.device if self.device in ["cuda", "cpu"] else "cpu"
        with torch.amp.autocast(device_type=autocast_device, dtype=torch.bfloat16, enabled=autocast_enabled):
            # Generate conditional samples
            x_gen_cond = self.model(z, labels, cfg_scale=cfg_scale)
            
            # Generate unconditional samples
            uncond_labels = torch.full(
                (batch_size,), self.sample_queue.num_classes,
                device=self.device, dtype=torch.long
            )
            x_gen_uncond = self.model(z, uncond_labels, cfg_scale=1.0)
            
            # Get positive and unconditional samples from queue
            x_pos = self.sample_queue.sample(labels, num_samples=1)
            x_uncond = self.sample_queue.sample_unconditional(batch_size)
            
            # Compute CFG-weighted loss
            loss = self.loss_fn.compute_with_cfg(
                x_gen_cond, x_pos, x_uncond, cfg_scale
            )
            
            # Also add unconditional loss
            loss_uncond = self.loss_fn(x_gen_uncond, x_uncond)
            
            # Combine losses
            total_loss = loss + self.uncond_prob * loss_uncond
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping and norm calculation
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        grad_norm_value = grad_norm.item()
        
        # Warn if gradient norm is zero (potential training issue)
        if grad_norm_value == 0.0:
            warnings.warn("Gradient norm is 0.0 - model may not be learning!")
        
        self.optimizer.step()
        
        self._update_ema()
        self.global_step += 1
        
        return {
            'loss': total_loss.item(),
            'loss_cond': loss.item(),
            'loss_uncond': loss_uncond.item(),
            'cfg_scale': cfg_scale,
            'grad_norm': grad_norm_value,
        }
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        use_cfg_training: bool = True,
        log_interval: int = 100,
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            dataloader: DataLoader providing (latents, labels)
            use_cfg_training: Whether to use CFG-aware training
            log_interval: Logging interval
            
        Returns:
            Dictionary of average metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_steps = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (x_data, labels) in enumerate(pbar):
            x_data = x_data.to(self.device)
            labels = labels.to(self.device)
            
            if use_cfg_training:
                metrics = self.train_step_with_cfg(x_data, labels)
            else:
                metrics = self.train_step(x_data, labels)
            
            if metrics.get('skip', False):
                continue
            
            total_loss += metrics['loss']
            num_steps += 1
            
            if batch_idx % log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'cfg': f"{metrics.get('cfg_scale', 0):.2f}",
                    'grad_norm': f"{metrics.get('grad_norm', 0):.4f}",
                })
        
        self.epoch += 1
        
        if self.scheduler is not None:
            self.scheduler.step()
        
        return {
            'avg_loss': total_loss / max(num_steps, 1),
            'num_steps': num_steps,
        }
    
    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        labels: torch.Tensor,
        cfg_scale: float = 1.0,
        use_ema: bool = True,
    ) -> torch.Tensor:
        """
        Generate samples (one-step inference!).
        
        Args:
            batch_size: Number of samples to generate
            labels: Class labels of shape (batch_size,)
            cfg_scale: CFG scale for conditioning strength
            use_ema: Whether to use EMA model
            
        Returns:
            Generated latent samples of shape (B, C, H, W)
        """
        model = self.model_ema if (use_ema and self.model_ema is not None) else self.model
        model.eval()
        
        return model.generate(batch_size, labels, cfg_scale, device=self.device)
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'queue_stats': self.sample_queue.stats(),
        }
        
        if self.model_ema is not None:
            checkpoint['model_ema'] = self.model_ema.state_dict()
        
        if self.scheduler is not None:
            checkpoint['scheduler'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        if 'model_ema' in checkpoint and self.model_ema is not None:
            self.model_ema.load_state_dict(checkpoint['model_ema'])
        
        if 'scheduler' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler'])


def create_trainer(
    model: DriftingDiT,
    feature_extractor: str = 'latent',
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    warmup_steps: int = 5000,
    total_steps: int = 1000000,
    device: str = "cuda",
    mae_checkpoint: str = None,
    **kwargs,
) -> DriftingTrainer:
    """
    Create a DriftingTrainer with default settings.
    
    Args:
        model: DriftingDiT model
        feature_extractor: Feature extractor type
        learning_rate: Learning rate
        weight_decay: Weight decay
        warmup_steps: Warmup steps for scheduler
        total_steps: Total training steps
        device: Device to train on
        mae_checkpoint: Path to pretrained MAE encoder checkpoint (optional).
                       If provided, loads pretrained weights into the feature extractor.
                       This is crucial for achieving SOTA performance per paper Appendix A.3.
        **kwargs: Additional arguments for DriftingTrainer
        
    Returns:
        Configured DriftingTrainer
    """
    # Create loss function
    loss_fn = DriftingLoss(
        feature_extractor=feature_extractor,
        in_channels=model.in_chans,
    )
    
    # Load pretrained MAE encoder weights if checkpoint is provided
    # Per paper Appendix A.3: "We found it crucial to use a feature extractor...
    # raw pixels/latents failed."
    if mae_checkpoint is not None:
        print(f"Loading pretrained MAE encoder from: {mae_checkpoint}")
        if hasattr(loss_fn.feature_extractor, 'load_pretrained'):
            loss_fn.feature_extractor.load_pretrained(mae_checkpoint, strict=False)
            print("Successfully loaded pretrained feature extractor weights")
        else:
            print("Warning: Feature extractor does not support load_pretrained method")
    
    # Create optimizer with Paper Table 8 settings: no weight decay, standard Adam betas
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    
    # Create cosine scheduler with warmup
    import math
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * math.pi)).item())
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return DriftingTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        **kwargs,
    )
