"""
Drifting Loss Module

This module implements the full Drifting Loss computation in high-dimensional
feature space, as described in the paper.

The loss is computed on features f(x), not pixels, which is critical for
success on complex datasets like ImageNet.

Key components:
- Multi-scale loss at 4 stages of the feature extractor
- Feature normalization (Eq. 20)
- Drift normalization (Eq. 24)
- Multiple temperature support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from .feature_extractor import FeatureExtractor, LatentFeatureExtractor
from ..utils.drifting_field import compute_V, compute_kernel, drifting_loss


class DriftingLoss(nn.Module):
    """
    Drifting Loss module for training generative models.
    
    This module wraps a feature extractor, extracts features at multiple scales,
    applies the normalization formulas, and computes the drifting loss.
    
    The loss encourages the generator to produce samples that:
    1. Move towards the data distribution (attraction)
    2. Move away from other generated samples (repulsion)
    
    Args:
        feature_extractor: Feature extraction network (or 'resnet18', 'latent', etc.)
        temperatures: List of temperature values for multi-scale loss
        sigma: Kernel bandwidth
        normalize_features: Whether to apply feature normalization (Eq. 20)
        normalize_drift: Whether to apply drift normalization (Eq. 24)
        loss_weights: Weights for each feature scale
        stop_gradient: Whether to apply stop-gradient to targets
    """
    
    def __init__(
        self,
        feature_extractor: Union[str, nn.Module] = 'latent',
        temperatures: List[float] = None,
        sigma: float = 1.0,
        normalize_features: bool = True,
        normalize_drift: bool = True,
        loss_weights: Optional[List[float]] = None,
        stop_gradient: bool = True,
        in_channels: int = 4,
    ):
        super().__init__()
        
        if temperatures is None:
            temperatures = [0.1, 0.5, 1.0, 2.0]
        
        self.temperatures = temperatures
        self.sigma = sigma
        self.normalize_features = normalize_features
        self.normalize_drift = normalize_drift
        self.stop_gradient = stop_gradient
        
        # Build feature extractor
        if isinstance(feature_extractor, str):
            if feature_extractor in ('latent', 'mae'):
                self.feature_extractor = LatentFeatureExtractor(in_channels=in_channels)
            elif feature_extractor.startswith('resnet'):
                self.feature_extractor = FeatureExtractor(
                    pretrained_encoder=feature_extractor,
                    in_channels=in_channels,
                )
            else:
                raise ValueError(f"Unknown feature extractor: {feature_extractor}")
        else:
            self.feature_extractor = feature_extractor
        
        # Track whether feature extractor is frozen
        self._feature_extractor_frozen = False
        
        # Loss weights for each scale
        num_scales = 4  # Default 4 scales
        if loss_weights is None:
            loss_weights = [1.0] * num_scales
        self.register_buffer('loss_weights', torch.tensor(loss_weights))
    
    def freeze_feature_extractor(self) -> None:
        """
        Freeze the feature extractor for training.
        
        CRITICAL for single-GPU training with Class-Grouped Sampling (Paper requirement):
        - Sets feature extractor to eval() mode to disable BatchNorm running stats updates
        - Sets requires_grad=False for all parameters to prevent gradient computation
        
        This MUST be called when using pretrained MAE encoder to prevent:
        1. BatchNorm statistics corruption from class-grouped batches
        2. Unintended updates to feature extractor weights during training
        
        Per Paper review: "Feature Extractor (MAE) MUST be in .eval() mode with 
        requires_grad=False to prevent BatchNorm statistics pollution from grouped batches."
        """
        # Set to eval mode - critical for BatchNorm layers
        self.feature_extractor.eval()
        
        # Freeze all parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        self._feature_extractor_frozen = True
    
    def is_feature_extractor_frozen(self) -> bool:
        """
        Check if the feature extractor is properly frozen.
        
        Returns:
            True if feature extractor is in eval mode and all params have requires_grad=False
        """
        if not self._feature_extractor_frozen:
            return False
        
        # Verify all parameters are frozen
        for param in self.feature_extractor.parameters():
            if param.requires_grad:
                return False
        
        # Verify in eval mode
        if self.feature_extractor.training:
            return False
        
        return True
    
    def train(self, mode: bool = True) -> 'DriftingLoss':
        """
        Override train() to keep feature extractor frozen if it was frozen.
        
        This ensures that even when DriftingLoss.train() is called (e.g., when
        the trainer calls model.train()), the feature extractor stays in eval mode.
        
        Args:
            mode: Whether to set training mode (True) or eval mode (False)
            
        Returns:
            self
        """
        super().train(mode)
        
        # Keep feature extractor frozen regardless of training mode
        if self._feature_extractor_frozen:
            self.feature_extractor.eval()
        
        return self
    
    def normalize_feature_map(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Feature Normalization (Eq. 20).
        
        Scale features so that the average pairwise distance is sqrt(D).
        
        Args:
            feat: Feature map of shape (B, C, H, W) or (B, D)
            
        Returns:
            Normalized features with average distance = sqrt(D)
        """
        # Flatten if needed
        if feat.dim() == 4:
            feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B, C)
        
        if not self.normalize_features:
            return feat
        
        # L2 normalize and scale by sqrt(D)
        D = feat.shape[-1]
        feat_normalized = F.normalize(feat, p=2, dim=-1) * (D ** 0.5)
        
        return feat_normalized
    
    def normalize_drift_vector(self, V: torch.Tensor) -> torch.Tensor:
        """
        Drift Normalization (Eq. 24).
        
        Scale drift vectors so that the average norm is 1.
        
        Args:
            V: Drift vectors of shape (B, D)
            
        Returns:
            Normalized drift vectors with average norm = 1
        """
        if not self.normalize_drift:
            return V
        
        norms = torch.norm(V, dim=-1, keepdim=True) + 1e-8
        avg_norm = norms.mean()
        
        return V / (avg_norm + 1e-8)
    
    def compute_scale_loss(
        self,
        feat_gen: torch.Tensor,
        feat_data: torch.Tensor,
        temperature: float,
        num_channels: int = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute drifting loss at a single feature scale.
        
        Args:
            feat_gen: Generated features of shape (B, D)
            feat_data: Data features of shape (N, D)
            temperature: Base temperature for softmax normalization
            num_channels: Number of channels for temperature scaling (Paper Eq. 22)
            
        Returns:
            Tuple of (loss, drift field V)
        """
        # Temperature scaling per Paper Eq. 22: τ_l = τ / sqrt(C_l)
        # This ensures features from different layers contribute equally to the loss
        if num_channels is not None and num_channels > 0:
            scaled_temperature = temperature / (num_channels ** 0.5)
        else:
            scaled_temperature = temperature
        
        # Compute drifting field
        V = compute_V(
            feat_gen, feat_data, feat_gen,
            sigma=self.sigma,
            temperature=scaled_temperature,
            normalize_features=False,  # Already normalized
            normalize_drift=False,  # Normalize after
        )
        
        # Apply drift normalization
        V = self.normalize_drift_vector(V)
        
        # Compute target
        target = feat_gen + V
        if self.stop_gradient:
            target = target.detach()
        
        # Compute loss
        loss = drifting_loss(feat_gen, V, target)
        
        return loss, V
    
    def forward(
        self,
        x_gen: torch.Tensor,
        x_data: torch.Tensor,
        return_drift: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Compute the multi-scale drifting loss.
        
        Args:
            x_gen: Generated samples of shape (B, C, H, W)
            x_data: Data samples of shape (N, C, H, W)
            return_drift: Whether to return drift fields
            
        Returns:
            Total loss (and optionally list of drift fields at each scale)
        """
        # Extract multi-scale features
        feats_gen = self.feature_extractor(x_gen)  # List of (B, C_i, H_i, W_i)
        feats_data = self.feature_extractor(x_data)  # List of (N, C_i, H_i, W_i)
        
        total_loss = 0.0
        drift_fields = []
        
        for scale_idx, (feat_gen, feat_data) in enumerate(zip(feats_gen, feats_data)):
            # Get number of channels for temperature scaling (Paper Eq. 22)
            # Features are 4D (B, C, H, W) at this point before normalization
            # We use the channel count C for scaling: τ_l = τ / sqrt(C_l)
            num_channels = feat_gen.shape[1] if feat_gen.dim() == 4 else feat_gen.shape[-1]
            
            # Normalize features (converts from 4D to 2D via adaptive_avg_pool2d)
            feat_gen_norm = self.normalize_feature_map(feat_gen)
            feat_data_norm = self.normalize_feature_map(feat_data)
            
            # Compute loss at multiple temperatures with channel-based scaling
            scale_loss = 0.0
            scale_drifts = []
            
            for temp in self.temperatures:
                loss, V = self.compute_scale_loss(
                    feat_gen_norm, feat_data_norm, temp, num_channels=num_channels
                )
                scale_loss = scale_loss + loss
                scale_drifts.append(V)
            
            # Average over temperatures
            scale_loss = scale_loss / len(self.temperatures)
            
            # Weight and accumulate
            total_loss = total_loss + self.loss_weights[scale_idx] * scale_loss
            
            if return_drift:
                # Average drift field across temperatures
                avg_drift = sum(scale_drifts) / len(scale_drifts)
                drift_fields.append(avg_drift)
        
        if return_drift:
            return total_loss, drift_fields
        return total_loss
    
    def compute_with_cfg(
        self,
        x_gen: torch.Tensor,
        x_data: torch.Tensor,
        x_uncond: torch.Tensor,
        cfg_scale: float,
    ) -> torch.Tensor:
        """
        Compute drifting loss with Classifier-Free Guidance.
        
        This implements training-time CFG by mixing unconditional samples
        into the negative set based on the CFG scale.
        
        Args:
            x_gen: Generated samples (conditional)
            x_data: Data samples (positive)
            x_uncond: Unconditional data samples (for CFG)
            cfg_scale: CFG scale (gamma)
            
        Returns:
            Loss with CFG
        """
        # Extract features
        feats_gen = self.feature_extractor(x_gen)
        feats_data = self.feature_extractor(x_data)
        feats_uncond = self.feature_extractor(x_uncond)
        
        total_loss = 0.0
        
        for scale_idx, (feat_gen, feat_data, feat_uncond) in enumerate(
            zip(feats_gen, feats_data, feats_uncond)
        ):
            # Get number of channels for temperature scaling (Paper Eq. 22)
            # Features are 4D (B, C, H, W) at this point before normalization
            # We use the channel count C for scaling: τ_l = τ / sqrt(C_l)
            num_channels = feat_gen.shape[1] if feat_gen.dim() == 4 else feat_gen.shape[-1]
            
            # Normalize features (converts from 4D to 2D via adaptive_avg_pool2d)
            feat_gen_norm = self.normalize_feature_map(feat_gen)
            feat_data_norm = self.normalize_feature_map(feat_data)
            feat_uncond_norm = self.normalize_feature_map(feat_uncond)
            
            # Mix unconditional samples into negatives based on CFG scale
            # Higher CFG = more emphasis on conditional, less on unconditional
            w_uncond = 1.0 / (cfg_scale + 1)  # Weight for unconditional
            
            # Compute loss at multiple temperatures with channel-based scaling
            scale_loss = 0.0
            
            for temp in self.temperatures:
                # Apply temperature scaling per Paper Eq. 22: τ_l = τ / sqrt(C_l)
                scaled_temp = temp / (num_channels ** 0.5) if num_channels > 0 else temp
                
                # Conditional drifting field
                V_cond = compute_V(
                    feat_gen_norm, feat_data_norm, feat_gen_norm,
                    sigma=self.sigma,
                    temperature=scaled_temp,
                    normalize_features=False,
                    normalize_drift=False,
                )
                
                # Unconditional drifting field (repulsion from unconditional data)
                V_uncond = compute_V(
                    feat_gen_norm, feat_uncond_norm, feat_gen_norm,
                    sigma=self.sigma,
                    temperature=scaled_temp,
                    normalize_features=False,
                    normalize_drift=False,
                )
                
                # Combine with CFG weighting
                V = (1 - w_uncond) * V_cond + w_uncond * V_uncond
                V = self.normalize_drift_vector(V)
                
                # Compute target and loss
                target = feat_gen_norm + V
                if self.stop_gradient:
                    target = target.detach()
                
                loss = drifting_loss(feat_gen_norm, V, target)
                scale_loss = scale_loss + loss
            
            scale_loss = scale_loss / len(self.temperatures)
            total_loss = total_loss + self.loss_weights[scale_idx] * scale_loss
        
        return total_loss


class SimpleDriftingLoss(nn.Module):
    """
    Simplified Drifting Loss for toy problems and testing.
    
    Works directly on input vectors without feature extraction.
    """
    
    def __init__(
        self,
        sigma: float = 1.0,
        temperature: float = 1.0,
        normalize_features: bool = True,
        normalize_drift: bool = True,
    ):
        super().__init__()
        self.sigma = sigma
        self.temperature = temperature
        self.normalize_features = normalize_features
        self.normalize_drift = normalize_drift
    
    def forward(
        self,
        x_gen: torch.Tensor,
        x_data: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute drifting loss.
        
        Args:
            x_gen: Generated samples of shape (B, D)
            x_data: Data samples of shape (N, D)
            
        Returns:
            Scalar loss
        """
        V = compute_V(
            x_gen, x_data, x_gen,
            sigma=self.sigma,
            temperature=self.temperature,
            normalize_features=self.normalize_features,
            normalize_drift=self.normalize_drift,
        )
        
        target = (x_gen + V).detach()
        return drifting_loss(x_gen, V, target)
