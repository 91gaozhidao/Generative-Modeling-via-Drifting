"""
Masked Autoencoder (MAE) for Latent Feature Extractor Pretraining

This module implements MAE pretraining for the feature extractor as described
in the paper's Appendix A.3:

> "We use a ResNet-style network... pre-trained with **Masked Autoencoding (MAE)**... 
> on the VAE latent space."
> "We found it crucial to use a feature extractor... raw pixels/latents failed."
> "All residual blocks are 'basic' blocks"
> "ResNet-style encoder... blocks/stage [3, 4, 6, 3]"

The pretrained encoder provides a structured feature space for computing
the drifting field, which is essential for achieving SOTA performance.

Key components:
- BasicBlock: Standard ResNet basic block with skip connection
- LatentMAEEncoder: ResNet-style encoder backbone with configurable blocks per stage
- LatentMAEDecoder: Lightweight decoder for reconstruction
- LatentMAE: Full MAE model with masking, encoding, and decoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math


def _make_norm(channels: int, num_groups: int = 32) -> nn.Module:
    """Create a GroupNorm layer with a safe number of groups.

    If *channels* is not divisible by *num_groups*, fall back to a smaller
    divisor so that the layer can still be instantiated.
    """
    while num_groups > 1 and channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups, channels)


class BasicBlock(nn.Module):
    """
    Basic ResNet block with skip connection (Paper Appendix A.3).
    
    Two conv layers with GroupNorm and ReLU, plus a residual shortcut.
    Per paper Appendix A.3: "GroupNorm (GN) in place of BatchNorm".
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        stride: Stride for the first convolution (for downsampling)
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = _make_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = _make_norm(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _make_norm(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class LatentMAEEncoder(nn.Module):
    """
    ResNet-style Encoder for Latent MAE (Paper Appendix A.3 & Table 8).
    
    Uses BasicBlock residual blocks with configurable blocks per stage.
    Channel widths expand naturally without artificial caps.
    
    Per paper Appendix A.3: for latent-space inputs the encoder uses a 3×3
    stride-1 convolution (no downsampling) as the input stem, and no MaxPool.
    
    Per paper Appendix A.5: features are extracted from the output of every
    2 residual blocks within each stage, together with the final output.
    This provides denser supervision signals for the drifting field.
    
    Args:
        in_channels: Number of input channels (default: 4 for VAE latent)
        hidden_channels: Base hidden dimension / base width (default: 64)
        num_stages: Number of encoder stages (default: 4)
        patch_size: Patch size for masking (default: 4)
        blocks_per_stage: Number of BasicBlocks per stage (default: [3, 4, 6, 3] for L/2)
        extract_every_n: Extract intermediate features every N blocks (default: 2, per A.5)
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 64,
        num_stages: int = 4,
        patch_size: int = 4,
        blocks_per_stage: Optional[List[int]] = None,
        extract_every_n: int = 2,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_stages = num_stages
        self.patch_size = patch_size
        self.extract_every_n = extract_every_n
        
        if blocks_per_stage is None:
            blocks_per_stage = [3, 4, 6, 3]
        
        # Ensure blocks_per_stage matches num_stages
        if len(blocks_per_stage) < num_stages:
            blocks_per_stage = blocks_per_stage + [blocks_per_stage[-1]] * (num_stages - len(blocks_per_stage))
        blocks_per_stage = blocks_per_stage[:num_stages]
        self.blocks_per_stage = blocks_per_stage
        
        # Build encoder stages with proper ResNet BasicBlocks
        # Use ModuleList of ModuleLists to allow intermediate feature extraction
        self.stages = nn.ModuleList()
        
        # Feature dimensions for each extracted feature map
        self.feature_dims = []
        
        in_ch = in_channels
        out_ch = hidden_channels
        
        for i in range(num_stages):
            stride = 2 if i > 0 else 1
            num_blocks = blocks_per_stage[i]
            
            # Build stage as ModuleList of individual blocks
            blocks = nn.ModuleList()
            blocks.append(BasicBlock(in_ch, out_ch, stride=stride))
            for _ in range(1, num_blocks):
                blocks.append(BasicBlock(out_ch, out_ch, stride=1))
            
            self.stages.append(blocks)
            
            # Record feature dims for intermediate extraction points
            for b_idx in range(num_blocks):
                if (b_idx + 1) % extract_every_n == 0 or b_idx == num_blocks - 1:
                    self.feature_dims.append(out_ch)
            
            in_ch = out_ch
            out_ch = out_ch * 2  # Natural channel expansion, no cap
        
        # Final feature dimension (after global average pooling)
        self.final_dim = self.feature_dims[-1]
    
    def _forward_with_intermediates(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract features every ``extract_every_n`` blocks (Paper A.5)."""
        features = []
        for stage_blocks in self.stages:
            num_blocks = len(stage_blocks)
            for b_idx, block in enumerate(stage_blocks):
                x = block(x)
                # Extract at every Nth block and always at the stage end
                if (b_idx + 1) % self.extract_every_n == 0 or b_idx == num_blocks - 1:
                    features.append(x)
        return features
    
    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tuple of (list of multi-scale features, final pooled features)
        """
        features = self._forward_with_intermediates(x)
        
        # Global average pool the final features
        pooled = F.adaptive_avg_pool2d(features[-1], 1).flatten(1)
        
        return features, pooled
    
    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features (compatible with LatentFeatureExtractor).
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            List of feature tensors at each scale (including intermediates)
        """
        return self._forward_with_intermediates(x)


class LatentMAEDecoder(nn.Module):
    """
    Lightweight decoder for MAE reconstruction.
    
    Reconstructs masked patches from encoder features using
    transposed convolutions and residual connections.
    
    Args:
        in_channels: Number of latent channels to reconstruct (default: 4)
        encoder_dim: Dimension of encoder output (default: 512)
        hidden_channels: Hidden dimension (default: 256)
        output_size: Spatial size of output (default: 32)
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        encoder_dim: int = 512,
        hidden_channels: int = 256,
        output_size: int = 32,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.output_size = output_size
        
        # Calculate starting spatial size (after encoder downsampling)
        # Encoder has 3 stride-2 stages (stages 1,2,3), so: 32 -> 16 -> 8 -> 4
        start_size = output_size // 8
        
        # Project encoder features to spatial representation
        self.proj = nn.Sequential(
            nn.Linear(encoder_dim, hidden_channels * start_size * start_size),
            nn.ReLU(inplace=True),
        )
        
        self.start_size = start_size
        self.hidden_channels = hidden_channels
        
        # Upsample decoder blocks (using GroupNorm per paper A.3)
        self.decoder_blocks = nn.ModuleList([
            # 4x4 -> 8x8
            nn.Sequential(
                nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
                _make_norm(hidden_channels),
                nn.ReLU(inplace=True),
            ),
            # 8x8 -> 16x16
            nn.Sequential(
                nn.ConvTranspose2d(hidden_channels, hidden_channels // 2, kernel_size=4, stride=2, padding=1),
                _make_norm(hidden_channels // 2),
                nn.ReLU(inplace=True),
            ),
            # 16x16 -> 32x32
            nn.Sequential(
                nn.ConvTranspose2d(hidden_channels // 2, hidden_channels // 4, kernel_size=4, stride=2, padding=1),
                _make_norm(hidden_channels // 4),
                nn.ReLU(inplace=True),
            ),
        ])
        
        # Final projection to latent channels
        self.final_conv = nn.Conv2d(hidden_channels // 4, in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode features to reconstruct input.
        
        Args:
            features: Encoder features of shape (B, encoder_dim)
            
        Returns:
            Reconstructed latent of shape (B, C, H, W)
        """
        B = features.shape[0]
        
        # Project to spatial
        x = self.proj(features)
        x = x.view(B, self.hidden_channels, self.start_size, self.start_size)
        
        # Upsample
        for block in self.decoder_blocks:
            x = block(x)
        
        # Final projection
        x = self.final_conv(x)
        
        return x


class LatentMAE(nn.Module):
    """
    Masked Autoencoder for VAE Latent Space Pretraining.
    
    This module implements MAE pretraining as described in the paper's Appendix A.3.
    The encoder learns to extract meaningful features from latent representations
    by reconstructing masked patches.
    
    Supports two training modes:
    1. Reconstruction (MAE pretraining): Mask patches, reconstruct, MSE loss
    2. Classification fine-tuning: Add linear head, CrossEntropy loss (Appendix A.3)
    
    Masking Strategy:
    - Divide input into non-overlapping patches
    - Randomly mask 75% of patches (following original MAE paper)
    - Encoder processes only visible patches
    - Decoder reconstructs full input from encoder features
    - Loss computed only on masked regions
    
    Args:
        in_channels: Number of latent channels (default: 4 for SD-VAE)
        hidden_channels: Encoder hidden dimension / base width (default: 64)
        num_stages: Number of encoder stages (default: 4)
        patch_size: Patch size for masking (default: 4)
        mask_ratio: Ratio of patches to mask (default: 0.75)
        input_size: Spatial size of input (default: 32)
        blocks_per_stage: Number of BasicBlocks per stage (default: [3, 4, 6, 3])
        num_classes: Number of classes for classification head (default: 1000)
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 64,
        num_stages: int = 4,
        patch_size: int = 4,
        mask_ratio: float = 0.75,
        input_size: int = 32,
        blocks_per_stage: Optional[List[int]] = None,
        num_classes: int = 1000,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.input_size = input_size
        
        # Number of patches
        self.num_patches_per_dim = input_size // patch_size
        self.num_patches = self.num_patches_per_dim ** 2
        
        # Encoder (ResNet-style with configurable blocks per stage)
        self.encoder = LatentMAEEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_stages=num_stages,
            patch_size=patch_size,
            blocks_per_stage=blocks_per_stage,
        )
        
        # Decoder
        self.decoder = LatentMAEDecoder(
            in_channels=in_channels,
            encoder_dim=self.encoder.final_dim,
            hidden_channels=256,
            output_size=input_size,
        )
        
        # Classification head for fine-tuning (Paper Appendix A.3, Table 3)
        self.classifier_head = nn.Linear(self.encoder.final_dim, num_classes)
        nn.init.trunc_normal_(self.classifier_head.weight, std=0.02)
        nn.init.zeros_(self.classifier_head.bias)
        
        # Mask token for masked patches
        self.mask_token = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        nn.init.normal_(self.mask_token, std=0.02)
    
    def generate_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Generate random patch mask.
        
        Args:
            batch_size: Batch size
            device: Device to create mask on
            
        Returns:
            Boolean mask of shape (B, num_patches) where True = masked
        """
        num_masked = int(self.num_patches * self.mask_ratio)
        
        # Generate random permutation for each sample
        noise = torch.rand(batch_size, self.num_patches, device=device)
        
        # Sort to get random order, then mask the first num_masked
        ids_shuffle = torch.argsort(noise, dim=1)
        
        # Create mask: True for masked positions
        mask = torch.zeros(batch_size, self.num_patches, device=device, dtype=torch.bool)
        mask.scatter_(1, ids_shuffle[:, :num_masked], True)
        
        return mask
    
    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patches.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Patches of shape (B, num_patches, C * patch_size * patch_size)
        """
        B, C, H, W = x.shape
        p = self.patch_size
        
        # Reshape to patches
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 1, 3, 5)  # (B, H//p, W//p, C, p, p)
        x = x.reshape(B, self.num_patches, C * p * p)
        
        return x
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to image.
        
        Args:
            x: Patches of shape (B, num_patches, C * patch_size * patch_size)
            
        Returns:
            Image of shape (B, C, H, W)
        """
        B = x.shape[0]
        p = self.patch_size
        h = w = self.num_patches_per_dim
        C = self.in_channels
        
        x = x.reshape(B, h, w, C, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C, h, p, w, p)
        x = x.reshape(B, C, h * p, w * p)
        
        return x
    
    def apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply mask to input by replacing masked patches with mask token.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            mask: Boolean mask of shape (B, num_patches)
            
        Returns:
            Masked input of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        p = self.patch_size
        
        # Expand mask to spatial dimensions
        # mask: (B, num_patches) -> (B, 1, H, W)
        mask_spatial = mask.view(B, self.num_patches_per_dim, self.num_patches_per_dim)
        mask_spatial = mask_spatial.unsqueeze(1)  # (B, 1, h, w)
        mask_spatial = mask_spatial.repeat_interleave(p, dim=2)  # (B, 1, H, w)
        mask_spatial = mask_spatial.repeat_interleave(p, dim=3)  # (B, 1, H, W)
        
        # Apply mask: replace masked regions with mask token
        mask_token_expanded = self.mask_token.expand(B, C, H, W)
        x_masked = torch.where(mask_spatial, mask_token_expanded, x)
        
        return x_masked
    
    def forward(
        self,
        x: torch.Tensor,
        return_reconstruction: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with masking, encoding, and reconstruction.
        
        Args:
            x: Input latent of shape (B, C, H, W)
            return_reconstruction: Whether to return full reconstruction
            
        Returns:
            Tuple containing:
            - loss: Reconstruction loss (MSE on masked patches only)
            - mask: Boolean mask indicating masked patches
            - (optional) reconstruction: Full reconstructed input
        """
        B, C, H, W = x.shape
        
        # Generate mask
        mask = self.generate_mask(B, x.device)
        
        # Apply mask to input
        x_masked = self.apply_mask(x, mask)
        
        # Encode masked input
        _, features = self.encoder(x_masked)
        
        # Decode to reconstruct
        reconstruction = self.decoder(features)
        
        # Compute loss only on masked patches
        # Convert to patches for loss computation
        x_patches = self.patchify(x)
        recon_patches = self.patchify(reconstruction)
        
        # Masked loss: only compute loss on masked patches
        # mask: (B, num_patches), expand to match patch features
        mask_expanded = mask.unsqueeze(-1).expand_as(x_patches)
        
        # MSE loss on masked patches only
        loss = F.mse_loss(
            recon_patches[mask_expanded],
            x_patches[mask_expanded],
            reduction='mean'
        )
        
        if return_reconstruction:
            return loss, mask, reconstruction
        
        return loss, mask
    
    def get_encoder_state_dict(self) -> dict:
        """
        Get state dict for encoder only (for loading into LatentFeatureExtractor).
        
        Returns:
            State dict of encoder weights
        """
        return self.encoder.state_dict()
    
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full reconstruction without masking (for evaluation).
        
        Args:
            x: Input latent of shape (B, C, H, W)
            
        Returns:
            Reconstructed latent of shape (B, C, H, W)
        """
        _, features = self.encoder(x)
        return self.decoder(features)
    
    def forward_cls(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classification forward pass (Paper Appendix A.3, Table 3).
        
        Passes input through encoder and classification head without masking.
        Used during classification fine-tuning phase.
        
        Args:
            x: Input latent of shape (B, C, H, W)
            
        Returns:
            Class logits of shape (B, num_classes)
        """
        _, pooled = self.encoder(x)
        return self.classifier_head(pooled)


def create_mae(
    in_channels: int = 4,
    hidden_channels: int = 64,
    num_stages: int = 4,
    patch_size: int = 4,
    mask_ratio: float = 0.75,
    input_size: int = 32,
    blocks_per_stage: Optional[List[int]] = None,
    num_classes: int = 1000,
) -> LatentMAE:
    """
    Factory function to create a LatentMAE model.
    
    Args:
        in_channels: Number of latent channels (default: 4)
        hidden_channels: Encoder hidden dimension / base width (default: 64)
        num_stages: Number of encoder stages (default: 4)
        patch_size: Patch size for masking (default: 4)
        mask_ratio: Ratio of patches to mask (default: 0.75)
        input_size: Spatial size of input (default: 32)
        blocks_per_stage: BasicBlocks per stage (default: [3, 4, 6, 3])
        num_classes: Number of classes for classification head (default: 1000)
        
    Returns:
        Configured LatentMAE model
    """
    return LatentMAE(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_stages=num_stages,
        patch_size=patch_size,
        mask_ratio=mask_ratio,
        input_size=input_size,
        blocks_per_stage=blocks_per_stage,
        num_classes=num_classes,
    )
