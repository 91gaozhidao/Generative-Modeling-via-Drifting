"""
Feature Extractor for Drifting Loss

This module implements the Feature Extractor (ResNet-style encoder) that extracts
multi-scale features for computing the drifting loss.

The paper states the method fails on ImageNet without a feature encoder.
The loss is computed on features f(x), not pixels.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import torchvision.models as models


def _make_norm(channels: int, num_groups: int = 32) -> nn.Module:
    """Create a GroupNorm layer with a safe number of groups.

    If *channels* is not divisible by *num_groups*, fall back to a smaller
    divisor so that the layer can still be instantiated.
    """
    while num_groups > 1 and channels % num_groups != 0:
        num_groups //= 2
    return nn.GroupNorm(num_groups, channels)


class ResNetBlock(nn.Module):
    """Basic ResNet block with skip connection and GroupNorm (Paper A.3)."""
    
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


class FeatureExtractor(nn.Module):
    """
    Multi-scale Feature Extractor for computing drifting loss.
    
    Extracts features at multiple scales (stages) of a ResNet-style encoder
    to capture different granularity of the target distribution.
    
    The features are used to compute the drifting field V(x) in feature space,
    which is more meaningful than pixel space for complex images.
    
    Args:
        in_channels: Number of input channels (default: 4 for VAE latent)
        base_channels: Base number of channels (default: 64)
        num_blocks: Number of blocks per stage (default: [2, 2, 2, 2])
        pretrained_encoder: Optional pretrained encoder to use (e.g., 'resnet18')
        freeze_pretrained: Whether to freeze pretrained weights
        is_latent: If True, use a 3×3 stride-1 stem without MaxPool for
            latent-space inputs (Paper A.3). Default False.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        base_channels: int = 64,
        num_blocks: List[int] = None,
        pretrained_encoder: Optional[str] = None,
        freeze_pretrained: bool = True,
        is_latent: bool = False,
    ):
        super().__init__()
        
        if num_blocks is None:
            num_blocks = [2, 2, 2, 2]
        
        self.in_channels = in_channels
        self.pretrained_encoder = pretrained_encoder
        self.is_latent = is_latent
        
        if pretrained_encoder:
            self._build_pretrained(pretrained_encoder, freeze_pretrained)
        else:
            self._build_custom(in_channels, base_channels, num_blocks)
    
    def _build_pretrained(self, encoder_name: str, freeze: bool):
        """Build using a pretrained encoder."""
        if encoder_name == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif encoder_name == 'resnet34':
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif encoder_name == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")
        
        # Modify first conv if input channels != 3
        if self.in_channels != 3:
            self.input_conv = nn.Conv2d(
                self.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        else:
            self.input_conv = resnet.conv1
        
        self.norm1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.stage1 = resnet.layer1  # Output: 64 channels
        self.stage2 = resnet.layer2  # Output: 128 channels
        self.stage3 = resnet.layer3  # Output: 256 channels
        self.stage4 = resnet.layer4  # Output: 512 channels
        
        # Feature dimensions at each stage
        if encoder_name in ['resnet18', 'resnet34']:
            self.feature_dims = [64, 128, 256, 512]
        else:  # resnet50, etc.
            self.feature_dims = [256, 512, 1024, 2048]
        
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            # Unfreeze input conv if modified
            if self.in_channels != 3:
                for param in self.input_conv.parameters():
                    param.requires_grad = True
    
    def _build_custom(self, in_channels: int, base_channels: int, num_blocks: List[int]):
        """Build a custom ResNet-style encoder with GroupNorm (Paper A.3).
        
        When ``self.is_latent`` is True the input stem uses a 3×3 stride-1
        convolution **without** MaxPool, preserving the spatial resolution of
        small latent inputs (e.g. 32×32).  Otherwise the standard 7×7 stride-2
        + MaxPool stem is used.
        """
        if self.is_latent:
            # Latent mode: 3×3 stride-1, no MaxPool (Paper A.3)
            self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            # Standard ResNet stem: 7×7 stride-2
            self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = _make_norm(base_channels)
        self.relu = nn.ReLU(inplace=True)
        if not self.is_latent:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Build stages
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        
        self.stage1 = self._make_stage(base_channels, channels[0], num_blocks[0], stride=1)
        self.stage2 = self._make_stage(channels[0], channels[1], num_blocks[1], stride=2)
        self.stage3 = self._make_stage(channels[1], channels[2], num_blocks[2], stride=2)
        self.stage4 = self._make_stage(channels[2], channels[3], num_blocks[3], stride=2)
        
        self.feature_dims = channels
    
    def _make_stage(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Create a stage with multiple blocks."""
        layers = [ResNetBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract multi-scale features.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            List of feature tensors at 4 scales
        """
        # Initial processing
        x = self.input_conv(x)
        x = self.norm1(x)
        x = self.relu(x)
        if not self.is_latent:
            x = self.maxpool(x)
        
        # Extract features at each stage
        features = []
        
        x = self.stage1(x)
        features.append(x)
        
        x = self.stage2(x)
        features.append(x)
        
        x = self.stage3(x)
        features.append(x)
        
        x = self.stage4(x)
        features.append(x)
        
        return features
    
    def extract_flat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features and flatten to a single vector.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Flattened features of shape (B, D)
        """
        features = self.forward(x)
        
        # Global average pooling and concatenation
        pooled = []
        for feat in features:
            pooled.append(F.adaptive_avg_pool2d(feat, 1).flatten(1))
        
        return torch.cat(pooled, dim=-1)


class LatentFeatureExtractor(nn.Module):
    """
    Lightweight feature extractor for VAE latent space.
    
    Since the latent space is already a good representation,
    we use a ResNet-style encoder with BasicBlocks.
    
    Per paper Appendix A.3: for latent-space inputs the first stage uses
    stride 1 (no downsampling) — no stem conv or MaxPool.
    
    Per paper Appendix A.5: features are extracted from the output of every
    2 residual blocks within each stage, together with the final output.
    
    This encoder can be pretrained using MAE (Masked Autoencoding) as described
    in the paper's Appendix A.3:
    > "We use a ResNet-style network... pre-trained with MAE... on the VAE latent space."
    > "We found it crucial to use a feature extractor... raw pixels/latents failed."
    > "All residual blocks are 'basic' blocks"
    > "ResNet-style encoder... blocks/stage [3, 4, 6, 3]"
    
    Args:
        in_channels: Input channels (default: 4 for VAE latent)
        hidden_channels: Base width (default: 64)
        num_stages: Number of encoder stages (default: 4)
        blocks_per_stage: Blocks per stage (default: [3, 4, 6, 3])
        extract_every_n: Extract intermediate features every N blocks (default: 2, per A.5)
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 64,
        num_stages: int = 4,
        blocks_per_stage: Optional[List[int]] = None,
        extract_every_n: int = 2,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_stages = num_stages
        self.extract_every_n = extract_every_n
        self.stages = nn.ModuleList()
        
        if blocks_per_stage is None:
            blocks_per_stage = [3, 4, 6, 3]
        
        # Ensure blocks_per_stage matches num_stages
        if len(blocks_per_stage) < num_stages:
            blocks_per_stage = blocks_per_stage + [blocks_per_stage[-1]] * (num_stages - len(blocks_per_stage))
        blocks_per_stage = blocks_per_stage[:num_stages]
        
        in_ch = in_channels
        out_ch = hidden_channels
        
        # Feature dims: one entry per extraction point
        self.feature_dims = []
        
        for i in range(num_stages):
            stride = 2 if i > 0 else 1
            num_blocks = blocks_per_stage[i]
            
            # Build stage as ModuleList to allow per-block feature extraction
            blocks = nn.ModuleList()
            blocks.append(ResNetBlock(in_ch, out_ch, stride=stride))
            for _ in range(1, num_blocks):
                blocks.append(ResNetBlock(out_ch, out_ch, stride=1))
            
            self.stages.append(blocks)
            
            # Record dims for each extraction point in this stage
            for b_idx in range(num_blocks):
                if (b_idx + 1) % extract_every_n == 0 or b_idx == num_blocks - 1:
                    self.feature_dims.append(out_ch)
            
            in_ch = out_ch
            out_ch = out_ch * 2  # Natural expansion, no cap
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features with intermediate extraction (Paper A.5)."""
        features = []
        for stage_blocks in self.stages:
            num_blocks = len(stage_blocks)
            for b_idx, block in enumerate(stage_blocks):
                x = block(x)
                # Extract at every Nth block and always at the stage end
                if (b_idx + 1) % self.extract_every_n == 0 or b_idx == num_blocks - 1:
                    features.append(x)
        return features
    
    def load_pretrained(self, checkpoint_path: str, strict: bool = True) -> None:
        """
        Load pretrained weights from MAE encoder checkpoint.
        
        This method loads weights from a pretrained MAE encoder, enabling
        the feature extractor to provide meaningful feature representations
        for the drifting loss computation.
        
        Args:
            checkpoint_path: Path to the MAE encoder checkpoint (.pt file)
            strict: Whether to strictly enforce that the keys in state_dict
                   match the keys returned by this module's state_dict()
                   
        Raises:
            FileNotFoundError: If checkpoint_path does not exist
            RuntimeError: If checkpoint format is incompatible
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            # Check if it's a full training checkpoint or just state dict
            if 'encoder_state_dict' in checkpoint:
                state_dict = checkpoint['encoder_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                # Assume the checkpoint is the state dict itself
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove 'encoder.' prefix if present (from MAE checkpoint)
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('encoder.'):
                new_key = key[len('encoder.'):]
            else:
                new_key = key
            new_state_dict[new_key] = value
        
        # Load the state dict
        missing, unexpected = self.load_state_dict(new_state_dict, strict=strict)
        
        if missing:
            print(f"Warning: Missing keys when loading pretrained weights: {missing}")
        if unexpected:
            print(f"Warning: Unexpected keys when loading pretrained weights: {unexpected}")
