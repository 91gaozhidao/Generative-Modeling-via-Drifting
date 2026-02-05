"""
DriftingDiT: Diffusion Transformer for Drifting Field Generative Modeling

This module implements the DiT-style transformer generator G(z, y, gamma) with
specific architectural modifications from the paper:
- SwiGLU activations
- RoPE (Rotary Positional Embeddings)
- QK-Norm for training stability
- AdaLN-zero for conditioning (class labels + CFG scale)
- 32 learnable "style tokens" for handling randomness beyond Gaussian noise

The model operates on latent space (32x32x4 for ImageNet with a VAE encoder).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange, repeat


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    
    As specified in Paper Appendix A.2, RMSNorm is used instead of LayerNorm
    for better training stability in deep transformers.
    
    Reference: Zhang & Sennrich, 2019 "Root Mean Square Layer Normalization"
    
    Args:
        dim: Dimension of the input features
        eps: Small epsilon for numerical stability
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Compute RMS normalization."""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization with learnable scale."""
        return self._norm(x.float()).type_as(x) * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU activation function.
    SwiGLU(x, W, V, b, c) = Swish(xW + b) ⊗ (xV + c)
    """
    
    def __init__(self, in_features: int, hidden_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.w1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE) for 2D spatial positions.
    
    RoPE encodes relative position information directly into the attention
    mechanism by rotating query and key vectors.
    """
    
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequency bands
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Build cos/sin cache
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Build the cos/sin cache for the given sequence length."""
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return cos and sin for the given sequence length."""
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
        return (
            self.cos_cached[:seq_len],
            self.sin_cached[:seq_len]
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embedding to query and key tensors."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class QKNorm(nn.Module):
    """
    QK-Norm: RMSNorm applied to queries and keys for training stability.
    
    This helps prevent attention entropy collapse during training.
    Uses RMSNorm per Paper Appendix A.2.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.q_norm = RMSNorm(dim, eps=eps)
        self.k_norm = RMSNorm(dim, eps=eps)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.q_norm(q), self.k_norm(k)


class AdaLNZero(nn.Module):
    """
    Adaptive Layer Normalization with Zero-initialization (AdaLN-Zero).
    
    Used for conditioning on class labels and CFG scale.
    Outputs: (shift, scale, gate) for modulating layer outputs.
    Uses RMSNorm per Paper Appendix A.2.
    """
    
    def __init__(self, hidden_size: int, cond_size: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = nn.Linear(cond_size, 6 * hidden_size)
        
        # Zero-initialize the linear layer
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (B, N, D)
            cond: Conditioning tensor of shape (B, cond_size)
            
        Returns:
            Tuple of (shift1, scale1, gate1, shift2, scale2, gate2) for attention and FFN
        """
        params = self.linear(cond)  # (B, 6*D)
        params = params.unsqueeze(1)  # (B, 1, 6*D)
        shift1, scale1, gate1, shift2, scale2, gate2 = params.chunk(6, dim=-1)
        return shift1, scale1, gate1, shift2, scale2, gate2
    
    def modulate(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Apply modulation: x = x * (1 + scale) + shift"""
        return self.norm(x) * (1 + scale) + shift


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention with RoPE and QK-Norm.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # QK-Norm for training stability
        self.qk_norm = QKNorm(self.head_dim)
        
        # RoPE
        self.rope = RotaryPositionalEmbedding(self.head_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply QK-Norm
        q, k = self.qk_norm(q, k)
        
        # Apply RoPE
        cos, sin = self.rope(x, N)
        cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, N, D)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.proj(out)
        
        return out


class DriftingTransformerBlock(nn.Module):
    """
    Transformer block with AdaLN-Zero conditioning, SwiGLU, RoPE, and QK-Norm.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        cond_size: int = 768,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        # AdaLN-Zero for conditioning
        self.adaLN = AdaLNZero(dim, cond_size)
        
        # Self-Attention with RoPE and QK-Norm
        self.attn = MultiHeadAttention(dim, num_heads, dropout=dropout)
        
        # SwiGLU FFN
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, mlp_hidden_dim, dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, D)
            cond: Conditioning tensor of shape (B, cond_size)
            
        Returns:
            Output tensor of shape (B, N, D)
        """
        # Get modulation parameters
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(x, cond)
        
        # Attention block with AdaLN-Zero
        x_mod = self.adaLN.modulate(x, shift1, scale1)
        x = x + gate1 * self.dropout(self.attn(x_mod))
        
        # FFN block with AdaLN-Zero
        x_mod = self.adaLN.modulate(x, shift2, scale2)
        x = x + gate2 * self.dropout(self.mlp(x_mod))
        
        return x


class StyleTokens(nn.Module):
    """
    Learnable style tokens for handling randomness beyond Gaussian noise.
    
    32 learnable tokens are added to the conditioning vector to capture
    different generation modes/styles.
    """
    
    def __init__(self, num_tokens: int = 32, dim: int = 768):
        super().__init__()
        self.num_tokens = num_tokens
        self.tokens = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
    
    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample style tokens for each item in the batch.
        
        Args:
            batch_size: Number of samples in the batch
            device: Device to create tensor on
            
        Returns:
            Sampled style vectors of shape (B, dim)
        """
        # Randomly select one style token per sample
        indices = torch.randint(0, self.num_tokens, (batch_size,), device=device)
        return self.tokens[indices]
    
    def get_mixed(self, batch_size: int, device: torch.device, temperature: float = 1.0) -> torch.Tensor:
        """
        Get a weighted mixture of style tokens.
        
        Args:
            batch_size: Number of samples
            device: Device
            temperature: Temperature for softmax weighting
            
        Returns:
            Mixed style vectors of shape (B, dim)
        """
        # Sample random weights
        weights = torch.randn(batch_size, self.num_tokens, device=device) / temperature
        weights = F.softmax(weights, dim=-1)
        
        # Weighted sum of style tokens
        mixed = torch.einsum('bn,nd->bd', weights, self.tokens)
        return mixed


class PatchEmbed(nn.Module):
    """
    Patch embedding layer for converting latent images to tokens.
    """
    
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        in_chans: int = 4,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Patch embeddings of shape (B, N, D)
        """
        x = self.proj(x)  # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class DriftingDiT(nn.Module):
    """
    DriftingDiT: Diffusion Transformer for Drifting Field Generative Modeling.
    
    This is the main generator G(z, y, gamma) that maps:
    - z: Gaussian noise
    - y: Class label
    - gamma: CFG scale (conditioning strength)
    
    to generated latent images.
    
    Architecture:
    - Patch embedding for latent images
    - DiT transformer blocks with AdaLN-Zero conditioning
    - SwiGLU activations, RoPE, QK-Norm
    - Style tokens for additional randomness
    
    Args:
        img_size: Size of the latent image (default: 32 for ImageNet VAE)
        patch_size: Size of each patch (default: 2)
        in_chans: Number of input channels (default: 4 for VAE latent)
        num_classes: Number of classes for conditioning (default: 1000 for ImageNet)
        embed_dim: Embedding dimension (default: 768)
        depth: Number of transformer blocks (default: 12)
        num_heads: Number of attention heads (default: 12)
        mlp_ratio: MLP hidden dimension ratio (default: 4.0)
        num_style_tokens: Number of learnable style tokens (default: 32)
        dropout: Dropout rate (default: 0.0)
    """
    
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        in_chans: int = 4,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        num_style_tokens: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes + 1, embed_dim)  # +1 for unconditional
        
        # CFG scale embedding (continuous)
        self.cfg_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        
        # Style tokens
        self.style_tokens = StyleTokens(num_style_tokens, embed_dim)
        
        # Conditioning size: Paper Appendix A.2 specifies element-wise sum
        # "These are summed and added to the conditioning vector"
        # This introduces negligible overhead compared to concatenation
        cond_size = embed_dim
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            DriftingTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                cond_size=cond_size,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        
        # Final layer
        self.final_norm = RMSNorm(embed_dim)
        self.final_proj = nn.Linear(embed_dim, patch_size ** 2 * in_chans)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize patch embedding
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.size(0), -1))
        
        # Initialize class embedding
        nn.init.normal_(self.class_embed.weight, std=0.02)
        
        # Initialize final projection to zero (for stable training start)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patch predictions back to image.
        
        Args:
            x: Patch predictions of shape (B, N, P*P*C)
            
        Returns:
            Image tensor of shape (B, C, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        c = self.in_chans
        
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, p, w, p)
        x = x.reshape(x.shape[0], c, h * p, w * p)
        
        return x
    
    def forward(
        self,
        z: torch.Tensor,
        y: torch.Tensor,
        cfg_scale: float = 1.0,
        use_style_tokens: bool = True,
    ) -> torch.Tensor:
        """
        Forward pass of the generator.
        
        Args:
            z: Noise tensor of shape (B, C, H, W) in latent space
            y: Class labels of shape (B,) with values in [0, num_classes-1]
               Use num_classes for unconditional generation
            cfg_scale: CFG scale (conditioning strength), typically in [1.0, 7.5]
            use_style_tokens: Whether to use style tokens
            
        Returns:
            Generated latent images of shape (B, C, H, W)
        """
        B = z.shape[0]
        device = z.device
        
        # Patch embed the noise
        x = self.patch_embed(z)  # (B, N, D)
        
        # Get class embedding
        class_emb = self.class_embed(y)  # (B, D)
        
        # Get CFG scale embedding
        cfg_tensor = torch.full((B, 1), cfg_scale, device=device, dtype=z.dtype)
        cfg_emb = self.cfg_embed(cfg_tensor)  # (B, D)
        
        # Get style token embedding
        if use_style_tokens:
            style_emb = self.style_tokens.get_mixed(B, device)  # (B, D)
        else:
            style_emb = torch.zeros(B, self.embed_dim, device=device, dtype=z.dtype)
        
        # Element-wise sum for conditioning (Paper Appendix A.2)
        # "These are summed and added to the conditioning vector... 
        # introduces negligible overhead"
        cond = class_emb + cfg_emb + style_emb  # (B, D)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, cond)
        
        # Final projection
        x = self.final_norm(x)
        x = self.final_proj(x)  # (B, N, P*P*C)
        
        # Unpatchify
        out = self.unpatchify(x)  # (B, C, H, W)
        
        return out
    
    def generate(
        self,
        batch_size: int,
        y: torch.Tensor,
        cfg_scale: float = 1.0,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Generate samples from random noise.
        
        This is the one-step (1-NFE) inference: just one forward pass!
        
        Args:
            batch_size: Number of samples to generate
            y: Class labels of shape (batch_size,)
            cfg_scale: CFG scale for conditioning strength
            device: Device to generate on
            
        Returns:
            Generated latent images of shape (B, C, H, W)
        """
        # Sample noise
        z = torch.randn(
            batch_size, self.in_chans, self.img_size, self.img_size,
            device=device
        )
        
        # Single forward pass
        return self.forward(z, y, cfg_scale)


class DriftingDiTSmall(DriftingDiT):
    """Small version of DriftingDiT for faster experimentation."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('embed_dim', 384)
        kwargs.setdefault('depth', 6)
        kwargs.setdefault('num_heads', 6)
        super().__init__(**kwargs)


class DriftingDiTBase(DriftingDiT):
    """Base version of DriftingDiT."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('embed_dim', 768)
        kwargs.setdefault('depth', 12)
        kwargs.setdefault('num_heads', 12)
        super().__init__(**kwargs)


class DriftingDiTLarge(DriftingDiT):
    """Large version of DriftingDiT."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('embed_dim', 1024)
        kwargs.setdefault('depth', 24)
        kwargs.setdefault('num_heads', 16)
        super().__init__(**kwargs)


class DriftingDiTXL(DriftingDiT):
    """XL version of DriftingDiT."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('embed_dim', 1152)
        kwargs.setdefault('depth', 28)
        kwargs.setdefault('num_heads', 16)
        super().__init__(**kwargs)
