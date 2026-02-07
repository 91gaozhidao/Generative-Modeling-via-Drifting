"""
Tests for DriftingDiT model.
"""

import pytest
import torch

from drifting.models.drifting_dit import (
    DriftingDiT,
    DriftingDiTSmall,
    SwiGLU,
    RotaryPositionalEmbedding,
    QKNorm,
    AdaLNZero,
    StyleTokens,
    PatchEmbed,
    DriftingTransformerBlock,
    MultiHeadAttention,
    RMSNorm,
)


class TestRMSNorm:
    """Tests for RMSNorm (Paper Appendix A.2)."""
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        norm = RMSNorm(dim=256)
        x = torch.randn(32, 10, 256)
        y = norm(x)
        assert y.shape == x.shape
    
    def test_learnable_weight(self):
        """Test that weight is learnable."""
        norm = RMSNorm(dim=256)
        assert norm.weight.requires_grad
        assert norm.weight.shape == (256,)
    
    def test_normalization(self):
        """Test that output has unit RMS per sample."""
        norm = RMSNorm(dim=256)
        x = torch.randn(32, 10, 256) * 10  # Large input
        y = norm(x)
        # With default weight=1, RMS should be very close to 1.0
        rms = torch.sqrt((y ** 2).mean(dim=-1))
        assert rms.mean().item() == pytest.approx(1.0, abs=0.01)
    
    def test_gradient_flow(self):
        """Test that gradients flow through normalization."""
        norm = RMSNorm(dim=256)
        x = torch.randn(32, 10, 256, requires_grad=True)
        y = norm(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None
    
    def test_no_elementwise_affine(self):
        """Test RMSNorm without learnable weight (for AdaLN-Zero)."""
        norm = RMSNorm(dim=256, elementwise_affine=False)
        assert norm.weight is None
        x = torch.randn(32, 10, 256) * 10
        y = norm(x)
        assert y.shape == x.shape
        # RMS should still be approximately 1.0
        rms = torch.sqrt((y ** 2).mean(dim=-1))
        assert rms.mean().item() == pytest.approx(1.0, abs=0.01)


class TestSwiGLU:
    """Tests for SwiGLU activation."""
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        swiglu = SwiGLU(128, 512, 128)
        x = torch.randn(32, 10, 128)
        y = swiglu(x)
        assert y.shape == (32, 10, 128)
    
    def test_different_dims(self):
        """Test with different input/output dimensions."""
        swiglu = SwiGLU(64, 256, 128)
        x = torch.randn(32, 10, 64)
        y = swiglu(x)
        assert y.shape == (32, 10, 128)


class TestRotaryPositionalEmbedding:
    """Tests for RoPE."""
    
    def test_output_shape(self):
        """Test that cos/sin have correct shapes."""
        rope = RotaryPositionalEmbedding(dim=64)
        x = torch.randn(32, 100, 64)
        cos, sin = rope(x, seq_len=100)
        assert cos.shape == (100, 64)
        assert sin.shape == (100, 64)
    
    def test_values_bounded(self):
        """Test that cos/sin are bounded."""
        rope = RotaryPositionalEmbedding(dim=64)
        x = torch.randn(32, 100, 64)
        cos, sin = rope(x, seq_len=100)
        assert (cos >= -1).all() and (cos <= 1).all()
        assert (sin >= -1).all() and (sin <= 1).all()


class TestQKNorm:
    """Tests for QK-Norm."""
    
    def test_output_shapes(self):
        """Test that normalized q/k have correct shapes."""
        qk_norm = QKNorm(dim=64)
        q = torch.randn(32, 8, 100, 64)
        k = torch.randn(32, 8, 100, 64)
        q_norm, k_norm = qk_norm(q, k)
        assert q_norm.shape == q.shape
        assert k_norm.shape == k.shape


class TestAdaLNZero:
    """Tests for AdaLN-Zero."""
    
    def test_output_shape(self):
        """Test that modulation parameters have correct shape."""
        adaln = AdaLNZero(hidden_size=256, cond_size=768)
        x = torch.randn(32, 100, 256)
        cond = torch.randn(32, 768)
        
        shift1, scale1, gate1, shift2, scale2, gate2 = adaln(x, cond)
        
        assert shift1.shape == (32, 1, 256)
        assert scale1.shape == (32, 1, 256)
        assert gate1.shape == (32, 1, 256)
    
    def test_zero_initialization(self):
        """Test that linear layer is zero-initialized."""
        adaln = AdaLNZero(hidden_size=256, cond_size=768)
        assert torch.allclose(adaln.linear.weight, torch.zeros_like(adaln.linear.weight))
        assert torch.allclose(adaln.linear.bias, torch.zeros_like(adaln.linear.bias))
    
    def test_modulate_function(self):
        """Test the modulate function."""
        adaln = AdaLNZero(hidden_size=256, cond_size=768)
        x = torch.randn(32, 100, 256)
        shift = torch.zeros(32, 1, 256)
        scale = torch.zeros(32, 1, 256)
        
        x_mod = adaln.modulate(x, shift, scale)
        # With zero shift/scale, should be normalized x
        assert x_mod.shape == x.shape


class TestStyleTokens:
    """Tests for Style Tokens."""
    
    def test_forward_shape(self):
        """Test that forward returns correct shape."""
        style = StyleTokens(num_tokens=32, dim=768)
        tokens = style(batch_size=16, device=torch.device('cpu'))
        assert tokens.shape == (16, 768)
    
    def test_get_mixed_shape(self):
        """Test that get_mixed returns correct shape."""
        style = StyleTokens(num_tokens=32, dim=768)
        mixed = style.get_mixed(batch_size=16, device=torch.device('cpu'))
        assert mixed.shape == (16, 768)
    
    def test_learnable(self):
        """Test that tokens are learnable parameters."""
        style = StyleTokens(num_tokens=32, dim=768)
        assert style.tokens.requires_grad


class TestPatchEmbed:
    """Tests for Patch Embedding."""
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        patch_embed = PatchEmbed(img_size=32, patch_size=2, in_chans=4, embed_dim=768)
        x = torch.randn(16, 4, 32, 32)
        y = patch_embed(x)
        # num_patches = (32 / 2)^2 = 256
        assert y.shape == (16, 256, 768)
    
    def test_different_sizes(self):
        """Test with different image sizes."""
        patch_embed = PatchEmbed(img_size=64, patch_size=4, in_chans=3, embed_dim=512)
        x = torch.randn(8, 3, 64, 64)
        y = patch_embed(x)
        # num_patches = (64 / 4)^2 = 256
        assert y.shape == (8, 256, 512)


class TestMultiHeadAttention:
    """Tests for Multi-Head Attention."""
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        attn = MultiHeadAttention(dim=256, num_heads=8)
        x = torch.randn(16, 100, 256)
        y = attn(x)
        assert y.shape == x.shape
    
    def test_gradient_flow(self):
        """Test that gradients flow through attention."""
        attn = MultiHeadAttention(dim=256, num_heads=8)
        x = torch.randn(16, 100, 256, requires_grad=True)
        y = attn(x)
        loss = y.sum()
        loss.backward()
        assert x.grad is not None


class TestDriftingTransformerBlock:
    """Tests for Transformer Block."""
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        block = DriftingTransformerBlock(dim=256, num_heads=8, cond_size=768)
        x = torch.randn(16, 100, 256)
        cond = torch.randn(16, 768)
        y = block(x, cond)
        assert y.shape == x.shape
    
    def test_residual_connection(self):
        """Test that block has residual connections."""
        block = DriftingTransformerBlock(dim=256, num_heads=8, cond_size=768)
        x = torch.randn(16, 100, 256)
        cond = torch.zeros(16, 768)  # Zero conditioning
        y = block(x, cond)
        # With zero-initialized AdaLN, output should be close to input
        # (due to zero gates)
        assert y.shape == x.shape


class TestDriftingDiT:
    """Tests for the full DriftingDiT model."""
    
    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        model = DriftingDiTSmall(
            img_size=32, patch_size=2, in_chans=4,
            num_classes=10, num_style_tokens=8
        )
        z = torch.randn(4, 4, 32, 32)
        y = torch.randint(0, 10, (4,))
        out = model(z, y, cfg_scale=1.0)
        assert out.shape == z.shape
    
    def test_generate(self):
        """Test that generate produces correct output shape."""
        model = DriftingDiTSmall(
            img_size=32, patch_size=2, in_chans=4,
            num_classes=10, num_style_tokens=8
        )
        y = torch.randint(0, 10, (4,))
        out = model.generate(batch_size=4, y=y, cfg_scale=1.5, device='cpu')
        assert out.shape == (4, 4, 32, 32)
    
    def test_unconditional_class(self):
        """Test that unconditional class label works."""
        model = DriftingDiTSmall(
            img_size=32, patch_size=2, in_chans=4,
            num_classes=10, num_style_tokens=8
        )
        z = torch.randn(4, 4, 32, 32)
        y = torch.full((4,), 10)  # Unconditional class (num_classes)
        out = model(z, y, cfg_scale=1.0)
        assert out.shape == z.shape
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = DriftingDiTSmall(
            img_size=32, patch_size=2, in_chans=4,
            num_classes=10, num_style_tokens=8
        )
        z = torch.randn(4, 4, 32, 32, requires_grad=True)
        y = torch.randint(0, 10, (4,))
        out = model(z, y, cfg_scale=1.0)
        loss = out.sum()
        loss.backward()
        assert z.grad is not None
    
    def test_cfg_scale_conditioning(self):
        """Test that CFG scale is properly embedded and affects the model."""
        model = DriftingDiTSmall(
            img_size=32, patch_size=2, in_chans=4,
            num_classes=10, num_style_tokens=8
        )
        
        # Check that cfg_embed produces different outputs for different scales
        cfg1 = torch.tensor([[1.0]])
        cfg2 = torch.tensor([[3.0]])
        
        emb1 = model.cfg_embed(cfg1)
        emb2 = model.cfg_embed(cfg2)
        
        # Different CFG scales should produce different embeddings
        assert not torch.allclose(emb1, emb2, atol=1e-5)
        
        # Check embedding shape
        assert emb1.shape == (1, model.embed_dim)
    
    def test_model_variants(self):
        """Test that different model variants have correct configurations."""
        from drifting.models.drifting_dit import DriftingDiTBase, DriftingDiTLarge
        
        small = DriftingDiTSmall(img_size=16, patch_size=2, num_classes=10)
        base = DriftingDiTBase(img_size=16, patch_size=2, num_classes=10)
        large = DriftingDiTLarge(img_size=16, patch_size=2, num_classes=10)
        
        assert small.embed_dim < base.embed_dim < large.embed_dim
        assert len(small.blocks) < len(base.blocks) < len(large.blocks)
    
    def test_dit_large_config(self):
        """Test DiT-L/2 configuration matches paper specification (Deng et al., 2026)."""
        from drifting.models.drifting_dit import DriftingDiTLarge
        
        model = DriftingDiTLarge(img_size=32, patch_size=2, num_classes=1000)
        
        # Core architecture (DiT-L/2)
        assert model.embed_dim == 1024, "Hidden size must be 1024"
        assert len(model.blocks) == 24, "Depth must be 24 transformer blocks"
        assert model.blocks[0].attn.num_heads == 16, "Must have 16 attention heads"
        assert model.patch_size == 2, "Patch size must be 2"
        
        # In-context Register Tokens (Paper A.2): 16 learnable tokens
        assert model.num_register_tokens == 16, "Must have 16 register tokens"
        assert model.register_tokens.shape == (1, 16, 1024)
        
        # Style Tokens Codebook (Paper A.2): 64-entry codebook, sample 32, sum
        assert model.codebook_size == 64, "Codebook size must be 64"
        assert model.num_style_samples == 32, "Must sample 32 style tokens"
        assert model.style_codebook.shape == (64, 1024)
        
        # Sequence length: 16 register tokens + 256 patch tokens = 272
        num_patches = (32 // 2) ** 2  # 256
        expected_seq_len = 16 + num_patches  # 272
        assert model.num_patches == num_patches
        assert model.num_register_tokens + model.num_patches == expected_seq_len
        
        # Verify forward pass produces correct output
        z = torch.randn(2, 4, 32, 32)
        y = torch.randint(0, 1000, (2,))
        out = model(z, y, cfg_scale=1.5)
        assert out.shape == (2, 4, 32, 32)
    
    def test_element_wise_sum_conditioning(self):
        """Test that conditioning uses element-wise sum (Paper Appendix A.2)."""
        model = DriftingDiTSmall(
            img_size=32, patch_size=2, in_chans=4,
            num_classes=10, num_style_tokens=8
        )
        
        # Verify transformer blocks use embed_dim (not 3*embed_dim) for cond_size
        # This confirms element-wise sum is being used
        for block in model.blocks:
            # AdaLN linear layer should expect embed_dim, not 3*embed_dim
            expected_in_features = model.embed_dim
            actual_in_features = block.adaLN.linear.in_features
            assert actual_in_features == expected_in_features, (
                f"Expected cond_size={expected_in_features} (element-wise sum), "
                f"got {actual_in_features}. Paper specifies summing embeddings."
            )
    
    def test_register_tokens_exist(self):
        """Test that register tokens are properly initialized (Paper Appendix A.2)."""
        model = DriftingDiTSmall(
            img_size=32, patch_size=2, in_chans=4,
            num_classes=10, num_register_tokens=16
        )
        
        # Check register tokens parameter exists with correct shape
        assert hasattr(model, 'register_tokens')
        assert model.register_tokens.shape == (1, 16, model.embed_dim)
        assert model.register_tokens.requires_grad
        
    def test_register_proj_exists(self):
        """Test that register projection layer exists."""
        model = DriftingDiTSmall(
            img_size=32, patch_size=2, in_chans=4,
            num_classes=10, num_register_tokens=16
        )
        
        # Check register projection layer exists
        assert hasattr(model, 'register_proj')
        assert model.register_proj.in_features == model.embed_dim
        assert model.register_proj.out_features == 16 * model.embed_dim
        
    def test_style_codebook_exists(self):
        """Test that style codebook is properly initialized (Paper Appendix A.2)."""
        model = DriftingDiTSmall(
            img_size=32, patch_size=2, in_chans=4,
            num_classes=10, codebook_size=64, num_style_samples=32
        )
        
        # Check style codebook parameter exists with correct shape
        assert hasattr(model, 'style_codebook')
        assert model.style_codebook.shape == (64, model.embed_dim)
        assert model.style_codebook.requires_grad
        assert model.num_style_samples == 32
        
    def test_sequence_length_with_registers(self):
        """Test that forward pass handles sequence length correctly with register tokens."""
        num_register_tokens = 16
        img_size = 32
        patch_size = 2
        num_patches = (img_size // patch_size) ** 2  # 256
        
        model = DriftingDiTSmall(
            img_size=img_size, patch_size=patch_size, in_chans=4,
            num_classes=10, num_register_tokens=num_register_tokens
        )
        
        z = torch.randn(4, 4, img_size, img_size)
        y = torch.randint(0, 10, (4,))
        out = model(z, y, cfg_scale=1.0)
        
        # Output should have the same shape as input (registers are removed in unpatchify)
        assert out.shape == z.shape


class TestDriftingDiTMemory:
    """Memory-related tests for DriftingDiT."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        model = DriftingDiTSmall(
            img_size=32, patch_size=2, in_chans=4,
            num_classes=10, num_style_tokens=8
        ).cuda()
        
        z = torch.randn(4, 4, 32, 32).cuda()
        y = torch.randint(0, 10, (4,)).cuda()
        out = model(z, y, cfg_scale=1.0)
        
        assert out.device.type == 'cuda'
        assert out.shape == z.shape
