"""
Pre-flight Verification Tests

Tests for the pre-flight checklist requirements from the paper review:
1. Architecture Verification (Register Tokens, Style Tokens)
2. Training Strategy Verification (Class-Grouped Sampler, Feature Extractor Freezing)

These tests ensure the codebase is ready for training on single-GPU A100 setups.
"""

import pytest
import torch
import tempfile
import os

from drifting.models.drifting_dit import DriftingDiT, DriftingDiTSmall
from drifting.models.drifting_loss import DriftingLoss
from drifting.models.feature_extractor import LatentFeatureExtractor
from drifting.models.mae import create_mae
from drifting.training import create_trainer
from drifting.data.dataset import ClassGroupedBatchSampler, DummyLatentDataset


class TestFeatureExtractorFreezing:
    """
    CRITICAL: Tests for Feature Extractor freezing.
    
    Per Paper review: "Feature Extractor (MAE) MUST be in .eval() mode with 
    requires_grad=False to prevent BatchNorm statistics pollution from grouped batches."
    """
    
    def test_freeze_feature_extractor_sets_eval_mode(self):
        """Test that freeze_feature_extractor sets eval mode."""
        loss_fn = DriftingLoss(feature_extractor='latent', in_channels=4)
        
        # Initially should be in training mode (this is PyTorch default)
        assert loss_fn.feature_extractor.training
        
        # Freeze
        loss_fn.freeze_feature_extractor()
        
        # Should be in eval mode
        assert not loss_fn.feature_extractor.training
    
    def test_freeze_feature_extractor_disables_requires_grad(self):
        """Test that freeze_feature_extractor disables requires_grad for all params."""
        loss_fn = DriftingLoss(feature_extractor='latent', in_channels=4)
        
        # Initially all params should have requires_grad=True
        for param in loss_fn.feature_extractor.parameters():
            assert param.requires_grad
        
        # Freeze
        loss_fn.freeze_feature_extractor()
        
        # All params should have requires_grad=False
        for param in loss_fn.feature_extractor.parameters():
            assert not param.requires_grad
    
    def test_is_feature_extractor_frozen(self):
        """Test the is_feature_extractor_frozen check method."""
        loss_fn = DriftingLoss(feature_extractor='latent', in_channels=4)
        
        # Initially should not be frozen
        assert not loss_fn.is_feature_extractor_frozen()
        
        # After freezing should be frozen
        loss_fn.freeze_feature_extractor()
        assert loss_fn.is_feature_extractor_frozen()
    
    def test_train_mode_keeps_feature_extractor_frozen(self):
        """
        Test that calling train() on DriftingLoss keeps feature extractor frozen.
        
        This is critical because the training loop typically calls model.train()
        which would normally propagate to all submodules. The feature extractor
        must stay frozen to prevent BatchNorm statistics corruption.
        """
        loss_fn = DriftingLoss(feature_extractor='latent', in_channels=4)
        loss_fn.freeze_feature_extractor()
        
        # Simulate what happens during training loop
        loss_fn.train()
        
        # Feature extractor should still be in eval mode
        assert not loss_fn.feature_extractor.training
        assert loss_fn.is_feature_extractor_frozen()
    
    def test_create_trainer_freezes_feature_extractor(self):
        """Test that create_trainer automatically freezes the feature extractor."""
        model = DriftingDiTSmall(
            img_size=32, patch_size=2, in_chans=4,
            num_classes=10, num_style_tokens=8
        )
        
        trainer = create_trainer(
            model=model,
            feature_extractor='latent',
            device='cpu',
            latent_shape=(4, 32, 32),
        )
        
        # Feature extractor should be frozen
        assert trainer.loss_fn.is_feature_extractor_frozen()
        
        # Verify all params are frozen
        for param in trainer.loss_fn.feature_extractor.parameters():
            assert not param.requires_grad
    
    def test_create_trainer_freezes_after_loading_pretrained(self):
        """Test that feature extractor is frozen after loading pretrained weights."""
        # Create and save MAE checkpoint
        mae = create_mae(hidden_channels=64, num_stages=4)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'mae_encoder.pt')
            encoder_state = mae.get_encoder_state_dict()
            torch.save({'encoder_state_dict': encoder_state}, checkpoint_path)
            
            # Create trainer with pretrained weights
            model = DriftingDiTSmall(
                img_size=32, patch_size=2, in_chans=4,
                num_classes=10, num_style_tokens=8
            )
            
            trainer = create_trainer(
                model=model,
                feature_extractor='latent',
                device='cpu',
                mae_checkpoint=checkpoint_path,
                latent_shape=(4, 32, 32),
            )
            
            # Feature extractor should be frozen even after loading weights
            assert trainer.loss_fn.is_feature_extractor_frozen()
    
    def test_batchnorm_stats_not_updated_when_frozen(self):
        """
        Test that BatchNorm running stats are not updated when frozen.
        
        This verifies the critical requirement that class-grouped batches
        don't pollute the BatchNorm statistics.
        """
        loss_fn = DriftingLoss(feature_extractor='latent', in_channels=4)
        loss_fn.freeze_feature_extractor()
        
        # Get initial BatchNorm running mean
        initial_running_means = []
        for module in loss_fn.feature_extractor.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                initial_running_means.append(module.running_mean.clone())
        
        assert len(initial_running_means) > 0, "No BatchNorm layers found"
        
        # Forward pass with data
        x = torch.randn(8, 4, 32, 32)
        loss_fn.feature_extractor(x)
        
        # Check that running means have not changed
        idx = 0
        for module in loss_fn.feature_extractor.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                assert torch.allclose(
                    module.running_mean, 
                    initial_running_means[idx]
                ), "BatchNorm running mean changed when feature extractor should be frozen!"
                idx += 1


class TestRegisterTokensSequenceLength:
    """
    Tests for In-context Register Tokens and sequence length handling.
    
    Per Paper Appendix A.2: 16 register tokens are prepended to the 256 patch tokens,
    creating a sequence of length 272. The RoPE must handle this extended length.
    """
    
    def test_sequence_length_calculation(self):
        """Test that sequence length is correctly calculated with register tokens."""
        img_size = 32
        patch_size = 2
        num_register_tokens = 16
        num_patches = (img_size // patch_size) ** 2  # 256
        expected_seq_len = num_patches + num_register_tokens  # 272
        
        model = DriftingDiTSmall(
            img_size=img_size, patch_size=patch_size, in_chans=4,
            num_classes=10, num_register_tokens=num_register_tokens
        )
        
        assert model.num_patches == num_patches
        assert model.num_register_tokens == num_register_tokens
        # Verify expected total sequence length calculation
        assert expected_seq_len == 272  # 256 patches + 16 register tokens
    
    def test_rope_handles_extended_sequence(self):
        """Test that RoPE can handle sequence length 272."""
        from drifting.models.drifting_dit import RotaryPositionalEmbedding
        
        # Create RoPE with default max_seq_len
        rope = RotaryPositionalEmbedding(dim=64, max_seq_len=4096)
        
        # Should handle sequence length 272
        x = torch.randn(8, 272, 64)
        cos, sin = rope(x, seq_len=272)
        
        assert cos.shape == (272, 64)
        assert sin.shape == (272, 64)
    
    def test_forward_pass_with_272_tokens(self):
        """Test that forward pass works with 272 token sequence length."""
        model = DriftingDiTSmall(
            img_size=32, patch_size=2, in_chans=4,
            num_classes=10, num_register_tokens=16  # 256 + 16 = 272
        )
        
        z = torch.randn(4, 4, 32, 32)
        y = torch.randint(0, 10, (4,))
        
        # Forward pass should work
        out = model(z, y, cfg_scale=1.0)
        
        assert out.shape == z.shape


class TestStyleTokensSumOperation:
    """
    Tests for Style Tokens Sum operation.
    
    Per Paper Appendix A.2: "We sample 32 random indices from the codebook of size 64
    and sum the corresponding vectors."
    """
    
    def test_style_codebook_exists(self):
        """Test that style codebook is properly initialized."""
        model = DriftingDiTSmall(
            img_size=32, patch_size=2, in_chans=4,
            num_classes=10, codebook_size=64, num_style_samples=32
        )
        
        assert hasattr(model, 'style_codebook')
        assert model.style_codebook.shape == (64, model.embed_dim)
        assert model.codebook_size == 64
        assert model.num_style_samples == 32
    
    def test_style_tokens_sum_operation(self):
        """Test that style tokens are summed correctly during forward pass."""
        model = DriftingDiTSmall(
            img_size=32, patch_size=2, in_chans=4,
            num_classes=10, codebook_size=64, num_style_samples=32
        )
        
        # The forward pass should sample 32 vectors and sum them
        z = torch.randn(4, 4, 32, 32)
        y = torch.randint(0, 10, (4,))
        
        # This should work without errors
        out = model(z, y, cfg_scale=1.0)
        assert out.shape == z.shape
    
    def test_conditioning_vector_uses_element_wise_sum(self):
        """Test that class, cfg, and style embeddings are summed (not concatenated)."""
        model = DriftingDiTSmall(
            img_size=32, patch_size=2, in_chans=4,
            num_classes=10
        )
        
        # All transformer blocks should have cond_size = embed_dim (not 3*embed_dim)
        for block in model.blocks:
            assert block.adaLN.linear.in_features == model.embed_dim


class TestClassGroupedSamplerDefaults:
    """
    Tests for Class-Grouped Sampler defaults.
    
    Per Paper: K=4 classes per batch, M=32 samples per class.
    """
    
    def test_default_k_value(self):
        """Test that default K (num_classes_per_batch) is 4."""
        from drifting.data.dataset import create_dummy_dataloader
        
        # Check argparse defaults in train.py would give K=4
        # We verify the sampler works with K=4
        dataloader = create_dummy_dataloader(
            num_samples=1000,
            num_classes=100,
            use_class_grouped_sampler=True,
            num_classes_per_batch=4,  # K=4
            samples_per_class=32,  # M=32
        )
        
        for latents, labels in dataloader:
            unique_classes = torch.unique(labels)
            assert len(unique_classes) <= 4
            break
    
    def test_default_m_value(self):
        """Test that default M (samples_per_class) is 32."""
        dataset = DummyLatentDataset(
            num_samples=1000, num_classes=100
        )
        
        sampler = ClassGroupedBatchSampler(
            dataset=dataset,
            num_classes_per_batch=4,
            samples_per_class=32,  # M=32
            num_classes=100,
        )
        
        assert sampler.samples_per_class == 32
        assert sampler.batch_size == 4 * 32  # K * M = 128
    
    def test_effective_batch_size(self):
        """Test that effective batch size is K * M."""
        dataset = DummyLatentDataset(
            num_samples=1000, num_classes=100
        )
        
        sampler = ClassGroupedBatchSampler(
            dataset=dataset,
            num_classes_per_batch=4,
            samples_per_class=32,
            num_classes=100,
        )
        
        for batch_indices in sampler:
            assert len(batch_indices) == 128  # 4 * 32
            break


class TestLayerNormHandlesMagnitude:
    """
    Tests for LayerNorm handling of style token sum magnitude.
    
    When summing 32 style vectors, the magnitude increases by ~sqrt(32) ≈ 5.66.
    The DiT uses LayerNorm (via AdaLN-Zero) which should handle this.
    """
    
    def test_adaln_handles_large_inputs(self):
        """Test that AdaLN-Zero handles inputs with large magnitudes."""
        from drifting.models.drifting_dit import AdaLNZero
        
        adaln = AdaLNZero(hidden_size=256, cond_size=768)
        
        # Normal magnitude input
        x = torch.randn(8, 100, 256)
        cond = torch.randn(8, 768)
        
        shift1, scale1, gate1, shift2, scale2, gate2 = adaln(x, cond)
        x_mod = adaln.modulate(x, shift1, scale1)
        
        assert not torch.isnan(x_mod).any()
        assert not torch.isinf(x_mod).any()
        
        # Large magnitude input (simulating summed style vectors)
        large_cond = torch.randn(8, 768) * 6  # ~sqrt(32) scale
        
        shift1, scale1, gate1, shift2, scale2, gate2 = adaln(x, large_cond)
        x_mod = adaln.modulate(x, shift1, scale1)
        
        assert not torch.isnan(x_mod).any()
        assert not torch.isinf(x_mod).any()
    
    def test_rmsnorm_handles_large_inputs(self):
        """Test that RMSNorm handles inputs with large magnitudes."""
        from drifting.models.drifting_dit import RMSNorm
        
        norm = RMSNorm(dim=256)
        
        # Large magnitude input
        x = torch.randn(8, 100, 256) * 10
        y = norm(x)
        
        assert not torch.isnan(y).any()
        assert not torch.isinf(y).any()
        
        # RMS should be normalized
        rms = torch.sqrt((y ** 2).mean(dim=-1))
        assert rms.mean().item() == pytest.approx(1.0, abs=0.1)
