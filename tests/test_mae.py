"""
Tests for MAE (Masked Autoencoder) module.

Tests the MAE pretraining pipeline for the feature extractor as described
in the paper's Appendix A.3.
"""

import pytest
import torch
import tempfile
import os

from drifting.models.mae import (
    LatentMAE,
    LatentMAEEncoder,
    LatentMAEDecoder,
    create_mae,
)
from drifting.models.feature_extractor import LatentFeatureExtractor


class TestLatentMAEEncoder:
    """Tests for LatentMAEEncoder."""
    
    def test_output_shapes(self):
        """Test that encoder produces correct output shapes with intermediate extraction."""
        encoder = LatentMAEEncoder(
            in_channels=4,
            hidden_channels=64,
            num_stages=4,
            patch_size=4,
        )
        
        x = torch.randn(8, 4, 32, 32)
        features, pooled = encoder(x)
        
        # With blocks_per_stage=[3,4,6,3] and extract_every_n=2:
        # Stage 0 (3 blocks): block2, block3 -> 2 features
        # Stage 1 (4 blocks): block2, block4 -> 2 features
        # Stage 2 (6 blocks): block2, block4, block6 -> 3 features
        # Stage 3 (3 blocks): block2, block3 -> 2 features
        # Total: 9 features
        assert len(features) == 9
        
        # All stage 0 features are 64ch, 32x32
        assert features[0].shape == (8, 64, 32, 32)
        assert features[1].shape == (8, 64, 32, 32)
        # Stage 1 features: 128ch, 16x16
        assert features[2].shape == (8, 128, 16, 16)
        assert features[3].shape == (8, 128, 16, 16)
        # Stage 2 features: 256ch, 8x8
        assert features[4].shape == (8, 256, 8, 8)
        assert features[5].shape == (8, 256, 8, 8)
        assert features[6].shape == (8, 256, 8, 8)
        # Stage 3 features: 512ch, 4x4
        assert features[7].shape == (8, 512, 4, 4)
        assert features[8].shape == (8, 512, 4, 4)
        
        assert pooled.shape == (8, 512)
    
    def test_forward_features(self):
        """Test forward_features compatibility method."""
        encoder = LatentMAEEncoder(
            in_channels=4,
            hidden_channels=64,
            num_stages=4,
        )
        
        x = torch.randn(4, 4, 32, 32)
        features = encoder.forward_features(x)
        
        assert len(features) == 9  # Intermediate extraction
        assert all(isinstance(f, torch.Tensor) for f in features)
    
    def test_feature_dims(self):
        """Test that feature_dims is correctly computed with intermediates."""
        encoder = LatentMAEEncoder(
            in_channels=4,
            hidden_channels=64,
            num_stages=4,
        )
        
        # With extract_every_n=2 and blocks=[3,4,6,3]
        assert encoder.feature_dims == [64, 64, 128, 128, 256, 256, 256, 512, 512]
        assert encoder.final_dim == 512


class TestLatentMAEDecoder:
    """Tests for LatentMAEDecoder."""
    
    def test_output_shape(self):
        """Test that decoder produces correct output shape."""
        decoder = LatentMAEDecoder(
            in_channels=4,
            encoder_dim=512,
            hidden_channels=256,
            output_size=32,
        )
        
        features = torch.randn(8, 512)
        output = decoder(features)
        
        assert output.shape == (8, 4, 32, 32)
    
    def test_gradient_flow(self):
        """Test that gradients flow through decoder."""
        decoder = LatentMAEDecoder(
            in_channels=4,
            encoder_dim=512,
            hidden_channels=256,
            output_size=32,
        )
        
        features = torch.randn(4, 512, requires_grad=True)
        output = decoder(features)
        loss = output.sum()
        loss.backward()
        
        assert features.grad is not None


class TestLatentMAE:
    """Tests for LatentMAE."""
    
    def test_creation(self):
        """Test MAE creation."""
        mae = create_mae(
            in_channels=4,
            hidden_channels=64,
            num_stages=4,
            patch_size=4,
            mask_ratio=0.75,
            input_size=32,
        )
        
        assert isinstance(mae, LatentMAE)
        assert mae.num_patches == 64  # (32/4)^2
        assert mae.mask_ratio == 0.75
    
    def test_forward_pass(self):
        """Test forward pass returns loss and mask."""
        mae = create_mae()
        
        x = torch.randn(4, 4, 32, 32)
        loss, mask = mae(x)
        
        assert loss.dim() == 0  # Scalar loss
        assert mask.shape == (4, 64)  # (batch_size, num_patches)
        assert mask.dtype == torch.bool
    
    def test_mask_ratio(self):
        """Test that masking ratio is approximately correct."""
        mae = create_mae(mask_ratio=0.75)
        
        x = torch.randn(16, 4, 32, 32)
        _, mask = mae(x)
        
        actual_ratio = mask.float().mean().item()
        assert abs(actual_ratio - 0.75) < 0.05
    
    def test_return_reconstruction(self):
        """Test returning reconstruction."""
        mae = create_mae()
        
        x = torch.randn(4, 4, 32, 32)
        loss, mask, recon = mae(x, return_reconstruction=True)
        
        assert recon.shape == x.shape
    
    def test_reconstruct_method(self):
        """Test reconstruct method (no masking)."""
        mae = create_mae()
        
        x = torch.randn(4, 4, 32, 32)
        recon = mae.reconstruct(x)
        
        assert recon.shape == x.shape
    
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        mae = create_mae()
        
        x = torch.randn(4, 4, 32, 32, requires_grad=True)
        loss, _ = mae(x)
        loss.backward()
        
        assert x.grad is not None
    
    def test_get_encoder_state_dict(self):
        """Test getting encoder state dict."""
        mae = create_mae(hidden_channels=64, num_stages=4)
        
        encoder_state = mae.get_encoder_state_dict()
        
        assert isinstance(encoder_state, dict)
        assert len(encoder_state) > 0
        # Should contain stage weights
        assert any('stages' in k for k in encoder_state.keys())
    
    def test_patchify_unpatchify(self):
        """Test that patchify and unpatchify are inverse operations."""
        mae = create_mae(patch_size=4, input_size=32)
        
        x = torch.randn(4, 4, 32, 32)
        patches = mae.patchify(x)
        reconstructed = mae.unpatchify(patches)
        
        assert torch.allclose(x, reconstructed, atol=1e-6)
    
    def test_apply_mask(self):
        """Test mask application."""
        mae = create_mae(patch_size=4, input_size=32)
        
        x = torch.randn(4, 4, 32, 32)
        mask = mae.generate_mask(4, x.device)
        x_masked = mae.apply_mask(x, mask)
        
        assert x_masked.shape == x.shape


class TestLoadPretrained:
    """Tests for load_pretrained functionality."""
    
    def test_load_pretrained_basic(self):
        """Test basic pretrained weight loading."""
        mae = create_mae(hidden_channels=64, num_stages=4)
        
        feature_extractor = LatentFeatureExtractor(
            in_channels=4,
            hidden_channels=64,
            num_stages=4,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'encoder.pt')
            encoder_state = mae.get_encoder_state_dict()
            torch.save({'encoder_state_dict': encoder_state}, checkpoint_path)
            
            # Should not raise
            feature_extractor.load_pretrained(checkpoint_path)
    
    def test_load_pretrained_changes_weights(self):
        """Test that loading pretrained actually changes weights."""
        mae = create_mae(hidden_channels=64, num_stages=4)
        
        # Get MAE encoder weights
        mae_state = mae.get_encoder_state_dict()
        
        # Create feature extractor with different weights
        feature_extractor = LatentFeatureExtractor(
            in_channels=4,
            hidden_channels=64,
            num_stages=4,
        )
        original_state = {k: v.clone() for k, v in feature_extractor.state_dict().items()}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, 'encoder.pt')
            torch.save({'encoder_state_dict': mae_state}, checkpoint_path)
            
            feature_extractor.load_pretrained(checkpoint_path)
            
            # At least some weights should have changed
            loaded_state = feature_extractor.state_dict()
            num_changed = sum(
                1 for k in original_state 
                if not torch.allclose(original_state[k], loaded_state[k])
            )
            assert num_changed > 0
    
    def test_load_pretrained_file_not_found(self):
        """Test that FileNotFoundError is raised for missing file."""
        feature_extractor = LatentFeatureExtractor(
            in_channels=4,
            hidden_channels=64,
            num_stages=4,
        )
        
        with pytest.raises(FileNotFoundError):
            feature_extractor.load_pretrained('/nonexistent/path/encoder.pt')
    
    def test_load_pretrained_alternative_formats(self):
        """Test loading from different checkpoint formats."""
        mae = create_mae(hidden_channels=64, num_stages=4)
        encoder_state = mae.get_encoder_state_dict()
        
        feature_extractor = LatentFeatureExtractor(
            in_channels=4,
            hidden_channels=64,
            num_stages=4,
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test 'state_dict' key
            path1 = os.path.join(tmpdir, 'format1.pt')
            torch.save({'state_dict': encoder_state}, path1)
            feature_extractor.load_pretrained(path1)
            
            # Test 'model' key
            path2 = os.path.join(tmpdir, 'format2.pt')
            torch.save({'model': encoder_state}, path2)
            feature_extractor.load_pretrained(path2)
            
            # Test direct state dict
            path3 = os.path.join(tmpdir, 'format3.pt')
            torch.save(encoder_state, path3)
            feature_extractor.load_pretrained(path3)


class TestMAEIntegration:
    """Integration tests for MAE pretraining pipeline."""
    
    def test_training_simulation(self):
        """Simulate a few training steps."""
        mae = create_mae()
        optimizer = torch.optim.Adam(mae.parameters(), lr=1e-4)
        
        initial_loss = None
        
        for i in range(5):
            x = torch.randn(8, 4, 32, 32)
            loss, _ = mae(x)
            
            if initial_loss is None:
                initial_loss = loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Loss should be finite
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    def test_end_to_end_pretrain_and_load(self):
        """Test full pipeline: train MAE, save encoder, load into feature extractor."""
        # Create and "train" MAE
        mae = create_mae(hidden_channels=64, num_stages=4)
        optimizer = torch.optim.Adam(mae.parameters(), lr=1e-4)
        
        for _ in range(3):
            x = torch.randn(4, 4, 32, 32)
            loss, _ = mae(x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save encoder
            checkpoint_path = os.path.join(tmpdir, 'encoder.pt')
            encoder_state = mae.get_encoder_state_dict()
            torch.save({'encoder_state_dict': encoder_state}, checkpoint_path)
            
            # Load into feature extractor
            feature_extractor = LatentFeatureExtractor(
                in_channels=4,
                hidden_channels=64,
                num_stages=4,
            )
            feature_extractor.load_pretrained(checkpoint_path)
            
            # Use feature extractor
            x = torch.randn(4, 4, 32, 32)
            features = feature_extractor(x)
            
            # With intermediate extraction, we get 9 features
            assert len(features) == 9
            assert all(not torch.isnan(f).any() for f in features)
