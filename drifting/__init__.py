"""
Generative Modeling via Drifting

A PyTorch implementation of the Drifting Field method for generative modeling.
This package provides tools for training generative models using the drifting field
equilibrium theory, which allows for one-step (1-NFE) inference.
"""

__version__ = "0.1.0"

from .models.drifting_dit import DriftingDiT
from .models.drifting_loss import DriftingLoss
from .models.feature_extractor import FeatureExtractor
from .utils.drifting_field import compute_V, compute_kernel
from .data.sample_queue import SampleQueue
from .data.dataset import LatentDataset, DummyLatentDataset, create_dataloader

__all__ = [
    "DriftingDiT",
    "DriftingLoss",
    "FeatureExtractor",
    "compute_V",
    "compute_kernel",
    "SampleQueue",
    "LatentDataset",
    "DummyLatentDataset",
    "create_dataloader",
]
