"""
Dataset Module for Drifting Field Generative Models

This module provides dataset classes for loading pre-cached latent representations,
enabling efficient training without VAE encoding during the training loop.

Key classes:
- LatentDataset: Load pre-cached latent files (batched or individual)
- LatentDatasetIndividual: Load individually saved latent files
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Union
import json

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class LatentDataset(Dataset):
    """
    Dataset for loading pre-cached VAE latent representations.
    
    This dataset loads latents that were pre-computed and saved by cache_latents.py,
    enabling efficient training without VAE encoding during the training loop.
    
    Supports two storage formats:
    1. Batched: Single latents.pt and labels.pt files (default)
    2. Individual: Separate .pt files for each sample
    
    Args:
        data_dir: Directory containing cached latents
        transform: Optional transform to apply to latents (e.g., augmentation)
        load_in_memory: Whether to load all latents into memory (default: True)
        subset_size: Optional limit on number of samples to use
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[callable] = None,
        load_in_memory: bool = True,
        subset_size: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.load_in_memory = load_in_memory
        
        # Check if metadata exists
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        # Determine storage format
        self.individual_storage = self.metadata.get('save_individually', False)
        
        if self.individual_storage:
            # Individual file storage
            self.sample_files = sorted(self.data_dir.glob("sample_*.pt"))
            self.num_samples = len(self.sample_files)
            self.latents = None
            self.labels = None
        else:
            # Batched storage
            latents_path = self.data_dir / "latents.pt"
            labels_path = self.data_dir / "labels.pt"
            
            if not latents_path.exists():
                raise FileNotFoundError(f"Latents file not found: {latents_path}")
            if not labels_path.exists():
                raise FileNotFoundError(f"Labels file not found: {labels_path}")
            
            if load_in_memory:
                self.latents = torch.load(latents_path, weights_only=True)
                self.labels = torch.load(labels_path, weights_only=True)
            else:
                # Memory-map for large datasets
                self.latents_path = latents_path
                self.labels_path = labels_path
                self.latents = None
                self.labels = None
                # Load labels for length calculation
                self._labels_cache = torch.load(labels_path, weights_only=True)
            
            self.num_samples = (
                len(self.latents) if self.latents is not None 
                else len(self._labels_cache)
            )
        
        # Apply subset if specified
        if subset_size is not None:
            self.num_samples = min(self.num_samples, subset_size)
        
        # Get latent shape from metadata or first sample
        self.latent_shape = self._get_latent_shape()
    
    def _get_latent_shape(self) -> Tuple[int, ...]:
        """Get the shape of a single latent tensor."""
        if 'latent_shape' in self.metadata:
            return tuple(self.metadata['latent_shape'])
        
        # Infer from data
        if self.individual_storage and self.sample_files:
            sample = torch.load(self.sample_files[0], weights_only=True)
            return tuple(sample['latent'].shape)
        elif self.latents is not None:
            return tuple(self.latents.shape[1:])
        else:
            # Load first sample to get shape
            latents = torch.load(self.latents_path, weights_only=True)
            return tuple(latents.shape[1:])
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single latent sample and its label.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (latent_tensor, class_label)
        """
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")
        
        if self.individual_storage:
            # Load from individual file
            sample = torch.load(self.sample_files[idx], weights_only=True)
            latent = sample['latent']
            label = sample['label']
        else:
            if self.latents is not None:
                # Direct access from memory
                latent = self.latents[idx]
                label = self.labels[idx].item()
            else:
                # Load from disk
                latents = torch.load(self.latents_path, weights_only=True)
                labels = torch.load(self.labels_path, weights_only=True)
                latent = latents[idx]
                label = labels[idx].item()
        
        # Apply transform if provided
        if self.transform is not None:
            latent = self.transform(latent)
        
        return latent, label
    
    @property
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        return self.metadata.get('num_classes', 1000)


class LatentDatasetIndividual(Dataset):
    """
    Dataset for loading individually saved latent files.
    
    This is optimized for very large datasets where loading all latents
    into memory is not feasible.
    
    Args:
        data_dir: Directory containing individual .pt files
        transform: Optional transform to apply to latents
        file_pattern: Glob pattern for sample files (default: "sample_*.pt")
    """
    
    def __init__(
        self,
        data_dir: str,
        transform: Optional[callable] = None,
        file_pattern: str = "sample_*.pt",
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Find all sample files
        self.sample_files = sorted(self.data_dir.glob(file_pattern))
        
        if not self.sample_files:
            raise FileNotFoundError(
                f"No sample files found in {data_dir} matching pattern {file_pattern}"
            )
        
        self.num_samples = len(self.sample_files)
        
        # Get latent shape from first sample
        first_sample = torch.load(self.sample_files[0], weights_only=True)
        self.latent_shape = tuple(first_sample['latent'].shape)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single latent sample and its label.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (latent_tensor, class_label)
        """
        sample = torch.load(self.sample_files[idx], weights_only=True)
        latent = sample['latent']
        label = sample['label']
        
        if self.transform is not None:
            latent = self.transform(latent)
        
        return latent, label


class DummyLatentDataset(Dataset):
    """
    Dummy latent dataset for testing and debugging.
    
    Generates random latent tensors without requiring pre-cached data.
    
    Args:
        num_samples: Number of samples
        num_classes: Number of classes
        latent_shape: Shape of latent tensors (default: (4, 32, 32))
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        num_classes: int = 1000,
        latent_shape: Tuple[int, ...] = (4, 32, 32),
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self._num_classes = num_classes
        self.latent_shape = latent_shape
        self.seed = seed
        
        # Generate fixed random labels
        rng = np.random.RandomState(seed)
        self.labels = rng.randint(0, num_classes, size=num_samples)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Generate a random latent tensor.
        
        Returns:
            Tuple of (latent_tensor, label)
        """
        # Use deterministic seed for reproducibility
        rng = np.random.RandomState(self.seed + idx)
        latent = rng.randn(*self.latent_shape).astype(np.float32)
        
        return torch.from_numpy(latent), int(self.labels[idx])
    
    @property
    def num_classes(self) -> int:
        """Return the number of classes."""
        return self._num_classes


def create_dataloader(
    data_dir: str,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for cached latent data.
    
    Args:
        data_dir: Directory containing cached latents
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        **kwargs: Additional arguments passed to LatentDataset
        
    Returns:
        Configured DataLoader
    """
    dataset = LatentDataset(data_dir, **kwargs)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


def create_dummy_dataloader(
    num_samples: int = 1000,
    num_classes: int = 1000,
    batch_size: int = 64,
    latent_shape: Tuple[int, ...] = (4, 32, 32),
    seed: int = 42,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader with dummy latent data for testing.
    
    Args:
        num_samples: Number of dummy samples
        num_classes: Number of classes
        batch_size: Batch size
        latent_shape: Shape of latent tensors
        seed: Random seed
        **kwargs: Additional arguments passed to DataLoader
        
    Returns:
        Configured DataLoader with dummy data
    """
    dataset = DummyLatentDataset(
        num_samples=num_samples,
        num_classes=num_classes,
        latent_shape=latent_shape,
        seed=seed,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=kwargs.pop('shuffle', True),
        num_workers=kwargs.pop('num_workers', 0),  # Use 0 for dummy data
        pin_memory=kwargs.pop('pin_memory', False),
        drop_last=kwargs.pop('drop_last', True),
        **kwargs,
    )
