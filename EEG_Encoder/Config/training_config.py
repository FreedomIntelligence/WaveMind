"""
Centralized Configuration for EEG CLIP Training

This module contains all configuration constants to avoid magic numbers
scattered throughout the codebase.
"""

import os
from dataclasses import dataclass, field
from typing import List

from data.Utils import get_wavemind_root


@dataclass
class EEGConfig:
    """EEG signal processing configuration"""

    # Default EEG parameters
    DEFAULT_SAMPLE_RATE: int = 512  # Hz
    DEFAULT_NUM_CHANNELS: int = 32
    SEGMENT_LENGTH_SEC: float = 1.0  # seconds
    SEGMENT_LENGTH_SAMPLES: int = 512  # DEFAULT_SAMPLE_RATE * SEGMENT_LENGTH_SEC

    # Model-specific sample rates (overrides for specific models)
    NEUROLM_SAMPLE_RATE: int = 200
    CBRAMOD_SAMPLE_RATE: int = 200
    THING_EEG_SAMPLE_RATE: int = 250

    # Channel configurations
    THING_EEG_CHANNELS: int = 63  # Before mapping
    STANDARD_CHANNELS: int = 32   # After mapping


@dataclass
class TrainingConfig:
    """Default training hyperparameters"""

    # Training parameters
    DEFAULT_BATCH_SIZE: int = 512
    DEFAULT_EPOCHS: int = 30
    DEFAULT_LEARNING_RATE: float = 1e-5
    DEFAULT_NUM_WORKERS: int = 16

    # Trainer configuration
    ACCUMULATE_GRAD_BATCHES: int = 2
    GRADIENT_CLIP_VAL: float = 1.0
    PRECISION: str = "32-true"
    LOG_EVERY_N_STEPS: int = 5
    CHECK_VAL_EVERY_N_EPOCH: int = 5
    NUM_SANITY_VAL_STEPS: int = 0

    # DDP settings
    DDP_TIMEOUT_MINUTES: int = 30
    BARRIER_TIMEOUT_MINUTES: int = 10
    CHECKPOINT_SAVE_TIMEOUT_MINUTES: int = 5
    CLEANUP_BARRIER_TIMEOUT_MINUTES: int = 1

    # Early stopping
    EARLY_STOP_PATIENCE: int = 10
    EARLY_STOP_MIN_DELTA: float = 0.0
    EARLY_STOP_MODE: str = 'min'


@dataclass
class MetricsConfig:
    """Metrics computation configuration"""

    # Top-k values for retrieval metrics
    THING_EEG_K_VALUES: List[int] = None  # [2, 4, 10, 50, 100, 200]
    IMAGENET_EEG_K_VALUES: List[int] = None  # [2, 4, 10, 40]

    # Classification thresholds
    CLASSIFICATION_MAX_CLASSES: int = 10  # Use classification metrics if num_classes <= this

    # Balanced accuracy settings
    USE_BALANCED_ACC_DEFAULT: bool = False

    def __post_init__(self):
        # Initialize mutable defaults
        if self.THING_EEG_K_VALUES is None:
            self.THING_EEG_K_VALUES = [2, 4, 10, 50, 100, 200]
        if self.IMAGENET_EEG_K_VALUES is None:
            self.IMAGENET_EEG_K_VALUES = [2, 4, 10, 40]


@dataclass
class PathConfig:
    """Default paths (can be overridden by environment variables or CLI args)"""

    # Root path (auto-detected via get_wavemind_root(), or via WaveMind_ROOT_PATH_ env var for backward compatibility)
    ROOT_PATH: str = field(default_factory=lambda: get_wavemind_root())

    # Data directories
    DATA_DIR: str = None
    HDF5_PATH: str = None
    CLIP_GROUND_TRUTH_DIR: str = None

    # Model directories
    CHECKPOINT_DIR: str = None

    # Shared memory
    SHM_DIR: str = '/dev/shm'
    SHM_AVAILABLE_BUFFER_RATIO: float = 0.8  # Reserve 20% safety buffer

    def __post_init__(self):
        # Initialize paths based on ROOT_PATH
        if self.DATA_DIR is None:
            self.DATA_DIR = os.path.join(self.ROOT_PATH, 'data/Total')
        if self.HDF5_PATH is None:
            self.HDF5_PATH = os.path.join(self.DATA_DIR, 'data_label.h5')
        if self.CLIP_GROUND_TRUTH_DIR is None:
            self.CLIP_GROUND_TRUTH_DIR = os.path.join(self.DATA_DIR, 'CLIP_groundTruth')
        if self.CHECKPOINT_DIR is None:
            self.CHECKPOINT_DIR = os.path.join(self.ROOT_PATH, 'EEG_Encoder/Resource/Checkpoint')


@dataclass
class TrainAugmentationConfig:
    """Train-time augmentation settings"""

    enable: bool = False  # Enable augmentation for training
    normalization_method: str = "random"  # random | zscore_global | zscore_channel | std_global | std_channel | identity
    zscore_epsilon: float = 1e-8  # Small value to prevent division by zero
    amplitude_fluctuation_enable: bool = True
    amplitude_fluctuation_range: tuple = (0.9, 1.1)  # 10% fluctuation


@dataclass
class TestAugmentationConfig:
    """Test/Val-time augmentation settings (deterministic normalization only)"""

    enable: bool = True  # Enable deterministic normalization
    normalization_method: str = "zscore_global"  # Deterministic global z-score only
    zscore_epsilon: float = 1e-8  # Small value to prevent division by zero


@dataclass
class AugmentationConfig:
    """Split-aware data augmentation configuration"""

    train: TrainAugmentationConfig = None
    test: TestAugmentationConfig = None

    def __post_init__(self):
        if self.train is None:
            self.train = TrainAugmentationConfig()
        if self.test is None:
            self.test = TestAugmentationConfig()


# Legacy configuration (deprecated, kept for backward compatibility)
@dataclass
class DataAugmentationConfig:
    """DEPRECATED: Legacy augmentation config for backward compatibility.
    Use AugmentationConfig instead."""

    # Noise generation parameters
    NOTCH_FILTER_FREQS: List[int] = None  # [50, 60] Hz - NOT USED
    ZSCORE_EPSILON: float = 1e-8  # Small value to prevent division by zero

    # Random fluctuation
    AMP_FLUCTUATION_RANGE: tuple = (0.9, 1.1)  # 10% fluctuation

    def __post_init__(self):
        if self.NOTCH_FILTER_FREQS is None:
            self.NOTCH_FILTER_FREQS = [50, 60]


@dataclass
class HDF5Config:
    """HDF5 file handling configuration"""

    # Cache settings for disk-based HDF5
    RDCC_NBYTES_DISK: int = 500 * 1024**2  # 500 MB
    RDCC_NSLOTS_DISK: int = 10000

    # Cache settings for SHM-based HDF5 (minimal cache)
    RDCC_NBYTES_SHM: int = 0
    RDCC_NSLOTS_SHM: int = 1

    # Common settings
    RDCC_W0: int = 0
    LIBVER: str = 'latest'
    SWMR_MODE: bool = True  # Single Writer Multiple Reader


# ============================================================================
# Global configuration instances (can be imported and used directly)
# ============================================================================

eeg_config = EEGConfig()
training_config = TrainingConfig()
metrics_config = MetricsConfig()
path_config = PathConfig()
augmentation_config = AugmentationConfig()  # New split-aware config
aug_config = DataAugmentationConfig()  # Legacy - kept for backward compatibility
hdf5_config = HDF5Config()


# ============================================================================
# Convenience functions
# ============================================================================

def get_model_sample_rate(model_name: str) -> int:
    """
    Get the appropriate sample rate for a given model

    Args:
        model_name: Name of the model

    Returns:
        Sample rate in Hz
    """
    model_name_lower = model_name.lower()

    if 'neurolm' in model_name_lower:
        return eeg_config.NEUROLM_SAMPLE_RATE
    elif 'cbramod' in model_name_lower:
        return eeg_config.CBRAMOD_SAMPLE_RATE
    else:
        return eeg_config.DEFAULT_SAMPLE_RATE


def get_k_values_for_dataset(dataset_name: str) -> List[int]:
    """
    Get appropriate k values for top-k retrieval metrics

    Args:
        dataset_name: Name of the dataset

    Returns:
        List of k values for top-k accuracy
    """
    dataset_name_lower = dataset_name.lower()

    if 'thing' in dataset_name_lower:
        return metrics_config.THING_EEG_K_VALUES
    elif 'imagenet' in dataset_name_lower:
        return metrics_config.IMAGENET_EEG_K_VALUES
    else:
        # Default to ImageNet k values for other datasets
        return metrics_config.IMAGENET_EEG_K_VALUES
