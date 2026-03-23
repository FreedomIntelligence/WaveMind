"""
Hydra Structured Configurations for WaveMind Training

This module wraps existing dataclasses with Hydra structured config decorators
to enable YAML-based configuration while maintaining type safety and backward compatibility.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from hydra.core.config_store import ConfigStore

from data.Utils import get_wavemind_root

# Import existing configs as base
from .training_config import (
    EEGConfig as BaseEEGConfig,
    TrainingConfig as BaseTrainingConfig,
    MetricsConfig as BaseMetricsConfig,
    DataAugmentationConfig as BaseDataAugmentationConfig,
    HDF5Config as BaseHDF5Config,
)


@dataclass
class EEGConfig(BaseEEGConfig):
    """EEG signal processing configuration (Hydra-enabled)"""
    pass


@dataclass
class TrainingConfig(BaseTrainingConfig):
    """Training hyperparameters (Hydra-enabled)"""
    pass


@dataclass
class MetricsConfig(BaseMetricsConfig):
    """Metrics computation (Hydra-enabled)"""
    THING_EEG_K_VALUES: List[int] = field(default_factory=lambda: [2, 4, 10, 50, 100, 200])
    IMAGENET_EEG_K_VALUES: List[int] = field(default_factory=lambda: [2, 4, 10, 40])


@dataclass
class PathConfig:
    """File paths (Hydra-enabled with proper type annotations)"""
    # Root path (auto-detected, or via WaveMind_ROOT_PATH_ env var for backward compatibility)
    ROOT_PATH: str = field(default_factory=lambda: get_wavemind_root())

    # Data directories (computed from ROOT_PATH)
    DATA_DIR: Optional[str] = None
    HDF5_PATH: Optional[str] = None
    CLIP_GROUND_TRUTH_DIR: Optional[str] = None

    # Model directories
    CHECKPOINT_DIR: Optional[str] = None

    # Logs directory
    LOGS_DIR: Optional[str] = None

    # Shared memory
    SHM_DIR: str = '/dev/shm'
    SHM_AVAILABLE_BUFFER_RATIO: float = 0.8  # Reserve 20% safety buffer

    def __post_init__(self):
        """Initialize paths based on ROOT_PATH (Hydra compatible)"""
        # Don't run if values are already set (e.g., from YAML)
        if self.DATA_DIR is None:
            self.DATA_DIR = os.path.join(self.ROOT_PATH, 'data/Total')
        if self.HDF5_PATH is None:
            self.HDF5_PATH = os.path.join(self.DATA_DIR if self.DATA_DIR else os.path.join(self.ROOT_PATH, 'data/Total'), 'data_label.h5')
        if self.CLIP_GROUND_TRUTH_DIR is None:
            self.CLIP_GROUND_TRUTH_DIR = os.path.join(self.DATA_DIR if self.DATA_DIR else os.path.join(self.ROOT_PATH, 'data/Total'), 'CLIP_groundTruth')
        if self.CHECKPOINT_DIR is None:
            self.CHECKPOINT_DIR = os.path.join(self.ROOT_PATH, 'EEG_Encoder/Resource/Checkpoint')
        if self.LOGS_DIR is None:
            self.LOGS_DIR = os.path.join(self.ROOT_PATH, 'EEG_Encoder/Resource/logs')


@dataclass
class DataAugmentationConfig(BaseDataAugmentationConfig):
    """Data augmentation settings (Hydra-enabled)"""
    NOTCH_FILTER_FREQS: List[int] = field(default_factory=lambda: [50, 60])


@dataclass
class HDF5Config(BaseHDF5Config):
    """HDF5 handling (Hydra-enabled)"""
    pass


@dataclass
class ExperimentConfig:
    """Experiment-specific parameters (CLI arguments)"""
    models: List[str] = field(default_factory=lambda: ['ATMSmodify'])
    datasets: List[str] = field(default_factory=lambda: ['all'])
    gpu_number: List[str] = field(default_factory=lambda: ['0'])
    limit_sample: Optional[int] = None


@dataclass
class ModeConfig:
    """Training/evaluation mode flags"""
    only_evaluate_SD: bool = False
    only_evaluate_SI: bool = False
    save_model: bool = False


@dataclass
class FeatureConfig:
    """Feature flags for optional functionality"""
    use_shm: bool = True
    use_dynamic_sampling: bool = True
    auto_clean_shm_when_exit: bool = True


@dataclass
class LoggerConfig:
    """Experiment logging configuration"""
    type: str = "none"  # "none", "comet", or "swanlab"
    experiment_name: Optional[str] = None  # Auto-generated if None
    save_dir: Optional[str] = None  # Log save directory (None = use paths.LOGS_DIR)


@dataclass
class AdvancedConfig:
    """Advanced/infrequently used parameters"""
    model_checkpoint_name: Optional[str] = None
    thing_specific_sub: List[str] = field(default_factory=lambda: ['sub-01'])


@dataclass
class ClassifierConfig:
    """Classifier training configuration

    Note: Classifier is automatically enabled when lambda_clip < 1.0
    """
    # Loss weighting: 1.0=pure CLIP, 0.5=joint, 0.0=pure classifier
    lambda_clip: float = 1.0

    # Classifier architecture
    hidden_dim: int = 512
    dropout: float = 0.5
    use_simple_head: bool = True  # true: LayerNorm+Linear, false: MLP

    # Training hyperparameters
    label_smoothing: float = 0.2

    # Dataset class counts (auto-detected, can override)
    dataset_n_class: dict = field(default_factory=lambda: {
        'TUAB': 2,
        'TUEV': 6,
        'SEED': 3,
        'SEED-IV': 4,
        'BCICIV2a': 4,
        'FACE': 9,
        'thingEEG': 1654,
        'ImageNetEEG': 40
    })


@dataclass
class WaveMindConfig:
    """Root configuration for WaveMind training pipeline"""

    # Nested configurations
    eeg: EEGConfig = field(default_factory=EEGConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    augmentation: DataAugmentationConfig = field(default_factory=DataAugmentationConfig)
    hdf5: HDF5Config = field(default_factory=HDF5Config)

    # CLI-specific configurations
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    modes: ModeConfig = field(default_factory=ModeConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)


def register_configs():
    """Register all configs with Hydra ConfigStore"""
    cs = ConfigStore.instance()
    # Don't register 'base' schema to avoid strict validation issues with Hydra keys
    # cs.store(name="base", node=WaveMindConfig)
    cs.store(group="eeg", name="default", node=EEGConfig)
    cs.store(group="training", name="default", node=TrainingConfig)
    cs.store(group="metrics", name="default", node=MetricsConfig)
    cs.store(group="paths", name="default", node=PathConfig)
    cs.store(group="augmentation", name="default", node=DataAugmentationConfig)
    cs.store(group="hdf5", name="default", node=HDF5Config)
    cs.store(group="logger", name="default", node=LoggerConfig)
    cs.store(group="classifier", name="default", node=ClassifierConfig)


# Auto-register on import
register_configs()
