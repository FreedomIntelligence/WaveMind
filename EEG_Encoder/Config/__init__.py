"""
EEG Encoder Configuration Package

Centralized configuration management for training hyperparameters,
EEG signal processing, and file paths.

Now supports both direct import (legacy) and Hydra-based configuration.
"""

from .training_config import (
    EEGConfig,
    TrainingConfig,
    MetricsConfig,
    PathConfig,
    DataAugmentationConfig,
    HDF5Config,
    eeg_config,
    training_config,
    metrics_config,
    path_config,
    aug_config,
    hdf5_config,
    get_model_sample_rate,
    get_k_values_for_dataset,
)

# Import Hydra configs (optional, only used by run_CLIPtraining.py)
try:
    from .hydra_configs import (
        WaveMindConfig,
        ExperimentConfig,
        ModeConfig,
        FeatureConfig,
        AdvancedConfig,
    )
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False
    WaveMindConfig = None

__all__ = [
    # Legacy dataclasses
    'EEGConfig',
    'TrainingConfig',
    'MetricsConfig',
    'PathConfig',
    'DataAugmentationConfig',
    'HDF5Config',
    'eeg_config',
    'training_config',
    'metrics_config',
    'path_config',
    'aug_config',
    'hdf5_config',
    'get_model_sample_rate',
    'get_k_values_for_dataset',

    # Hydra configs (if available)
    'WaveMindConfig',
    'ExperimentConfig',
    'ModeConfig',
    'FeatureConfig',
    'AdvancedConfig',
    'HYDRA_AVAILABLE',
]
