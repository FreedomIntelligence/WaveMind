"""
Base Dataset Classes for EEG Data Loading

This module provides abstract base classes for EEG datasets,
extracting common functionality from CLIPDataset and CLIPDataset_ThingEEG.
"""

from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset
from EEG_Encoder.Tools.Utils import EEGNoiseGenerator


class BaseEEGDataset(Dataset, ABC):
    """
    Abstract base class for EEG datasets with shared functionality.

    This class extracts common patterns from CLIPDataset and CLIPDataset_ThingEEG
    while keeping data loading logic separate (since they use different data sources:
    HDF5 vs .npy files).

    Subclasses must implement:
        - __len__(): Return total number of samples
        - __getitem__(): Load and return a single sample

    Subclasses should use the provided helper methods:
        - _apply_augmentation(): Apply train/test augmentation
        - _validate_eeg_data(): Check for NaN/Inf values
        - _resample_if_needed(): Resample to target sample rate
        - _standardize_return_dict(): Ensure consistent return format
    """

    def __init__(self, augmentation_config=None, use_aug=None, sample_rate=512, limit_sample=None,
                 float_type='float32', aug_verbose=False):
        """
        Initialize base EEG dataset with common parameters.

        Args:
            augmentation_config: AugmentationConfig object with train/test configs (preferred)
            use_aug: DEPRECATED - Use augmentation_config instead. Boolean for backward compatibility
            sample_rate: Target sampling rate in Hz (default: 512)
            limit_sample: Optional limit on number of samples (default: None)
            float_type: Data precision, 'float32' or 'float16' (default: 'float32')
            aug_verbose: Verbose logging for augmentation (default: False)
        """
        # Handle backward compatibility: convert use_aug boolean to augmentation_config
        if augmentation_config is None:
            if use_aug is not None:
                # Legacy mode: create config from boolean
                from EEG_Encoder.Config.training_config import AugmentationConfig, TrainAugmentationConfig, TestAugmentationConfig
                augmentation_config = AugmentationConfig(
                    train=TrainAugmentationConfig(enable=use_aug),
                    test=TestAugmentationConfig(enable=True)  # Always normalize for test
                )
            else:
                # No config provided: use defaults
                from EEG_Encoder.Config.training_config import AugmentationConfig
                augmentation_config = AugmentationConfig()

        # Store augmentation config
        self.augmentation_config = augmentation_config

        # For backward compatibility: keep use_aug attribute
        self.use_aug = self.augmentation_config.train.enable or self.augmentation_config.test.enable

        # Common attributes accessed by external code (CLIPDataModule, lightingModule)
        self.sample_rate = sample_rate
        self.limit_sample = limit_sample
        self.float_type = torch.float32 if float_type == 'float32' else torch.float16

        # Initialize augmentation pipeline
        self._initialize_noise_generator(aug_verbose)

    def _initialize_noise_generator(self, verbose=False):
        """
        Initialize EEG noise generator for augmentation with split-aware config.

        Args:
            verbose: Enable verbose logging for noise generator
        """
        # Always create noise generator with configs (it handles enable/disable internally)
        self.noise_gen = EEGNoiseGenerator(
            train_config=self.augmentation_config.train,
            test_config=self.augmentation_config.test,
            sample_rate=self.sample_rate,
            verbose=verbose,
        )

    def change_sample_rate(self, sample_rate):
        """
        Change the target sample rate for EEG data.

        This method is called polymorphically by CLIPDataModule.setup(),
        so the signature must remain stable for backward compatibility.

        Args:
            sample_rate: New target sample rate in Hz
        """
        self.sample_rate = sample_rate
        self.noise_gen.sample_rate = sample_rate

    def _apply_augmentation(self, eeg_data, is_train_mode):
        """
        Apply augmentation to EEG data based on mode.

        Args:
            eeg_data: EEG tensor [channels, time_points]
            is_train_mode: True for training augmentation, False for test/val

        Returns:
            Augmented EEG tensor with same shape
        """
        # Noise generator handles enable/disable internally based on config
        if is_train_mode:
            return self.noise_gen.signal_process_train(eeg_data)
        else:
            return self.noise_gen.signal_process_test(eeg_data)

    def _validate_eeg_data(self, eeg_data):
        """
        Check if EEG data contains NaN or Inf values.

        Args:
            eeg_data: EEG tensor to validate

        Returns:
            True if data is valid (no NaN/Inf), False otherwise
        """
        return not (torch.isnan(eeg_data).any() or torch.isinf(eeg_data).any())

    def _resample_if_needed(self, eeg_data):
        """
        Resample EEG data to target sample rate if necessary.

        Args:
            eeg_data: EEG array [channels, time_points]

        Returns:
            Resampled EEG array with shape [channels, self.sample_rate]
        """
        if eeg_data.shape[-1] != self.sample_rate:
            from scipy.signal import resample
            eeg_data = resample(eeg_data, self.sample_rate, axis=-1)
        return eeg_data

    def _standardize_return_dict(self, eeg_data, label, text, text_features,
                                  img_path, img_features, dataset_name):
        """
        Standardize return format across all datasets.

        This ensures consistent contract with external code (e.g., lightingModule).
        All datasets must return identical keys with compatible types.

        Args:
            eeg_data: EEG tensor [channels, time_points]
            label: Class label (int or tensor)
            text: Text description (str)
            text_features: CLIP text embeddings [768] or None
            img_path: Path to image (str)
            img_features: CLIP image embeddings [768] or None
            dataset_name: Source dataset name (str)

        Returns:
            Dictionary with standardized keys:
                - eeg_data: torch.Tensor (dtype=self.float_type)
                - label: torch.Tensor (dtype=torch.int)
                - text: str
                - text_features: torch.Tensor [768]
                - img_path: str
                - img_features: torch.Tensor [768]
                - dataset_name: str
        """
        # Ensure label is tensor with correct shape and dtype
        if not isinstance(label, torch.Tensor):
            # Handle different label formats
            if isinstance(label, list):
                label = torch.tensor(label, dtype=torch.int)
            else:
                label = torch.tensor([int(label)], dtype=torch.int)
        elif label.dim() == 0:  # Scalar tensor
            label = label.unsqueeze(0).type(torch.int)
        else:
            label = label.type(torch.int)

        # Ensure text features exist (create zero vector if None)
        if text_features is None:
            text_features = torch.zeros(768)

        # Ensure image features exist (create zero vector if None)
        if img_features is None:
            img_features = torch.zeros(768)

        return {
            'eeg_data': eeg_data.to(dtype=self.float_type),
            'label': label,
            'text': text,
            'text_features': text_features,
            'img_path': img_path,
            'img_features': img_features,
            'dataset_name': dataset_name
        }

    # Abstract methods that subclasses must implement
    @abstractmethod
    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples
        """
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Subclasses should follow this pattern:
        1. Load EEG data from their specific data source (HDF5, .npy, etc.)
        2. Apply resampling using _resample_if_needed() if needed
        3. Convert to torch.Tensor
        4. Apply augmentation using _apply_augmentation(eeg_data, is_train)
        5. Validate using _validate_eeg_data() and handle invalid data
        6. Return using _standardize_return_dict() for consistent format

        Args:
            idx: Sample index

        Returns:
            dict: Sample dictionary with standardized keys (see _standardize_return_dict)
        """
        pass
