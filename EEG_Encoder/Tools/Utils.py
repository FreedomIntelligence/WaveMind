import torch
import torchaudio
import torch.distributed as dist
import torch
from scipy.signal import iirnotch, filtfilt
import numpy as np
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn

def get_compute_device():
    """
    Get the appropriate compute device for current process.
    Supports DDP multi-GPU, single GPU, and CPU-only environments.
    
    Returns:
        torch.device: The compute device for current process
    """
    if torch.cuda.is_available():
        if dist.is_initialized():
            # DDP environment: use device corresponding to current rank
            rank = dist.get_rank()
            device = torch.device(f"cuda:{rank}")
        else:
            # Single GPU environment: use default device
            device = torch.device("cuda")
        return device
    else:
        return torch.device("cpu")

class EEGNoiseGenerator:
    """
    Minimal EEG Preprocessing Class with Pure PyTorch (GPU-ready)
    Handles 2D (C, T) inputs only.
    - Uses torchaudio for 50/60Hz notch filtering (bandreject_biquad)
    - All operations run on GPU if input is on GPU
    - No SciPy dependency
    """

    def __init__(self,
                 train_config=None,
                 test_config=None,
                 sample_rate: int = 512,
                 verbose: bool = False):
        """
        Args:
            train_config: TrainAugmentationConfig object for training augmentation
            test_config: TestAugmentationConfig object for test/val augmentation
            sample_rate: EEG sampling rate in Hz
            verbose: Enable debug prints
        """
        self.sample_rate = sample_rate
        self.verbose = verbose

        # Store config objects
        self.train_config = train_config
        self.test_config = test_config

        # For backward compatibility: if configs are None, use defaults
        if self.train_config is None:
            from EEG_Encoder.Config.training_config import TrainAugmentationConfig
            self.train_config = TrainAugmentationConfig()
        if self.test_config is None:
            from EEG_Encoder.Config.training_config import TestAugmentationConfig
            self.test_config = TestAugmentationConfig()

    def _apply_notch_filters(self,eeg: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Apply 50Hz and 60Hz notch filters using scipy (zero-phase distortion).
        
        Parameters:
            eeg (torch.Tensor): Input EEG data with shape (n_channels, n_times).
            sample_rate (float): Sampling rate of the EEG data in Hz.
            verbose (bool): Whether to print progress messages.

        Returns:
            torch.Tensor: Filtered EEG data with shape (n_channels, n_times).
        """
        if verbose:
            print("Applying 50Hz and 60Hz notch filters")
            
        # get eeg device
        device = eeg.device

        # Convert input tensor to NumPy array for scipy processing
        eeg_np = eeg.cpu().numpy()

        # Nyquist frequency
        nyquist = self.sample_rate / 2

        # Loop over each target frequency (50Hz and 60Hz)
        for freq in [50.0, 60.0]:
            if freq < nyquist:
                # Design the notch filter
                b, a = iirnotch(w0=freq, Q=30.0, fs=self.sample_rate)

                # Apply zero-phase filtering along the time axis (axis=-1)
                eeg_np = filtfilt(b, a, eeg_np, axis=-1)
                
        eeg_np = np.ascontiguousarray(eeg_np)

        # Convert back to torch.Tensor
        filtered_eeg = torch.from_numpy(eeg_np).to(device=device)

        return filtered_eeg

    def _zscore_global(self, eeg: torch.Tensor) -> torch.Tensor:
        """Global z-score normalization"""
        if self.verbose:
            print("Applying global z-score normalization")
        mean = torch.mean(eeg)
        std = torch.std(eeg)
        std = torch.clamp(std, min=1e-8)
        return (eeg - mean) / std

    def _scale_by_std_global(self, eeg: torch.Tensor) -> torch.Tensor:
        """Global std scaling (no mean subtraction)"""
        if self.verbose:
            print("Applying global std scaling")
        std = torch.std(eeg)
        std = torch.clamp(std, min=1e-8)
        return eeg / std

    def _zscore_channel(self, eeg: torch.Tensor) -> torch.Tensor:
        """Channel-wise z-score normalization"""
        if self.verbose:
            print("Applying channel-wise z-score normalization")
        mean = torch.mean(eeg, dim=-1, keepdim=True)
        std = torch.std(eeg, dim=-1, keepdim=True)
        std = torch.clamp(std, min=1e-8)
        return (eeg - mean) / std

    def _scale_by_std_channel(self, eeg: torch.Tensor) -> torch.Tensor:
        """Channel-wise std scaling (no mean subtraction)"""
        if self.verbose:
            print("Applying channel-wise std scaling")
        std = torch.std(eeg, dim=-1, keepdim=True)
        std = torch.clamp(std, min=1e-8)
        return eeg / std

    def _add_amplitude_fluctuation(self, eeg: torch.Tensor) -> torch.Tensor:
        """Apply random ±X% scaling across entire signal"""
        if self.verbose:
            print(f"Applying amplitude fluctuation of ±{self.fluctuation_amp*100:.1f}%")
        scale = 1.0 + self.fluctuation_amp * (2 * torch.rand((), device=eeg.device) - 1)
        return eeg * scale

    def signal_process_train(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Training-time preprocessing:
        1. Remove 50/60Hz powerline noise (GPU) - DISABLED
        2. Apply normalization based on config
        3. Add amplitude fluctuation based on config
        """
        if eeg.ndim != 2:
            raise ValueError(f"Input must be 2D (C, T), got {eeg.ndim}D")

        # Check if augmentation is enabled
        if not self.train_config.enable:
            return eeg  # No augmentation, return as-is

        # Step 1: Notch filtering (DISABLED - commented out in original code)
        # eeg = self._apply_notch_filters(eeg)

        # Step 2: Select normalization method based on config
        norm_method = self.train_config.normalization_method

        if norm_method == "random":
            # Randomly choose from 5 methods
            scale_methods = [
                lambda x: x,  # identity
                self._zscore_global,
                self._scale_by_std_global,
                self._zscore_channel,
                self._scale_by_std_channel,
            ]
            idx = torch.randint(len(scale_methods), (), device=eeg.device).item()
            scale_func = scale_methods[idx]
        elif norm_method == "zscore_global":
            scale_func = self._zscore_global
        elif norm_method == "zscore_channel":
            scale_func = self._zscore_channel
        elif norm_method == "std_global":
            scale_func = self._scale_by_std_global
        elif norm_method == "std_channel":
            scale_func = self._scale_by_std_channel
        elif norm_method == "identity":
            scale_func = lambda x: x
        else:
            raise ValueError(f"Unknown normalization method: {norm_method}")

        eeg = scale_func(eeg)

        # Step 3: Amplitude fluctuation based on config
        if self.train_config.amplitude_fluctuation_enable:
            amp_min, amp_max = self.train_config.amplitude_fluctuation_range
            fluctuation_amp = (amp_max - amp_min) / 2  # Convert range to ±amplitude
            if self.verbose:
                print(f"Applying amplitude fluctuation range {amp_min}-{amp_max}")
            scale = amp_min + (amp_max - amp_min) * torch.rand((), device=eeg.device)
            eeg = eeg * scale

        return eeg

    def signal_process_test(self, eeg: torch.Tensor) -> torch.Tensor:
        """
        Test-time preprocessing:
        1. Remove 50/60Hz powerline noise (GPU) - DISABLED
        2. Apply deterministic normalization based on config
        """
        if eeg.ndim != 2:
            raise ValueError(f"Input must be 2D (C, T), got {eeg.ndim}D")

        # Check if augmentation is enabled
        if not self.test_config.enable:
            return eeg  # No augmentation, return as-is

        # Apply deterministic normalization based on config
        norm_method = self.test_config.normalization_method

        if norm_method == "zscore_global":
            eeg = self._zscore_global(eeg)
        elif norm_method == "zscore_channel":
            eeg = self._zscore_channel(eeg)
        elif norm_method == "std_global":
            eeg = self._scale_by_std_global(eeg)
        elif norm_method == "std_channel":
            eeg = self._scale_by_std_channel(eeg)
        elif norm_method == "identity":
            pass  # No normalization
        else:
            raise ValueError(f"Unknown normalization method: {norm_method}")

        return eeg


def print_training_parameters(args, models_names, gpu_number, epoch, dataset, use_aug, limit_sample, load_checkpoint_name):
    """
    Print all training parameters
    """
    rank_zero_info("=" * 50)
    rank_zero_info("Training Parameters:")
    rank_zero_info("=" * 50)
    rank_zero_info(f"Models: {models_names}")
    rank_zero_info(f"GPU Number: {gpu_number}")
    rank_zero_info(f"Epochs: {epoch}")
    rank_zero_info(f"Dataset: {dataset}")
    rank_zero_info(f"Learning Rate: {args.lr}")
    rank_zero_info(f"Use Augmentation: {use_aug}")
    rank_zero_info(f"Limit Sample: {limit_sample}")
    rank_zero_info(f"Number of Workers: {args.num_workers}")
    rank_zero_info(f"Use Shared Memory: {args.use_shm}")
    rank_zero_info(f"Use Dynamic Sampling: {args.use_dynamic_sampling}")
    rank_zero_info(f"Only Evaluate SD: {args.only_evaluate_SD}")
    rank_zero_info(f"Only Evaluate SI: {args.only_evaluate_SI}")
    rank_zero_info(f"Model Checkpoint: {load_checkpoint_name}")
    rank_zero_info(f"HDF5 Path: {args.hdf5_path}")
    if dataset == 'thing_specific':
        rank_zero_info(f"Thing Specific Subjects: {args.thing_specific_sub}")
    rank_zero_info("=" * 50)



def get_gpu_with_lowest_utilization():
    """
    Get GPU with lowest utilization rate
    Returns: List of GPU numbers, returns [0] if cannot get GPU info
    """
    try:
        # Use pynvml to get GPU utilization info
        import pynvml
        
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        gpu_info = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used = mem_info.used / 1024**2  # Convert to MB
            memory_total = mem_info.total / 1024**2  # Convert to MB
            
            # Get GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_utilization = util.gpu
            
            # Calculate memory utilization percentage
            memory_utilization = (memory_used / memory_total) * 100
            
            # Calculate total utilization (average of GPU and memory utilization)
            total_utilization = (gpu_utilization + memory_utilization) / 2
            
            gpu_info.append({
                'id': i,
                'memory_used': memory_used,
                'memory_total': memory_total,
                'utilization': gpu_utilization,
                'memory_utilization': memory_utilization,
                'total_utilization': total_utilization
            })
        
        pynvml.nvmlShutdown()
        
        if not gpu_info:
            rank_zero_info("No GPU information found, using default GPU 0")
            return [0]
        
        # Sort by total utilization and select the lowest
        gpu_info.sort(key=lambda x: x['total_utilization'])
        best_gpu = gpu_info[0]['id']

        rank_zero_info("Available GPUs info:")
        for gpu in gpu_info:
            rank_zero_info(f"GPU {gpu['id']}: GPU utilization {gpu['utilization']}%, Memory utilization {gpu['memory_utilization']:.1f}%, Total utilization {gpu['total_utilization']:.1f}%")

        rank_zero_info(f"Selected GPU {best_gpu} with lowest utilization ({gpu_info[0]['total_utilization']:.1f}%)")
        return [best_gpu]
        
    except ImportError:
        rank_zero_info("pynvml not available, trying torch.cuda")
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_info = []
                
                for i in range(gpu_count):
                    # Get memory usage
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**2  # MB
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**2  # MB
                    memory_total = torch.cuda.get_device_properties(i).total_memory / 1024**2  # MB
                    
                    # Estimate utilization based on memory usage
                    memory_utilization = (memory_reserved / memory_total) * 100
                    
                    gpu_info.append({
                        'id': i,
                        'memory_allocated': memory_allocated,
                        'memory_reserved': memory_reserved,
                        'memory_total': memory_total,
                        'memory_utilization': memory_utilization,
                        'total_utilization': memory_utilization  # Use memory utilization as proxy
                    })
                
                if gpu_info:
                    gpu_info.sort(key=lambda x: x['total_utilization'])
                    best_gpu = gpu_info[0]['id']

                    rank_zero_info("Available GPUs info (using torch.cuda):")
                    for gpu in gpu_info:
                        rank_zero_info(f"GPU {gpu['id']}: Memory allocated {gpu['memory_allocated']:.1f}MB, Memory reserved {gpu['memory_reserved']:.1f}MB, Memory utilization {gpu['memory_utilization']:.1f}%")

                    rank_zero_info(f"Selected GPU {best_gpu} with lowest memory utilization ({gpu_info[0]['total_utilization']:.1f}%)")
                    return [best_gpu]
            
            rank_zero_info("No CUDA devices available, using default GPU 0")
            return [0]
            
        except Exception as e:
            rank_zero_warn(f"Error getting GPU info with torch.cuda: {e}")
            rank_zero_info("Using default GPU 0")
            return [0]
    
    except Exception as e:
        rank_zero_warn(f"Error getting GPU utilization info: {e}")
        rank_zero_info("Using default GPU 0")
        return [0]




