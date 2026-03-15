
import logging

# Configure logging BEFORE importing Lightning-related modules
# This prevents Lightning from overriding our logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
import os
import sys
import warnings
# Ensure rank_zero_info works correctly
logger = logging.getLogger("lightning.pytorch.utilities.rank_zero")
logger.setLevel(logging.INFO)

from EEG_Encoder.Tools.lightingModule import LitModel_CLIP
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks import Callback
import os
import lightning as L
# Import centralized configuration
from EEG_Encoder.Config import (
    training_config,
    path_config,
    eeg_config,
)
from EEG_Encoder.Config.hydra_configs import WaveMindConfig

torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision('medium')

from EEG_Encoder.Tools.logger_utils import get_experiment_logger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_warn


from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from EEG_Encoder.Model.CommonBlock import Config
from EEG_Encoder.Model.baseModel import model_selection
from EEG_Encoder.Tools.dataBuilder import select_data_module_for_train, CLIPDataModule, CLIPDataset
from EEG_Encoder.Tools.Utils import get_gpu_with_lowest_utilization
from lightning import LightningDataModule


import time


# Use centralized path configuration
load_dir = path_config.DATA_DIR



# Use centralized configuration for checkpoint directory
load_model_dir = os.path.join(path_config.CHECKPOINT_DIR, 'ALL/')
os.makedirs(load_model_dir, exist_ok=True)  # Ensure directory exists




def get_callbacks(dm_, cfg=None):
    class CleaningCallback(Callback):
        def __init__(self,dm):
            self.dm=dm
            super().__init__()
        def on_exception(self, trainer, pl_module, exception):
            rank_zero_info("Exception occurred during training. Cleaning up resources...")
            if self.dm is not None:
                self.dm.__del__()
            return super().on_exception(trainer, pl_module, exception)
    callbacks=[]

    # Use YAML config for early stopping if available, otherwise fall back to training_config
    if cfg is not None:
        patience = cfg.training.EARLY_STOP_PATIENCE
        min_delta = cfg.training.EARLY_STOP_MIN_DELTA
        mode = cfg.training.EARLY_STOP_MODE
    else:
        patience = training_config.EARLY_STOP_PATIENCE
        min_delta = training_config.EARLY_STOP_MIN_DELTA
        mode = training_config.EARLY_STOP_MODE

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        min_delta=min_delta,
        verbose=True,
        mode=mode
    )
    cleaning_callback=CleaningCallback(dm=dm_)
    callbacks.append(early_stop_callback)
    callbacks.append(cleaning_callback)
    return callbacks


# ============================================================================
# Training Pipeline Functions (Refactored for better readability)
# ============================================================================

def validate_config(cfg: DictConfig):
    """
    Validate Hydra configuration and perform consistency checks

    Args:
        cfg: Hydra DictConfig object

    Raises:
        ValueError: If configuration is invalid
    """
    # Validation checks (same as original)
    if cfg.modes.only_evaluate_SD and cfg.modes.only_evaluate_SI:
        raise ValueError("modes.only_evaluate_SD and modes.only_evaluate_SI cannot be both True")

    # Check required fields for evaluation modes
    if (cfg.modes.only_evaluate_SD or cfg.modes.only_evaluate_SI):
        if cfg.advanced.model_checkpoint_name is None:
            raise ValueError(
                "advanced.model_checkpoint_name must be provided when using evaluation modes. "
                "Use: advanced.model_checkpoint_name=/path/to/checkpoint.pth"
            )

    # Validate GPU numbers
    if cfg.experiment.gpu_number and cfg.experiment.gpu_number not in [['auto'], ['all']]:
        try:
            [int(x) for x in cfg.experiment.gpu_number if x not in ['auto', 'all']]
        except ValueError:
            raise ValueError(
                f"Invalid GPU numbers: {cfg.experiment.gpu_number}. "
                "Use integers (e.g., ['0', '1', '2']), ['auto'], or ['all']"
            )

    return cfg


def setup_gpu_and_dataset(cfg: DictConfig):
    """Setup GPU selection and dataset configuration from Hydra config"""

    # GPU selection: auto, all, or manual
    if not cfg.experiment.gpu_number or cfg.experiment.gpu_number == ['auto']:
        gpu_number = get_gpu_with_lowest_utilization()
    elif cfg.experiment.gpu_number == ['all']:
        # Auto-detect all available GPUs
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_number = list(range(gpu_count))
            rank_zero_info(f"Auto-detected {gpu_count} GPUs: {gpu_number}")
        else:
            raise RuntimeError("CUDA not available, cannot use gpu_number=['all']")
    else:
        gpu_number = [int(x) for x in cfg.experiment.gpu_number]

    # Dataset name handling: convert to string if single dataset
    if len(cfg.experiment.datasets) == 1:
        dataset = cfg.experiment.datasets[0]
    else:
        dataset = cfg.experiment.datasets

    return gpu_number, dataset


def setup_data_module(cfg: DictConfig, dataset):
    """Initialize and configure data module"""
    # Check if we're in evaluation-only mode
    is_eval_mode = cfg.modes.only_evaluate_SD or cfg.modes.only_evaluate_SI

    if is_eval_mode:
        # For evaluation mode, create a simplified datamodule that only loads test data
        dm = create_eval_data_module(
            dataset,
            load_dir=load_dir,
            hdf5_path=cfg.paths.HDF5_PATH,
            num_workers=cfg.training.DEFAULT_NUM_WORKERS,
            use_shm=cfg.features.use_shm,
            thing_specific_sub=cfg.advanced.thing_specific_sub,
            auto_clean_shm_when_exit=cfg.features.auto_clean_shm_when_exit,
            batch_size=cfg.training.DEFAULT_BATCH_SIZE
        )
    else:
        dm = select_data_module_for_train(
            dataset,
            augmentation_config=cfg.augmentation,
            limit_sample=cfg.experiment.limit_sample,
            load_dir=load_dir,
            hdf5_path=cfg.paths.HDF5_PATH,
            num_workers=cfg.training.DEFAULT_NUM_WORKERS,
            use_shm=cfg.features.use_shm,
            use_dynamic_sampling=cfg.features.use_dynamic_sampling,
            thing_specific_sub=cfg.advanced.thing_specific_sub,
            auto_clean_shm_when_exit=cfg.features.auto_clean_shm_when_exit,
            batch_size=cfg.training.DEFAULT_BATCH_SIZE
        )
    return dm


def create_eval_data_module(dataset, load_dir, hdf5_path, num_workers, use_shm, thing_specific_sub=None, auto_clean_shm_when_exit=True, batch_size=512):
    """
    Create a datamodule for evaluation mode (only loads test data, no training required).
    """

    # Convert omegaconf.ListConfig to list if needed
    if isinstance(dataset, str):
        dataset = [dataset]  # Wrap string in a list
    elif not isinstance(dataset, list):
        dataset = list(dataset)  # Convert ListConfig to list

    # Handle 'thing_specific' dataset name
    if 'thing_specific' in str(dataset):
        task = 'THING'
        subjects = thing_specific_sub if thing_specific_sub else ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09']
        dm = CLIPDataModule(task='THING', thing_train_val_same_set=False,
                           batch_size=batch_size, num_workers=num_workers,
                           subjects=subjects, limit_sample=None, load_dir=load_dir,
                           augmentation_config=None, hdf5_file_path=hdf5_path,
                           auto_clean_shm_when_exit=auto_clean_shm_when_exit)
        dm.setup()
    else:
        # For BCI tasks, only load test/val data directly using CLIPDataset
        ground_truth_dir = os.path.join(load_dir, 'CLIP_groundTruth')

        # Setup shared memory if needed
        tmp_hdf5_path = hdf5_path
        if use_shm:
            from EEG_Encoder.Tools.dataBuilder import CLIPDataModule
            try:
                dm_temp = CLIPDataModule(
                    task='BCI',
                    dataset_name=dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    limit_sample=None,
                    load_dir=load_dir,
                    augmentation_config=None,
                    hdf5_file_path=hdf5_path,
                    use_shm=use_shm,
                    use_dynamic_sampling=False,
                    auto_clean_shm_when_exit=auto_clean_shm_when_exit
                )
                tmp_hdf5_path = dm_temp.setup_shared_memory(hdf5_path)
                rank_zero_info(f"[Eval Mode] Using shared memory: {tmp_hdf5_path}")
            except Exception as e:
                rank_zero_warn(f"[Eval Mode] Shared memory setup failed, using original path: {e}")

        # Create a simple object that mimics CLIPDataModule for evaluation
        # Only create val_dataset (test mode uses 'test') and test_dataset (cross mode uses 'cross')
        val_dataset = CLIPDataset(
            hdf5_file_path=tmp_hdf5_path,
            mode='test',
            dataset_name=dataset,
            ground_truth_dir=ground_truth_dir,
            limit_sample=None
        )

        test_dataset = CLIPDataset(
            hdf5_file_path=tmp_hdf5_path,
            mode='cross',
            dataset_name=dataset,
            ground_truth_dir=ground_truth_dir,
            limit_sample=None
        )

        # Build feature_all_test from ground truths
        feature_all_test = {}
        ground_truths = val_dataset.ground_truths
        for ds_name, features in ground_truths.items():
            k_value = features.shape[0]
            feature_all_test[ds_name] = [features, k_value]

        # Create a LightningDataModule for evaluation
        class EvalDataModule(LightningDataModule):
            def __init__(self, val_ds, test_ds, features, batch_size_val, num_workers_val):
                super().__init__()
                self.val_dataset = val_ds
                self.test_dataset = test_ds if test_ds and test_ds.__len__() > 0 else None
                self.feature_all_test = features
                self.dataset_names = list(features.keys())
                self.batch_size = batch_size_val
                self.num_workers = num_workers_val

            def setup(self, stage=None):
                pass  # Already set up in __init__

            def val_dataloader(self):
                from torch.utils.data import DataLoader
                return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

            def test_dataloader(self):
                if self.test_dataset is None:
                    return None
                from torch.utils.data import DataLoader
                return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

            def teardown(self, stage=None):
                pass  # No cleanup needed for evaluation mode

        dm = EvalDataModule(val_dataset, test_dataset, feature_all_test, batch_size, num_workers)
        rank_zero_info(f"[Eval Mode] Data loaded: val={len(val_dataset)}, cross={len(test_dataset) if test_dataset else 0}")

    return dm


def setup_model(model_name, cfg: DictConfig, data_module):
    """
    Initialize EEG model and Lightning module

    Returns:
        (lightning_module, cleaned_model_name)
    """

    # Load model with optional checkpoint
    confirm_load = bool(cfg.advanced.model_checkpoint_name or cfg.modes.only_evaluate_SD or cfg.modes.only_evaluate_SI)
    rank_zero_info(f"[DEBUG-setup_model] confirm_load set to: {confirm_load}")

    eeg_model, sample_rate = model_selection(
        model_name,
        load_dir=load_model_dir,
        confirm_load=confirm_load,
        checkpoint_name=cfg.advanced.model_checkpoint_name
    )

    # Adjust data module sample rate if needed
    if sample_rate != eeg_config.DEFAULT_SAMPLE_RATE:
        rank_zero_info(f"Model uses sample rate {sample_rate} (default: {eeg_config.DEFAULT_SAMPLE_RATE}), adjusting data module...")
        data_module.change_sample_rate_hook(sample_rate)

    # Clean model name (remove underscore suffix if present)
    cleaned_name = model_name.split('_')[0] if '_' in model_name else model_name

    # Classifier configuration from Hydra config
    # Auto-enable classifier when lambda_clip < 1.0
    lambda_clip = cfg.classifier.lambda_clip if 'classifier' in cfg else 1.0
    use_CLS = (lambda_clip < 1.0)

    # Build classifier config dict from Hydra config
    classifier_config = None
    if use_CLS:
        from omegaconf import OmegaConf
        classifier_config = {
            'hidden_dim': cfg.classifier.hidden_dim,
            'dropout': cfg.classifier.dropout,
            'use_simple_head': cfg.classifier.use_simple_head,
            'label_smoothing': cfg.classifier.label_smoothing,
            'dataset_n_class': OmegaConf.to_container(cfg.classifier.dataset_n_class, resolve=True)
        }
        # Log classifier status at training start
        if lambda_clip == 0.0:
            rank_zero_info(f"[Classifier Training] Pure supervised classification (lambda_clip={lambda_clip:.2f})")
            rank_zero_info(f"[Classifier Training] CLIP loss disabled, only classification loss active")
        else:
            rank_zero_info(f"[Classifier Training] Joint CLIP + Classifier training (lambda_clip={lambda_clip:.2f})")
            rank_zero_info(f"[Classifier Training] Loss = {lambda_clip:.2f} * CLIP + {1-lambda_clip:.2f} * Classifier")

        # Log architecture details
        if classifier_config['use_simple_head']:
            arch_str = "Simple (LayerNorm+Linear)"
        else:
            arch_str = f"MLP (hidden_dim={classifier_config['hidden_dim']})"
        rank_zero_info(f"[Classifier Training] Architecture: {arch_str}")
        rank_zero_info(f"[Classifier Training] Config: dropout={classifier_config['dropout']}, label_smoothing={classifier_config['label_smoothing']}")
    else:
        rank_zero_info(f"[CLIP Training] Pure CLIP contrastive learning (lambda_clip={lambda_clip:.2f})")
        rank_zero_info(f"[CLIP Training] Classifier disabled")

    # Add suffix to model name for tracking
    if use_CLS:
        if lambda_clip == 0.0:
            suffix = "_CLS"
        else:
            suffix = f"_JOINT{int(lambda_clip*100)}"
        cleaned_name = cleaned_name + suffix

    # Wrap model in Lightning module if not already wrapped
    if not isinstance(eeg_model, L.LightningModule):
        lightning_module = LitModel_CLIP(
            EEGencoder=eeg_model,
            lr=cfg.training.DEFAULT_LEARNING_RATE,
            batch_size=data_module.batch_size,
            w_cls_compute=use_CLS,
            lambda_clip=lambda_clip,
            classifier_config=classifier_config
        ).type(torch.float32)
    else:
        lightning_module = eeg_model

    return lightning_module, cleaned_name


def setup_trainer(cfg: DictConfig, gpu_number, callbacks):
    """Configure and initialize PyTorch Lightning Trainer"""
    # Create experiment logger based on configuration
    experiment_logger = get_experiment_logger(cfg)

    # Determine if checkpointing should be enabled
    # Disable checkpointing in evaluation-only mode OR when save_model is False
    is_evaluation_only = cfg.modes.only_evaluate_SD or cfg.modes.only_evaluate_SI
    enable_checkpointing = not is_evaluation_only and cfg.modes.save_model

    # Determine Lightning log directory
    # Use same priority logic as logger_utils.py
    if hasattr(cfg.logger, 'save_dir') and cfg.logger.save_dir is not None:
        default_root_dir = cfg.logger.save_dir
    else:
        default_root_dir = cfg.paths.LOGS_DIR

    # Ensure directory exists
    import os
    os.makedirs(default_root_dir, exist_ok=True)
    rank_zero_info(f"Lightning logs will be saved to: {default_root_dir}")

    # Use centralized configuration for trainer parameters
    trainer = L.Trainer(
        max_epochs=cfg.training.DEFAULT_EPOCHS,
        devices=gpu_number,
        accelerator='gpu',
        accumulate_grad_batches=cfg.training.ACCUMULATE_GRAD_BATCHES,
        gradient_clip_val=cfg.training.GRADIENT_CLIP_VAL,
        precision=cfg.training.PRECISION,
        log_every_n_steps=cfg.training.LOG_EVERY_N_STEPS,
        check_val_every_n_epoch=cfg.training.CHECK_VAL_EVERY_N_EPOCH,
        num_sanity_val_steps=cfg.training.NUM_SANITY_VAL_STEPS,
        callbacks=callbacks,
        logger=experiment_logger,  # Use configured logger (comet/swanlab/none)
        enable_progress_bar=True,
        enable_checkpointing=enable_checkpointing,
        strategy=DDPStrategy(find_unused_parameters=True) if len(gpu_number) > 1 else 'auto',
        default_root_dir=default_root_dir,  # Control lightning_logs location
    )
    return trainer


def run_training_pipeline(trainer, model, data_module, cfg: DictConfig):
    """
    Execute training, validation, and testing based on mode

    Modes:
        - only_evaluate_SD: Run validation only (subject dependent)
        - only_evaluate_SI: Run test only (subject independent)
        - normal: Full training + validation + test pipeline
    """
    if cfg.modes.only_evaluate_SD:
        rank_zero_info("Running Subject Dependent (SD) evaluation only...")
        trainer.validate(model, data_module)

    elif cfg.modes.only_evaluate_SI:
        rank_zero_info("Running Subject Independent (SI) evaluation only...")
        if hasattr(data_module, 'test_dataset'):
            trainer.test(model, data_module)
        else:
            raise RuntimeError("No test dataset available for only_evaluate_SI mode")

    else:
        rank_zero_info("Running full training pipeline...")

        # Pre-training validation to establish baseline performance
        rank_zero_info("Running pre-training validation (baseline)...")
        trainer.validate(model, data_module)

        # Training (includes periodic validation during training)
        rank_zero_info("Starting training...")
        trainer.fit(model, data_module)

        # Final validation on best checkpoint to measure improvement
        rank_zero_info("Running post-training validation (final)...")
        trainer.validate(model, data_module)

        # Test if available
        if hasattr(data_module, 'test_dataset') and data_module.test_dataset is not None:
            rank_zero_info("Running test evaluation...")
            trainer.test(model, data_module)


def save_model_checkpoint(model, model_name, dataset, cfg: DictConfig):
    """
    Save trained model checkpoint with DDP-safe filename generation

    The filename is generated on rank 0 and broadcast to all ranks to ensure
    consistency in multi-GPU training.
    """
    if not cfg.modes.save_model:
        rank_zero_info("Model checkpoint saving disabled. Use modes.save_model=true to save the trained model.")
        return

    from datetime import timedelta

    # Generate deterministic filename on rank 0 and broadcast to all ranks
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            timestamp = int(time.time())
            limit_str = cfg.experiment.limit_sample if cfg.experiment.limit_sample is not None else ''
            checkpoint_name = f"{model_name}_{dataset}_{limit_str}_{timestamp}.pth"
            save_path = os.path.join(load_model_dir, checkpoint_name)
            path_container = [save_path]
        else:
            path_container = [None]

        # Broadcast filename from rank 0 to all ranks
        torch.distributed.broadcast_object_list(path_container, src=0)
        save_path = path_container[0]

        rank_zero_info(f"All ranks using checkpoint path: {save_path}")

        # Barrier BEFORE save to ensure model state is synchronized
        torch.distributed.barrier(timeout=timedelta(minutes=5))

        # Only rank 0 saves the checkpoint
        if torch.distributed.get_rank() == 0:
            try:
                torch.save(model.EEGencoder.state_dict(), save_path)
                rank_zero_info(f"[Rank0] Checkpoint saved successfully: {save_path}")
            except Exception as e:
                rank_zero_info(f"[Rank0] Checkpoint save failed: {str(e)}")
                raise  # Re-raise to trigger barrier timeout on other ranks

        # Barrier AFTER save to ensure file is complete before other ranks proceed
        torch.distributed.barrier(timeout=timedelta(minutes=5))
        rank_zero_info(f"All ranks synchronized after checkpoint save")

    else:
        # Single GPU case
        timestamp = int(time.time())
        limit_str = cfg.experiment.limit_sample if cfg.experiment.limit_sample is not None else ''
        checkpoint_name = f"{model_name}_{dataset}_{limit_str}_{timestamp}.pth"
        save_path = os.path.join(load_model_dir, checkpoint_name)
        torch.save(model.EEGencoder.state_dict(), save_path)
        rank_zero_info(f"Checkpoint saved successfully: {save_path}")



@hydra.main(version_base=None, config_path="examples", config_name=None)
def main(cfg: DictConfig) -> None:

    # Validate configuration
    validate_config(cfg)

    # Setup GPU selection and dataset configuration
    gpu_number, dataset = setup_gpu_and_dataset(cfg)

    # Initialize data module (shared across all models)
    dm = setup_data_module(cfg, dataset)

    # Train each model in the model list
    for model_name in cfg.experiment.models:
        rank_zero_info(f"\n{'='*80}")
        rank_zero_info(f"Training Model: {model_name}")
        rank_zero_info(f"{'='*80}\n")

        # Setup model and Lightning module
        lm, cleaned_model_name = setup_model(model_name, cfg, dm)

        # Setup trainer with callbacks
        callbacks = get_callbacks(dm, cfg)
        trainer = setup_trainer(cfg, gpu_number, callbacks)

        # Run training/evaluation pipeline
        run_training_pipeline(trainer, lm, dm, cfg)

        # Save checkpoint if requested
        save_model_checkpoint(lm, cleaned_model_name, dataset, cfg)

        rank_zero_info(f"\n{'='*80}")
        rank_zero_info(f"Completed Model: {model_name}")
        rank_zero_info(f"{'='*80}\n")


if __name__ == '__main__':
    main()


    











