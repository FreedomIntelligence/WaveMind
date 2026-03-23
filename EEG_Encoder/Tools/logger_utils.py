"""
Logger utilities for WaveMind training
Supports CometML and SwanLab experiment tracking
"""

from typing import Optional
from omegaconf import DictConfig
from lightning.pytorch.utilities.rank_zero import rank_zero_info


def get_experiment_logger(cfg: DictConfig) -> Optional[object]:
    """
    Create experiment logger based on configuration

    Args:
        cfg: Hydra configuration containing logger settings

    Returns:
        Logger instance (CometLogger/SwanLabLogger) or None

    Supported loggers:
        - "comet": CometML logger (requires comet-ml package)
        - "swanlab": SwanLab logger (requires swanlab package)
        - "none" or None: No experiment tracking

    Configuration example:
        logger:
          type: "comet"
          experiment_name: "test_run_001"
    """

    # Check if logger is disabled
    if not hasattr(cfg, 'logger') or cfg.logger.type == "none":
        rank_zero_info("Experiment tracking disabled (logger.type='none')")
        return None

    logger_type = cfg.logger.type.lower()
    project_name = "wavemind"  # Fixed project name (lowercase)

    # Auto-generate experiment name if not specified
    experiment_name = cfg.logger.get('experiment_name', None)
    if experiment_name is None:
        import time
        # Default: all_models + "_" + all_datasets + "_" + timestamp
        # Example: "ATMSmodify_NICE_ImageNetEEG_SEED_1736511234"
        models_str = "_".join(cfg.experiment.models) if cfg.experiment.models else "unknown_model"
        datasets_str = "_".join(cfg.experiment.datasets) if cfg.experiment.datasets else "unknown_dataset"
        timestamp = int(time.time())
        experiment_name = f"{models_str}_{datasets_str}_{timestamp}"
        rank_zero_info(f"Auto-generated experiment name: {experiment_name}")

    if logger_type == "comet":
        return _create_comet_logger(cfg, project_name, experiment_name)
    elif logger_type == "swanlab":
        return _create_swanlab_logger(cfg, project_name, experiment_name)
    else:
        raise ValueError(
            f"Unknown logger type: '{logger_type}'. "
            f"Supported types: 'comet', 'swanlab', 'none'"
        )


def _create_comet_logger(cfg, project_name, experiment_name):
    """Create CometML logger with unified log directory"""
    try:
        from lightning.pytorch.loggers import CometLogger
    except ImportError:
        raise ImportError(
            "CometML logger requires 'comet-ml' package. "
            "Install with: pip install comet-ml"
        )

    # Same priority logic as SwanLab
    if hasattr(cfg.logger, 'save_dir') and cfg.logger.save_dir is not None:
        log_dir = cfg.logger.save_dir
    else:
        log_dir = cfg.paths.LOGS_DIR

    import os
    os.makedirs(log_dir, exist_ok=True)

    # Prepare CometLogger kwargs
    kwargs = {
        'project_name': project_name,
        'save_dir': log_dir,  # Updated from hardcoded '.'
    }

    # Add experiment name if specified
    if experiment_name:
        kwargs['experiment_name'] = experiment_name

    comet_logger = CometLogger(**kwargs)

    rank_zero_info(f"CometML logger enabled: project='{project_name}', save_dir='{log_dir}'")

    return comet_logger


def _convert_cfg_to_dict(cfg: DictConfig) -> dict:
    """
    Convert OmegaConf DictConfig to a plain Python dictionary

    Args:
        cfg: OmegaConf configuration object

    Returns:
        Plain Python dictionary
    """
    from omegaconf import OmegaConf

    # Convert DictConfig to nested dict, resolve=True resolves references
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    return cfg_dict


def _create_swanlab_logger(cfg, project_name, experiment_name):
    """Create SwanLab logger with unified log directory"""
    try:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
    except ImportError:
        raise ImportError(
            "SwanLab logger requires 'swanlab' package. "
            "Install with: pip install swanlab"
        )

    # Determine log directory with priority:
    # 1. logger.save_dir (per-experiment override)
    # 2. paths.LOGS_DIR (always available from PathConfig)
    if hasattr(cfg.logger, 'save_dir') and cfg.logger.save_dir is not None:
        log_dir = cfg.logger.save_dir
    else:
        log_dir = cfg.paths.LOGS_DIR

    # Ensure directory exists
    import os
    os.makedirs(log_dir, exist_ok=True)

    # Convert cfg to dict for SwanLab config
    cfg_dict = _convert_cfg_to_dict(cfg)

    # Prepare SwanLabLogger kwargs
    kwargs = {
        'project': project_name,
        'logdir': log_dir,
        'config': cfg_dict,
    }

    # Add experiment name if specified
    if experiment_name:
        kwargs['experiment_name'] = experiment_name

    swanlab_logger = SwanLabLogger(**kwargs)

    rank_zero_info(f"SwanLab logger enabled: project='{project_name}', logdir='{log_dir}'")

    return swanlab_logger
