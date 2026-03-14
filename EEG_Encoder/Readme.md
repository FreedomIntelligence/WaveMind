# EEG Encoder

WaveMind's EEG encoder training module for aligning EEG signals with CLIP semantic space through contrastive learning and supervised classification.

## Quick Start

```bash
# Ensure environment is set up
export WaveMind_ROOT_PATH_=/path/to/WaveMind

# Basic CLIP training
python EEG_Encoder/run_CLIPtraining.py --config-name=train_atms
```

## Training Modes

**Pure CLIP Training** (default):
```bash
python EEG_Encoder/run_CLIPtraining.py --config-name=train_atms
```

**Joint CLIP + Classifier** (lambda=0.5):
```bash
python EEG_Encoder/run_CLIPtraining.py --config-name=train_classifier
```

**Pure Classifier Training** (lambda=0.0):
```bash
python EEG_Encoder/run_CLIPtraining.py --config-name=train_classifier_only
```

## Available Configuration Presets

All presets are in `EEG_Encoder/examples/`. See YAML files for detailed parameters:
- `train_atms.yaml` - CLIP training (default)
- `train_classifier.yaml` - Joint CLIP + Classifier (lambda=0.5)
- `train_classifier_only.yaml` - Pure supervised classifier (lambda=0.0)
- `eval_sd.yaml` - Subject-dependent evaluation
- `eval_si.yaml` - Subject-independent evaluation
- `advanced_shm.yaml` - High-performance with shared memory
- `base.yaml` - Full parameter reference

## Common Usage Examples

```bash
# Train on specific dataset
python EEG_Encoder/run_CLIPtraining.py --config-name=train_atms experiment.datasets=[TUEV]

# Use different model
python EEG_Encoder/run_CLIPtraining.py --config-name=train_atms experiment.models=[EEGITNet]

# Multi-GPU training
python EEG_Encoder/run_CLIPtraining.py --config-name=train_atms experiment.gpu_number=['0','1','2']

# Adjust classifier lambda (80% CLIP, 20% classifier)
python EEG_Encoder/run_CLIPtraining.py --config-name=train_classifier classifier.lambda_clip=0.8

# Evaluation with checkpoint
python EEG_Encoder/run_CLIPtraining.py --config-name=eval_sd \
    advanced.model_checkpoint_name=/path/to/model.pth

# Advanced: shared memory + dynamic sampling
python EEG_Encoder/run_CLIPtraining.py --config-name=advanced_shm \
    training.DEFAULT_NUM_WORKERS=64
```

## Available Models

ATMSmodify (default), ATMS, NICE, EEGITNet, EEGConformer, ShallowFBCSPNet, CBraMod, NeuroLM-B/L, MLP

See `examples/base.yaml` for complete list and model-specific parameters.

## Output

Checkpoints saved to: `EEG_Encoder/Resource/Checkpoint/ALL/`

Naming convention: `{Model}_{Dataset}_{Timestamp}.pth`
- CLIP-only: No suffix
- Pure classifier: `_CLS` suffix
- Joint training: `_JOINT{lambda}` suffix (e.g., `_JOINT50` for lambda=0.5)

## Documentation

- **Configuration details**: See YAML files in `EEG_Encoder/examples/`
- **Project overview**: `CLAUDE.md` in project root
- **Full pipeline**: WaveMind documentation
