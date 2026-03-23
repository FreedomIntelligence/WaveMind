# Training Pipeline

WaveMind uses a progressive three-stage training approach. See [CLAUDE.md](../CLAUDE.md) for an overview.

## Stage 1: Dual-Representation Alignment (EEG Encoder Training)

Train EEG encoder to align features with CLIP space using contrastive learning on 1.3M samples.

**IMPORTANT**: Must specify a configuration file via `--config-name` parameter.

```bash
# Basic training with ATMSmodify (uses train_atms preset)
python EEG_Encoder/run_CLIPtraining.py --config-name=train_atms

# Training with custom overrides
python EEG_Encoder/run_CLIPtraining.py --config-name=xxx
```

**Available config presets** (in `EEG_Encoder/examples/`):
- `base.yaml`: Default configuration
- `train_atms.yaml`: Quick-start for ATMSmodify training
- `eval_sd.yaml`: Subject Dependent evaluation
- `eval_si.yaml`: Subject Independent evaluation
- `advanced_shm.yaml`: Advanced shared memory configuration
- `experiment_full.yaml`: Full dataset experiment

**Available Models**: MLP, ATMS, ShallowFBCSPNet, channelNet, NICE, ATMSmodify (primary), EEGITNet, CBraMod, NeuroLM-B, NeuroLM-L

**Output**: Trained encoder checkpoint saved to `EEG_Encoder/Resource/Checkpoint/ALL/`

### Subject Dependent (SD) vs Subject Independent (SI) Evaluation

WaveMind supports two evaluation modes:

| Mode | Description | Data Split Used | Use Case |
|------|-------------|-----------------|----------|
| **SD** (Subject Dependent) | Evaluate on same subjects as training | `val_dataset` = `*_test` split | Intra-subject generalization |
| **SI** (Subject Independent) | Evaluate on held-out subjects/data | `test_dataset` = `*_cross` split | Cross-subject generalization |

**Data Splits in HDF5**:
- `*_train`: Training data
- `*_test`: Test data (used as `val_dataset` for SD evaluation)
- `*_cross` or `*_test_`: Cross-validation data (used as `test_dataset` for SI evaluation)

**Note**: Not all datasets have a `*_cross` split. If absent, `test_dataset` will be `None`.

**Evaluation Configurations**:
```bash
# Subject Dependent (uses test split as validation)
python EEG_Encoder/run_CLIPtraining.py --config-name=eval_sd

# Subject Independent (uses cross split as test)
python EEG_Encoder/run_CLIPtraining.py --config-name=eval_si
```

## Stage 2: Cold-Start Training (Modality Adapter Pretraining)

Pretrain the cross-modal adapter on LLaVA-Pretrain dataset to bridge CLIP and language spaces.

```bash
# Requires LLaVA-Pretrain-558k dataset
# Download from: https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
# Place in: EEGLLM/LLaVA/playground/data/LLaVA-Pretrain/

bash ./EEGLLM/examples/stage2_pretrain/pretrain.sh
```

**Key Parameters**:
- Uses DeepSpeed with zero2 optimization
- Freezes EEG encoder (`freeze_neuro_tower=True`)
- Only tunes the MLP adapter (`tune_mm_mlp_adapter=True`)
- Single epoch training

## Stage 3: EEG Instruction Tuning (LoRA Fine-tuning)

Fine-tune the complete model on EEG instruction data using LoRA.

```bash
# Edit the script to set data_path and origin_load_path
bash ./EEGLLM/examples/stage3_finetune/finetune_lora_eeg.sh
```

**Key Configuration**:
- LoRA enabled (`lora_enable=True`, `lora_r=8`, `lora_alpha=128`)
- EEG encoder frozen (`freeze_neuro_tower=True`)
- Adapter frozen (`tune_mm_mlp_adapter=False`)
- Training mode: `mela` (for EEG instruction data)
