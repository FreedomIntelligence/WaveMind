# Architecture

## EEG Encoding and LLM Injection Pipeline

WaveMind treats EEG signals as "visual tokens" within the LLM's embedding space, enabling the language model to process brain activity alongside text.

### Technical Implementation Files

| Component | File | Key Lines |
|-----------|------|-----------|
| **NeuroTower** | `EEGLLM/LLaVA/llava/model/multimodal_encoder/eeg_encoder.py` | 21-86 |
| **encode_eegs()** | `EEGLLM/LLaVA/llava/model/llava_arch.py` | 238-246 |
| **prepare_inputs_labels_for_multimodal()** | `EEGLLM/LLaVA/llava/model/llava_arch.py` | 297-593 |
| **Token replacement logic** | `EEGLLM/LLaVA/llava/model/llava_arch.py` | 421-517 |
| **mm_projector builder** | `EEGLLM/LLaVA/llava/model/multimodal_projector/builder.py` | 33-52 |
| **EEG encoder models** | `EEG_Encoder/Model/baseModel.py` | 1-710 |
| **Model selection** | `EEG_Encoder/Model/baseModel.py` | 56-234 |

### Differences from Vision Token Processing

| Aspect | Vision Tokens | EEG Tokens |
|--------|--------------|------------|
| **Input shape** | (3, 336, 336) RGB | (32, 512) time-series |
| **Encoder** | CLIP ViT-L-14-336 | ATMSmodify/NICE/etc. |
| **Feature dim** | 1024 (vision tower) | 768 (neuro tower) |
| **Patch tokens** | 576 (24×24 grid) | 1 (pooled output) |
| **Position encoding** | 2D positional | 1D temporal |
| **Preprocessing** | Resize + normalize | Bandpass filter + normalize |

**Key difference**: EEG produces **single pooled embedding** (1×4096), while vision produces **sequence of patch embeddings** (576×4096). This design reflects that EEG is typically interpreted holistically rather than spatially.

### Debugging and Validation

**Check encoder output dimension**:
```python
neuro_tower = model.get_model().get_neuro_tower()
test_eeg = torch.randn(1, 32, 512)
output = neuro_tower(test_eeg)
print(output.shape)  # Should be (1, 768)
```

**Check mm_projector output**:
```python
eeg_features = model.encode_eegs(test_eeg)
print(eeg_features.shape)  # Should be (1, 4096)
```

**Verify CLIP alignment** (optional):
```python
# EEG features should be on unit hypersphere after normalization
eeg_norm = torch.norm(eeg_features, p=2, dim=-1)
assert torch.allclose(eeg_norm, torch.ones_like(eeg_norm), atol=1e-5)
```

## Brain Cognition vs Brain State

### Brain Cognition (Image-EEG Pairs)

**Definition**: Neural responses to external visual stimuli, representing higher-order cognitive processes like visual perception and object recognition.

**Characteristics**:
- Datasets: THING-EEG, ImageNet-EEG
- Experimental paradigm: Participants view images/visual stimuli
- EEG captures: Active cognitive processing of external inputs
- Paired modality: Images → CLIP-ViT features (768-dim)
- Alignment target: Image embeddings from CLIP visual encoder

**Neuroscience Interpretation**:
- Reflects how the brain encodes and processes visual information
- Captures attention, recognition, and semantic understanding
- Related to language comprehension and higher-order cognition

### Brain State (Text-EEG Pairs)

**Definition**: Internal neural states at rest or during specific tasks, annotated with clinical/emotional descriptors.

**Characteristics**:
- Datasets: SEED (emotion), TUAB (abnormality), TUEV (events)
- Experimental paradigm: Resting-state or task-based recording
- EEG captures: Intrinsic neural patterns and states
- Paired modality: Text annotations → CLIP-BERT features (768-dim)
- Alignment target: Text embeddings from CLIP language encoder

**Neuroscience Interpretation**:
- Reflects internal emotional states, pathological patterns, or artifacts
- Captures baseline neural dynamics and state-dependent activity
- Clinically relevant for diagnosis and assessment

### Complementary Relationship

The paper's pilot study (Table 2) demonstrates that these two modality types are **complementary**:

| Training Setup | THING-EEG (2-way) | TUEV (6-way) |
|----------------|-------------------|--------------|
| Image-EEG only | 0.648 | - |
| Text-EEG only | - | 0.742 |
| **Fusion of Both** | **0.671 (+0.023)** | **0.788 (+0.046)** |

**Key Insight**: Training with both modalities simultaneously improves performance on both tasks, suggesting that:
- Brain Cognition and Brain State represent different yet related neural mechanisms
- The brain maintains cognitive clarity in dynamic environments while remaining sensitive to internal states
- Multi-modal training enables better generalization across diverse EEG interpretation tasks

### Dual-Supervision Alignment

WaveMind's Stage I training uses both modalities via dual-supervision loss:

```
L = λL_img + (1-λ)L_txt
```

Where:
- `L_img`: InfoNCE contrastive loss for Image-EEG pairs (Brain Cognition)
- `L_txt`: InfoNCE contrastive loss for Text-EEG pairs (Brain State)
- Both losses align EEG features to the same CLIP semantic space

This unified approach allows WaveMind to:
1. Leverage diverse upstream training data (1.3M total samples)
2. Interpret EEG across multiple downstream tasks without architectural changes
3. Support flexible conversational interactions about both cognitive and state-related brain activity
