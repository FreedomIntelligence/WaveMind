# RAG System

WaveMind enhances LLM responses using a CLIP-based RAG system that retrieves semantically similar labels from a pre-computed database of 1,824 feature embeddings across 5 datasets. This provides the LLM with task-relevant context before processing the EEG signal.

## Architecture

**Database Structure**:
- **Total entries**: 1,824 pre-computed CLIP embeddings
- **Feature dimension**: 768 (CLIP ViT-L/14-336 text embeddings)
- **Storage location**: `data/Total/CLIP_groundTruth/`
  - Features: `*.npy` files (float16)
  - Labels: `*.pkl` files (Python lists)
- **Datasets included**:
  - ImageNetEEG: 40 visual categories
  - SEED: 3 emotion states (positive, negative, neutral)
  - THING-EEG: 1,854 visual concepts (zero-shot + close-set)
  - TUAB: 2 classes (normal, abnormal EEG)
  - TUEV: 6 event types (SPSW, GPED, PLED, EYEM, ARTF, BCKG)

## Caption Templates

**ImageNetEEG**: Raw class names (e.g., "canoe", "broom", "daisy")

**SEED**: Contextual emotion descriptions
- Template: `"Emotion Recognition Task. This person is watching {label} video. + {emotional_context}"`
- Emotional context examples:
  - positive: "+ It shows a joy, laughter, happiness, bright colors, celebration, warmth atmosphere"
  - negative: "+ It shows a sadness, loneliness, tears, grief, sorrowful atmosphere"
  - neutral: "+ It shows a calm, peaceful, quiet, everyday activity, serene landscape"

**TUAB**:
- Normal: "Normal EEG: Characterized by rhythmic patterns with stable amplitudes. The brain wave is normal and stable."
- Abnormal: "Abnormal EEG: Exhibits deviations such as slowed/fast rhythms, asymmetry, or irregular amplitudes. May include spikes, sharp waves, or spike-wave complexes."

**TUEV**: Medical event descriptions for each of the 6 event types.

## Retrieval Algorithm

**Two-stage retrieval process**:
1. **Category-wise sampling**: Sample top items from each of 5 dataset types
2. **Visual stimuli expansion**: Fill remaining slots with additional ImageNetEEG items
3. **Top-k calculation**: Default 25% of database (min(len(labels)*0.25, 420)) ≈ 420 items

**Similarity metric**: Cosine similarity (dot product of L2-normalized vectors)

**Retrieval time**: ~25ms per query (CPU-only, no GPU required)

## Data Generation

```bash
# Generate CLIP ground truth features from preprocessed data
python data/create_dataset_pkl.py

# Outputs:
# - data/Total/CLIP_groundTruth/ImageNetEEG.npy (40 embeddings)
# - data/Total/CLIP_groundTruth/SEED.npy (3 embeddings)
# - data/Total/CLIP_groundTruth/THING.npy (1854 embeddings)
# - data/Total/CLIP_groundTruth/TUAB.npy (2 embeddings)
# - data/Total/CLIP_groundTruth/TUEV.npy (6 embeddings)
# - Corresponding .pkl label files for each
```

**Requirements**: Must have `data/Total/data_label.h5` preprocessed first.

## Integration Points

**Training (Stages 1-3)**: RAG is NOT used during training.

**Inference**: Enable with `RAG=True` parameter in `WaveMind_inference()`.

**Evaluation**: Enable with `--modilities RAG+modility` flag.

Available modality modes:
- `only_RAG`: Text context only, no EEG embeddings
- `only_modility`: EEG embeddings only, no RAG text
- `RAG+modility`: Both (recommended, +2-5% accuracy)
- `random`: Random baseline

## Prompt Format

Retrieved labels are formatted and appended to the user's question. The `<image>` token is replaced with EEG embeddings during model forward pass.

## Performance Impact

- **Accuracy boost**: +2-5% on classification tasks (SEED, TUAB, TUEV)
- **No GPU memory overhead**: Database loaded to CPU, similarity search on CPU
- **Inference latency**: +25ms per query
- **Initialization time**: ~1 second to load database at model startup

## Configuration

See `EEGLLM/config.py` for `InferenceConfig` with `enable_rag`, `rag_topk`, and `rag_random_variance` options.

## Implementation Details

**Core class**: `DBsearch` in `EEGLLM/LLaVA/llava/model/multimodal_encoder/eeg_encoder.py:186`

**Search method**: `get_search_result_from_EEG()` at line 456:
1. Encode EEG through NeuroTower → (768-dim embedding)
2. L2 normalize EEG embedding
3. Compute cosine similarity with all 1,824 database embeddings
4. Select top-k results with category-wise sampling
5. Format as "If task is X: label1 | label2 | ..." text
6. Return formatted string to be appended to prompt
