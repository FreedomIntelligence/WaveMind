# Datasets

## Supported Datasets

### Brain Cognition Datasets (Image-EEG Alignment)

These datasets capture neural responses to external visual stimuli, representing higher-order cognitive processes:

- **THING-EEG**: Visual object recognition dataset
  - **Subjects**: 10 participants performing rapid serial visual presentation (RSVP) task
  - **Categories**: 1,573 training object categories (closed-set) + 200 zero-shot test categories
  - **Acquisition**: 64-channel EEG at 1000Hz sampling rate
  - **Paradigm**: RSVP with orthogonal target detection task
  - **Training data**: 16,540 image conditions (4 repetitions each)
  - **Test data**: 200 image conditions (80 repetitions each)
  - **Total trials**: 82,160 image presentations
  - **Use case**: Visual-stimulus interpretation

- **ImageNet-EEG**: Image category classification
  - **Subjects**: 6 participants viewing ImageNet images
  - **Categories**: 40 object categories
  - **Images**: 50 images per category (2,000 total images)
  - **Acquisition**: 128-channel EEG at 1kHz sampling rate
  - **Segment length**: 0.5-second (temporally repeated to 1-second for standardization)
  - **Use case**: Visual-stimulus interpretation

### Brain State Datasets (Text-EEG Alignment)

These datasets capture internal neural states at rest or during specific tasks, annotated with clinical/emotional labels:

- **SEED**: Emotion recognition dataset
  - **Subjects**: 15 participants
  - **Stimuli**: 15 film clips (~4 minutes each) designed to elicit emotions
  - **Categories**: 3 emotions (Positive, Neutral, Negative)
  - **Acquisition**: 62-channel ESI NeuroScan system at 1kHz
  - **Preprocessing**: Downsampled to 200Hz, 0-75Hz bandpass filtering
  - **Use case**: Emotion recognition from brain states

- **TUAB**: EEG abnormality detection
  - **Source**: Subset of Temple University Hospital (TUH) EEG Corpus
  - **Categories**: 2 classes (Normal, Abnormal)
  - **Acquisition**: 22 channels at 250Hz sampling rate
  - **Segment length**: 10-second segments (split into 1-second samples)
  - **Annotations**: Clinical doctor assessments
  - **Use case**: Abnormality detection in clinical EEG

- **TUEV**: EEG event classification
  - **Source**: Subset of TUH EEG Corpus
  - **Samples**: 16,986 annotated EEG segments from 10,874 subjects
  - **Categories**: 6 event types
    - **Signal classes**:
      - SPSW (Spike and Sharp Wave): Abrupt, high-amplitude transients
      - GPED (Generalized Periodic Epileptiform Discharges): Bilateral synchronous spikes
      - PLED (Periodic Lateralized Epileptiform Discharges): Unilateral periodic discharges
    - **Noise classes**:
      - EYEM (Eye Movement): Frontal slow waves synchronized with blinks
      - ARTF (Artifact): Non-physiological signals
      - BCKG (Background): Stable, symmetric rhythms
  - **Use case**: Event detection and classification

## Unified Preprocessing Format

All datasets are standardized to:
- **Channels**: 32 channels (10-20 system montage)
- **Sampling rate**: 512Hz
- **Duration**: 1-second segments (512 time points)
- **Format**: R^(32×512) spatiotemporal tensors
- **Storage**: HDF5 format at `data/Total/data_label.h5`

## Training Data Statistics

| Dataset | Stage I (Alignment) | Stage II (Cold-Start) | Stage III (Instruction) | Downstream Task |
|---------|---------------------|----------------------|------------------------|-----------------|
| **THING-EEG** | 528K | - | 153K | Visual-Stimulus Interpretation |
| **ImageNet-EEG** | 8K | - | 103K | Visual-Stimulus Interpretation |
| **SEED** | 122K | - | 28K | Emotion Recognition |
| **TUEV** | 113K | - | 28K | Event Detection |
| **TUAB** | 548K | - | 39K | Abnormality Detection |
| **LLaVA-Pretrain** | - | 558K | - | Adapter Initialization |
| **Total** | **1.3M** | **558K** | **338K** | - |

**Stage I - Dual-Representation Alignment**: Trains the ATMM encoder to align EEG features with CLIP space using 1.3M EEG samples paired with images (Brain Cognition) or text (Brain State).

**Stage II - Cold-Start**: Initializes the modality adapter using 558K image-caption pairs from LLaVA-Pretrain to bridge CLIP and language spaces.

**Stage III - Instruction Tuning**: Fine-tunes the complete model using 338K instruction-response pairs from WaveMind-Instruct, enabling conversational capabilities.

## Data Preprocessing

### Preprocessing Steps

1. **Montage Unification** (32 standard channels, 10-20 system):
   ```
   Channels: ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2',
              'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP5', 'CP1', 'CP2', 'CP6',
              'P7', 'P3', 'Pz', 'P4', 'P8', 'POz', 'O1', 'Oz', 'O2', 'AFz',
              'CPz', 'FCz']
   ```
   - Identifies and retains valid EEG channels from raw data
   - For missing standard channels: Uses linear interpolation from neighboring electrodes
   - Ensures consistent channel space across all datasets

2. **Resampling**: Target 512Hz via linear interpolation (handles 200Hz - 1kHz)

3. **Fixed-Duration Segmentation**: Target 1 second (512 time points). Shorter segments use temporal repetition; longer segments are partitioned.

4. **Output**: R^(32×512) tensors stored in HDF5 at `data/Total/data_label.h5`

### Data Augmentation (Stage I Training)

- **Global Z-score normalization**: Normalize across all channels and time points
- **Channel-wise Z-score normalization**: Independent normalization per channel
- **Global/Channel-wise standard deviation scaling**: Preserve mean while standardizing variance
- **Amplitude fluctuation**: ±10% random amplitude variation

### Preprocessing Scripts

```bash
# Automated preprocessing of all datasets
./data/preprocess_wavemind_dataset.sh

# Process specific dataset
./data/preprocess_wavemind_dataset.sh SEED

# Show available datasets
./data/preprocess_wavemind_dataset.sh --list

# Skip dependency checks
./data/preprocess_wavemind_dataset.sh --skip-check
```

**Critical Files**:
- `data/Total/data_label.h5`: Preprocessed EEG segments (generated)
- `data/Total/CLIP_groundTruth/`: Aligned features for RAG (generated)
  - Image features from CLIP-ViT-large-patch14-336
  - Text features from CLIP-BERT
