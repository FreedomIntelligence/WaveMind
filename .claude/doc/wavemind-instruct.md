# WaveMind-Instruct Dataset

WaveMind-Instruct is the first open-source EEG instruction-tuning dataset, comprising 338K instruction-response pairs designed to enable conversational EEG interpretation.

## Dataset Composition

| Dataset | Description QA | Open-ended QA | MCQ | Consultation | Analysis | Total |
|---------|----------------|---------------|-----|--------------|----------|-------|
| **THING-EEG** | 54,303 | 22,659 | 76,962 | ✗ | ✓ | 153,924 |
| **ImageNet-EEG** | 28,596 | 54,919 | 19,628 | ✗ | ✓ | 103,143 |
| **SEED** | - | 17,031 | 10,270 | ✗ | ✓ | 28,301 |
| **TUEV** | - | 9,585 | 7,541 | ✓ | ✓ | 17,126 |
| **TUAB** | - | 26,437 | 12,760 | ✓ | ✓ | 39,197 |
| **Empty-EEG Rejection** | - | - | - | - | - | 1,676 |
| **Total** | **82,899** | **130,631** | **127,161** | - | - | **338,952** |

## Instruction Types

1. **Description**: Direct EEG-to-text descriptions (Brain Cognition only)
   - Generated from image captions using Qwen2.5-VL
   - Image-related keywords replaced with EEG terminology

2. **Question-Answer (QA)**:
   - **Brain Cognition**: Transformed from image captions into diverse Q&A pairs
   - **Brain State**: Generated from manually curated factual definitions using LLM synthesis
   - Both closed-ended and open-ended questions

3. **Multiple-Choice Questions (MCQ)**:
   - Number of options matches actual class count in each dataset
   - Used for classification-style tasks

## Dialogue Scenarios

- **Consultation** (TUEV, TUAB only):
  - Doctor-patient roleplay for clinical diagnosis
  - Covers event detection and abnormality detection
  - More colloquial medical conversation style

- **Analysis** (All datasets):
  - General scientific analysis without specific roles
  - Covers all four downstream tasks
  - More neutral technical language

## Quality Control

- **Image captions**: 0.945±0.023 acceptance rate (3 human reviewers)
- **Clinical definitions**: 0.966±0.013 acceptance rate (3 human reviewers)
- **Deduplication**: 2-gram and ROUGE-L filtering (retained ~70% of synthesized data)
- **Hallucination prevention**: 1,676 empty-EEG rejection samples included
- **Diversity**: Explicit prohibition of EEG-specific details during synthesis

## Instruction Construction

**Brain Cognition (Image-EEG)**:
1. Qwen2.5-VL generates image captions (5 per image)
2. Regular expression removes image-related keywords
3. Qwen2.5-Instruct transforms captions into diverse QA pairs

**Brain State (Text-EEG)**:
1. Manual collection of factual definitions for each annotation
2. Human refinement into seed QA instructions
3. LLM rewriting to generate diverse instructions matching tone/semantics
