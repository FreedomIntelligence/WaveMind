# WaveMind-Bench

WaveMind-Bench is a comprehensive benchmark for evaluating chat-like EEG-MLLMs, consisting of 36K multiple-choice questions sampled from test sets.

## Benchmark Specifications

```bash
# Requires data_label.h5 to be generated first
bash ./Data_Engineering/Script/WaveMind_Bench/construct_WaveMind_bench.sh
```

**Dataset Coverage**:

| Dataset | k-way Options | Samples per Option | Total Samples |
|---------|---------------|-------------------|---------------|
| **THING-EEG** | 2, 4, 40 | 3,000 each | 9,000 |
| **ImageNet-EEG** | 2, 4, 40 | 3,000 each | 9,000 |
| **TUEV** | 2, 4, 6 | 3,000 each | 9,000 |
| **TUAB** | 2 | 3,000 | 3,000 |
| **SEED** | 2, 3 | 3,000 each | 6,000 |
| **Total** | - | - | **36,000** |

**Question Format**:
- Each question consists of 1 correct answer + (k-1) randomly selected incorrect options
- THING-EEG uses k=40 for 40-way (reduced from 200 for practicality in zero-shot setting)
- Model outputs single letter (A, B, C, etc.) without explanation

**Example MCQ**:
```
What emotional moment is captured in this man's EEG?
Choose one letter.
(A) Negative  (B) Neutral  (C) Positive
```

## Evaluation Metrics

**Classification Metrics**:
- **Weighted K-way accuracy**: Primary metric for classification ability
- Each k-way accuracy independently measures performance with k candidates

**Natural Language Generation Metrics** (for open-ended responses):

*Model-free*:
- BLEU-1/2: Lexical overlap at 1-gram/2-gram level
- METEOR: Considers synonyms and stemming
- ROUGE-1/2: Recall-oriented n-gram overlap

*Model-based*:
- MiniLM-L12-v2: Semantic similarity score
- GPT-4o: Matching score (whether response contains correct category)

## Performance Highlights (Subject-Dependent, with RAG)

| Dataset | Task | k-way | Accuracy |
|---------|------|-------|----------|
| TUEV | Event Detection | 6 | 0.904 |
| TUAB | Abnormality Detection | 2 | 0.741 |
| SEED | Emotion Recognition | 3 | 0.529 |
| ImageNet-EEG | Visual-Stimulus | 40 | 0.603 |
| THING-EEG | Visual-Stimulus | 40 | 0.250 |

**Availability**: https://huggingface.co/datasets/CocoNutZENG/WaveMind_Bench
