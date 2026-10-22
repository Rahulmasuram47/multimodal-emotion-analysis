# CREMA-D Dev-Only Baseline Evaluation Report

## Experiment Metadata
- **Active Voice Model**: `emotion2vec/emotion2vec_plus_base` (emotion2vec)
- **Active Backend Selection**: Configured for emotion2vec and funasr is loadable.
- **Alternative Backend (emotion2vec)**: Not installed
- **Evaluation Split**: DEV actors only (N = 9 actors, 400 clips)
- **Random Seed**: 42
- **Pre-processing**: Mono 16 kHz resampled; feature extractor padding
- **Dataset License**: Dataset license file is missing from data/external_datasets/crema_d/.
(Note: CREMA-D is publicly distributed under the Open Database License (ODbL) v1.0 / Open Data Commons Open Database License by the CREMA-D consortium).

## Aggregate Metrics
- **Accuracy**: `94.00%` (95% Actor-Cluster CI: `[90.06%, 97.14%]`)
- **Unweighted Average Recall (UAR)**: `95.19%` (95% Actor-Cluster CI: `[90.09%, 97.14%]`)
- **Macro-F1**: `95.22%`
- **Chance Level**: `25.00%` (1/4 classes)

## Per-Class Performance
| Emotion Class | Support | Recall | Precision | F1 | Highlight |
|---|---|---|---|---|---|
| **neutral** | 100 | 1.0000 | 0.8696 | 0.9302 | ⭐ **Key Metric** |
| **happy** | 99 | 0.9596 | 0.9794 | 0.9694 |  |
| **angry** | 97 | 0.9794 | 1.0000 | 0.9896 |  |
| **sad** | 99 | 0.8687 | 0.9773 | 0.9198 | ⭐ **Key Metric** |

## Confusion Matrix
| True \ Pred | neutral | happy | angry | sad |
|---|---|---|---|---|
| **neutral** | 100 | 0 | 0 | 0 |
| **happy** | 3 | 95 | 0 | 1 |
| **angry** | 0 | 1 | 95 | 1 |
| **sad** | 12 | 1 | 0 | 86 |

> [!IMPORTANT]
> **Test Split Isolation**: Test actor metrics were strictly excluded from this report. Test evaluation is locked until `evaluation/results/head_decision.json` is generated.
