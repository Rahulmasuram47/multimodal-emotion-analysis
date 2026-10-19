# CREMA-D Dev-Only Baseline Evaluation Report

## Experiment Metadata
- **Active Voice Model**: `emotion2vec/emotion2vec_plus_base` (emotion2vec)
- **Active Backend Selection**: Configured for emotion2vec and funasr is loadable.
- **Alternative Backend (emotion2vec)**: Not installed
- **Evaluation Split**: DEV actors only (N = 1 actors, 20 clips)
- **Random Seed**: 42
- **Pre-processing**: Mono 16 kHz resampled; feature extractor padding
- **Dataset License**: Dataset license file is missing from data/external_datasets/crema_d/.
(Note: CREMA-D is publicly distributed under the Open Database License (ODbL) v1.0 / Open Data Commons Open Database License by the CREMA-D consortium).

## Aggregate Metrics
- **Accuracy**: `70.00%` (95% Actor-Cluster CI: `[70.00%, 70.00%]`)
- **Unweighted Average Recall (UAR)**: `80.00%` (95% Actor-Cluster CI: `[70.00%, 70.00%]`)
- **Macro-F1**: `75.29%`
- **Chance Level**: `25.00%` (1/4 classes)

## Per-Class Performance
| Emotion Class | Support | Recall | Precision | F1 | Highlight |
|---|---|---|---|---|---|
| **neutral** | 5 | 1.0000 | 0.6250 | 0.7692 | ⭐ **Key Metric** |
| **happy** | 5 | 1.0000 | 0.8333 | 0.9091 |  |
| **angry** | 3 | 1.0000 | 1.0000 | 1.0000 |  |
| **sad** | 5 | 0.2000 | 1.0000 | 0.3333 | ⭐ **Key Metric** |

## Confusion Matrix
| True \ Pred | neutral | happy | angry | sad |
|---|---|---|---|---|
| **neutral** | 5 | 0 | 0 | 0 |
| **happy** | 0 | 5 | 0 | 0 |
| **angry** | 0 | 0 | 3 | 0 |
| **sad** | 3 | 1 | 0 | 1 |

> [!IMPORTANT]
> **Test Split Isolation**: Test actor metrics were strictly excluded from this report. Test evaluation is locked until `evaluation/results/head_decision.json` is generated.
