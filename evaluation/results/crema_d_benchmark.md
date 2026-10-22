# CREMA-D One-Shot Test-Split Evaluation Benchmark

## Evaluation Protocol & Caveats
- **Nature of Corpus**: **acted speech, actor-disjoint split, not natural speech**
- **Evaluation Partition**: Strictly held-out test actors (N = 31 actors, 1673 clips)
- **Active Model**: `emotion2vec/emotion2vec_plus_base` (emotion2vec)
- **Alternative Backend**: SUPERB comparison disabled (run with --compare-superb to include).
- **Random Seed**: 42
- **Pre-processing**: Mono 16 kHz resampled, feature extractor padding
- **Excluded Classes**: Raw CREMA-D `DIS` and `FEA` mapped to unmodeled/other; evaluated on 4 canonical classes (['neutral', 'happy', 'angry', 'sad'])
- **Dataset License**: Dataset license file is missing from data/external_datasets/crema_d/.
(Note: CREMA-D is publicly distributed under the Open Database License (ODbL) v1.0 / Open Data Commons Open Database License by the CREMA-D consortium).
- **Sampling Budget**: Full test partition evaluated without class capping.

## Baseline Test Metrics (emotion2vec: emotion2vec/emotion2vec_plus_base)
- **Accuracy**: `93.31%` (95% Actor-Cluster CI: `[91.63%, 94.68%]`)
- **Unweighted Average Recall (UAR)**: `94.88%` (95% Actor-Cluster CI: `[91.86%, 94.86%]`)
- **Macro-F1**: `94.73%`
- **Chance Level**: `25.00%` (1/4 classes)

## Per-Class Performance (Baseline)
| Emotion Class | Support | Recall | Precision | F1 | Highlight |
|---|---|---|---|---|---|
| **neutral** | 371 | 0.9838 | 0.8528 | 0.9136 | ⭐ **Key Metric** |
| **happy** | 431 | 0.9443 | 0.9784 | 0.9610 |  |
| **angry** | 431 | 0.9907 | 0.9749 | 0.9827 |  |
| **sad** | 413 | 0.8765 | 0.9945 | 0.9318 | ⭐ **Key Metric** |

## Confusion Matrix (Baseline)
| True \ Pred | neutral | happy | angry | sad |
|---|---|---|---|---|
| **neutral** | 365 | 4 | 1 | 1 |
| **happy** | 13 | 407 | 10 | 1 |
| **angry** | 2 | 2 | 427 | 0 |
| **sad** | 48 | 3 | 0 | 362 |


## Candidate Head Status
- **Status**: **Not Adopted (Dev decision: not promising; baseline maintained)**
- **Action**: Per the frozen dev decision in `head_decision.json`, the candidate head was not adopted. The emotion2vec model remains the active production head.


> [!NOTE]
> **One-Shot Freeze Guarantee**: This evaluation was executed exactly once on the held-out test split. No hyperparameters, mappings, thresholds, or model weights were tuned or re-evaluated after observing these test results.
