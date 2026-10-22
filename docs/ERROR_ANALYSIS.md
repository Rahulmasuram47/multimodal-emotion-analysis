# CREMA-D Dev Error Analysis

## 1. Setup & Evaluation Scope
- **Backend**: `emotion2vec` (`emotion2vec/emotion2vec_plus_base`).
- **Dataset**: CREMA-D (acted speech).
- **Split**: Dev split only (N = 400 clips across 9 actors; Seed: 42).
- **Test Split**: Strictly not evaluated (locked and untouched per protocol).

## 2. Confusion Matrix & Abstention / Coverage
Target 4 classes: `[neutral, happy, angry, sad]` (rows: true, columns: predicted).

| True \ Pred | Neutral | Happy | Angry | Sad |
| :--- | :---: | :---: | :---: | :---: |
| **Neutral** | 100 | 0 | 0 | 0 |
| **Happy** | 3 | 95 | 0 | 1 |
| **Angry** | 0 | 1 | 95 | 1 |
| **Sad** | 12 | 1 | 0 | 86 |

- **Answered / Coverage**: 395 / 400 clips (98.75%).
- **Abstention**: 5 / 400 clips (1.25%) predicted as non-4class (`uncertain`).

## 3. Top Confusions (Counts & Rates)
1. **Sad -> Neutral**: 12 counts (12.12% of true Sad; 3.00% of all clips).
2. **Happy -> Neutral**: 3 counts (3.03% of true Happy; 0.75% of all clips).
3. **Sad -> Happy**: 1 count (1.01% of true Sad; 0.25% of all clips).
4. **Happy -> Sad**: 1 count (1.01% of true Happy; 0.25% of all clips).
5. **Angry -> Happy**: 1 count (1.03% of true Angry; 0.25% of all clips).
*(Tied #5: **Angry -> Sad**: 1 count, 1.03% of true Angry, 0.25% of all clips).*

## 4. Per-Class Precision & Recall
- **Neutral**: Recall = 100.00% (100/100), Precision = 86.96% (100/115), F1 = 0.9302
- **Happy**: Recall = 95.96% (95/99), Precision = 97.94% (95/97), F1 = 0.9694
- **Angry**: Recall = 97.94% (95/97), Precision = 100.00% (95/95), F1 = 0.9896
- **Sad**: Recall = 86.87% (86/99), Precision = 97.73% (86/88), F1 = 0.9198
- **Summary**: Accuracy = 94.00%, UAR = 95.19%, Macro-F1 = 0.9522.
- **Recall by Level (LO/MD/HI/XX)**: not yet measured (not present in JSON).

## 5. Explanatory Hypotheses
- **Acoustic Low-Activation Hypothesis**: Subdued sadness exhibits lower pitch variance and muted energy resembling neutral vocal patterns [hypothesis].
- **Arousal Overlap Hypothesis**: Occasional cross-valence confusions between happy and angry/sad stem from overlapping acoustic energy dynamics [hypothesis].
