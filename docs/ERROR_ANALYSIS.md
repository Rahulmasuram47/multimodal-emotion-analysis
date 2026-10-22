# CREMA-D Dev Error Analysis

## 1. Evaluation Scope & Setup
- **Dataset**: CREMA-D (acted speech).
- **Split**: Dev split only (N = 400 clips, 9 actors, Seed 42).
- **Test Split**: Not evaluated (locked and untouched per protocol).
- **Active Backend**: `emotion2vec` (`emotion2vec/emotion2vec_plus_base`).

## 2. Confusion Matrix & Abstention
Target classes: `[neutral, happy, angry, sad]` (rows: true, columns: predicted).

| True \ Pred | Neutral | Happy | Angry | Sad |
| :--- | :---: | :---: | :---: | :---: |
| **Neutral** | 100 | 0 | 0 | 0 |
| **Happy** | 3 | 95 | 0 | 1 |
| **Angry** | 0 | 1 | 95 | 1 |
| **Sad** | 12 | 1 | 0 | 86 |

- **Coverage**: 98.75% (395 / 400 target clips classified into 4 classes).
- **Abstention**: 1.25% (5 / 400 clips predicted as non-4class / uncertain).

## 3. Per-Class Precision & Recall
- **Neutral**: Precision = 86.96%, Recall = 100.00%, F1 = 0.9302 (Support = 100)
- **Happy**: Precision = 97.94%, Recall = 95.96%, F1 = 0.9694 (Support = 99)
- **Angry**: Precision = 100.00%, Recall = 97.94%, F1 = 0.9896 (Support = 97)
- **Sad**: Precision = 97.73%, Recall = 86.87%, F1 = 0.9198 (Support = 99)
- **Overall**: Accuracy = 94.00% (95% CI: [90.06%, 97.14%]), UAR = 95.19%, Macro-F1 = 0.9522.
- **Recall by Level (LO / MD / HI / XX)**: Not yet measured (not recorded in baseline JSON).

## 4. Top Confusions
1. **Sad -> Neutral**: 12 counts (12.12% of True Sad).
2. **Happy -> Neutral**: 3 counts (3.03% of True Happy).
3. **Sad -> Happy**: 1 count (1.01% of True Sad).
4. **Happy -> Sad**: 1 count (1.01% of True Happy).
5. **Angry -> Happy**: 1 count (1.03% of True Angry).
*(Tied: **Angry -> Sad**: 1 count, 1.03% of True Angry).*

## 5. Explanatory Hypotheses
- **Sad to Neutral Confusion [Hypothesis]**: Subdued or low-activation sadness exhibits flat pitch contours and reduced acoustic energy that closely overlap with baseline neutral speech.
- **High Pitch Overlap in High Arousal [Hypothesis]**: Exclamatory happy and angry expressions occasionally share acoustic pitch dynamics, causing rare cross-arousal confusion.
