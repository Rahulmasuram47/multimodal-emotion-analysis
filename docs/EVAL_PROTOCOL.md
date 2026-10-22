# CREMA-D Evaluation Protocol

## 1. Label Space & Mapping
- **Raw Vocabulary (6 classes)**: `ANG`, `DIS`, `FEA`, `HAP`, `NEU`, `SAD`.
- **Target Vocabulary (4 classes)**: `ANG` -> angry, `HAP` -> happy, `NEU` -> neutral, `SAD` -> sad.
- **Abstention / Non-Target**: Ground-truth `DIS` and `FEA` are non-target emotions. Voice model predictions of `uncertain` (or unmapped classes) act as abstentions.
- **Reporting**:
  - Target 4-class evaluation reports coverage (`answered / total_target`).
  - Accuracy is reported both on **answered clips** (`correct / answered`) and **overall target clips** (`correct / total_target`, treating abstention as incorrect).
  - Ground-truth `FEA` and `DIS` clips are tracked separately to evaluate out-of-domain rejection / false-alarm behavior.
  - **Chance Level**: Computed empirically from the real label distribution across evaluated clips (not assuming uniform 1/6 or 1/4).

## 2. Core Metrics
- **Metrics**: Accuracy (overall & answered), Unweighted Average Recall (UAR), Macro-F1, per-class Recall & Precision, and Full Confusion Matrix.
- **Confidence Intervals**: 95% CIs computed via Actor-Cluster Bootstrap (seed 42, 1000 iterations).
- **Actor Balancing**: Clips per actor are capped to prevent vocal dominance by over-represented speakers.

## 3. Head-Adoption Criteria
A new classification head / fine-tuned model is adopted only if:
1. **Mean UAR Gain**: >= +0.03 gain over frozen zero-shot baseline in actor-grouped dev cross-validation.
2. **Consistency**: Achieves higher UAR in a strict majority of CV folds.
3. **Safety / Vulnerable Classes**: Per-class recall on `sad` and `neutral` must NOT regress compared to baseline.

## 4. Split Isolation & Protocol Integrity
- **Dev Split**: Used exclusively for exploration, calibration tuning, confidence caps, and head architecture selection via actor-grouped CV.
- **Test Split**: Strictly locked and evaluated **exactly once** using locked split definitions (`evaluation/results/crema_d_splits.json`). No parameter re-tuning after test evaluation.
