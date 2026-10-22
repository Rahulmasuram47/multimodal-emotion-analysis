# Acted speech; argmax and accuracy are unchanged by temperature scaling

## Evaluation Scope & Partition Details
- **Dataset Partition**: DEV split (N = 1451 clips across 28 actors, Seed: 42)
- **Active Backend Model**: `emotion2vec` (4 canonical classes: ['neutral', 'happy', 'angry', 'sad'])
- **Learned Temperature (T)**: `3.1192` (Dev NLL optimized via bounded scalar search)

## Calibration Metrics (DEV)
- **Negative Log-Likelihood (NLL)**:
  - Before Calibration ($T=1.00$): `0.3799`
  - After Calibration ($T=3.12$): `0.1944`
- **Expected Calibration Error (ECE - 10 Bins)**:
  - Before Calibration ($T=1.00$): `0.0482` (`4.82%`)
  - After Calibration ($T=3.12$): `0.0155` (`1.55%`) (95% Actor-Cluster CI: `[0.0110, 0.0328]`)

## Reliability Diagram Bins (Calibrated DEV)
| Confidence Range | Clip Count | Mean Confidence | Accuracy | Calibration Gap |
|---|---|---|---|---|
| `0.00 - 0.10` | 0 | `5.00%` | `0.00%` | `0.00%` |
| `0.10 - 0.20` | 0 | `15.00%` | `0.00%` | `0.00%` |
| `0.20 - 0.30` | 0 | `25.00%` | `0.00%` | `0.00%` |
| `0.30 - 0.40` | 8 | `35.53%` | `50.00%` | `14.47%` |
| `0.40 - 0.50` | 26 | `45.41%` | `30.77%` | `14.64%` |
| `0.50 - 0.60` | 44 | `54.26%` | `54.55%` | `0.28%` |
| `0.60 - 0.70` | 38 | `64.61%` | `81.58%` | `16.97%` |
| `0.70 - 0.80` | 42 | `74.62%` | `78.57%` | `3.95%` |
| `0.80 - 0.90` | 94 | `85.67%` | `87.23%` | `1.57%` |
| `0.90 - 1.00` | 1199 | `98.82%` | `98.17%` | `0.65%` |

## Relationship to VOICE_CONF_CAP (0.65)
- **Voice Confidence Cap Placement**: Situated in bin `[0.60, 0.70]` (Bin Count: 38).
- **Policy Recommendation**: Maintain VOICE_CONF_CAP = 0.65 as an empirical safety threshold in decision fusion. At T=3.12, probability distributions are softened/aligned, and the 0.65 cap prevents over-confident acoustic overrides.


## Cached Test Split Evaluation (REPORT-ONLY)
> [!NOTE]
> **Split Isolation Compliance**: Temperature $T=3.12$ was fixed exclusively from the DEV split. Test metrics are evaluated as REPORT-ONLY with strictly no refitting or parameter adjustment.

- **Test Clips**: 1673 across 31 held-out actors
- **Test ECE Before**: `0.0406` -> **After**: `0.0207`
- **Test NLL Before**: `0.3341` -> **After**: `0.1768`

## Invariance Statement
> [!IMPORTANT]
> Argmax predictions and classification accuracy are strictly invariant under temperature scaling; temperature calibration re-aligns probability magnitudes but does not alter top-1 decision boundaries.
