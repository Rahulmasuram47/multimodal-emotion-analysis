# Evaluation: small, self-recorded and acted clips; not an accuracy benchmark

> **Notice**: This comparison analyzes differences between the app's online voice inference (whole-clip and Whisper-segmented) and the offline baseline emotion2vec pipeline on local sample and acted demo clips. This is NOT an accuracy benchmark.

- **Total files**: 30
- **Total comparison entries**: 74
- **Directories**: sample_audio, C:\Project\crema_demo

| File | True Label | Mode | Backend | App Label | App Final Conf | App Raw Conf | Baseline Label | Baseline Conf | Label Diff | Conf Diff | Cause |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `angry_sample.wav` | **angry** | whole-clip | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `angry_sample.wav` | **angry** | segment-1 | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `angry_sample.wav` | **angry** | segment-2 | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `angry_sample.wav` | **angry** | segment-3 | emotion2vec | `neutral` | 0.6500 | 0.9972 | `neutral` | 0.6500 | No | No | Identical |
| `Happy2.wav` | **happy** | whole-clip | emotion2vec | `happy` | 0.6500 | 1.0000 | `happy` | 0.6500 | No | No | Identical |
| `Happy2.wav` | **happy** | segment-1 | emotion2vec | `happy` | 0.6500 | 1.0000 | `happy` | 0.6500 | No | No | Identical |
| `Happy2.wav` | **happy** | segment-2 | emotion2vec | `happy` | 0.6500 | 1.0000 | `happy` | 0.6500 | No | No | Identical |
| `happy_sample.wav` | **happy** | whole-clip | emotion2vec | `sad` | 0.6500 | 0.9980 | `sad` | 0.6500 | No | No | Identical |
| `happy_sample.wav` | **happy** | segment-1 | emotion2vec | `sad` | 0.6500 | 0.9998 | `sad` | 0.6500 | No | No | Identical |
| `happy_sample.wav` | **happy** | segment-2 | emotion2vec | `neutral` | 0.6500 | 1.0000 | `sad` | 0.6500 | ⚠️ YES | No | Segment boundary/padding effect or VAD vs whole clip (neutral vs sad) |
| `Neutral1.wav` | **neutral** | whole-clip | emotion2vec | `sad` | 0.6500 | 0.9999 | `sad` | 0.6500 | No | No | Identical |
| `Neutral1.wav` | **neutral** | segment-1 | emotion2vec | `sad` | 0.6500 | 0.9996 | `sad` | 0.6500 | No | No | Identical |
| `Neutral1.wav` | **neutral** | segment-2 | emotion2vec | `neutral` | 0.6500 | 1.0000 | `sad` | 0.6500 | ⚠️ YES | No | Segment boundary/padding effect or VAD vs whole clip (neutral vs sad) |
| `Neutral2.wav` | **neutral** | whole-clip | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `Neutral2.wav` | **neutral** | segment-1 | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `neutral_sample.wav` | **neutral** | whole-clip | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `neutral_sample.wav` | **neutral** | segment-1 | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `neutral_sample.wav` | **neutral** | segment-2 | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `neutral_sample.wav` | **neutral** | segment-3 | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `Noisy1.wav` | **noisy** | whole-clip | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `Noisy1.wav` | **noisy** | segment-1 | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `Noisy2.wav` | **noisy** | whole-clip | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `Noisy2.wav` | **noisy** | segment-1 | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `Noisy2.wav` | **noisy** | segment-2 | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `Non_English1.wav` | **non** | whole-clip | emotion2vec | `happy` | 0.6500 | 0.9999 | `happy` | 0.6500 | No | No | Identical |
| `Non_English1.wav` | **non** | segment-1 | emotion2vec | `happy` | 0.6500 | 1.0000 | `happy` | 0.6500 | No | No | Identical |
| `Non_English2.wav` | **non** | whole-clip | emotion2vec | `sad` | 0.6500 | 0.9949 | `sad` | 0.6500 | No | No | Identical |
| `Non_English2.wav` | **non** | segment-1 | emotion2vec | `sad` | 0.6500 | 0.9999 | `sad` | 0.6500 | No | No | Identical |
| `Non_English2.wav` | **non** | segment-2 | emotion2vec | `uncertain` | 0.3000 | 0.6000 | `sad` | 0.6500 | ⚠️ YES | ⚠️ YES | Segment boundary/padding effect or VAD vs whole clip (uncertain vs sad); Segment confidence calibration/cap (app=0.3000 vs base=0.6500) |
| `real_voice.wav` | **real** | whole-clip | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `real_voice.wav` | **real** | segment-1 | emotion2vec | `neutral` | 0.6500 | 0.9988 | `neutral` | 0.6500 | No | No | Identical |
| `real_voice.wav` | **real** | segment-2 | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `rising_tone.wav` | **rising** | whole-clip | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `rising_tone.wav` | **rising** | segment-1 | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `rising_tone.wav` | **rising** | segment-2 | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `rising_tone.wav` | **rising** | segment-3 | emotion2vec | `angry` | 0.5497 | 0.5497 | `neutral` | 0.6500 | ⚠️ YES | ⚠️ YES | Segment boundary/padding effect or VAD vs whole clip (angry vs neutral); Segment confidence calibration/cap (app=0.5497 vs base=0.6500) |
| `Sad1.wav` | **sad** | whole-clip | emotion2vec | `sad` | 0.6500 | 0.9927 | `sad` | 0.6500 | No | No | Identical |
| `Sad1.wav` | **sad** | segment-1 | emotion2vec | `neutral` | 0.6500 | 1.0000 | `sad` | 0.6500 | ⚠️ YES | No | Segment boundary/padding effect or VAD vs whole clip (neutral vs sad) |
| `Sad2.wav` | **sad** | whole-clip | emotion2vec | `sad` | 0.6500 | 0.9986 | `sad` | 0.6500 | No | No | Identical |
| `Sad2.wav` | **sad** | segment-1 | emotion2vec | `sad` | 0.6500 | 0.9318 | `sad` | 0.6500 | No | No | Identical |
| `Sad3.wav` | **sad** | whole-clip | emotion2vec | `sad` | 0.6500 | 1.0000 | `sad` | 0.6500 | No | No | Identical |
| `Sad3.wav` | **sad** | segment-1 | emotion2vec | `sad` | 0.6500 | 1.0000 | `sad` | 0.6500 | No | No | Identical |
| `sad_sample.wav` | **sad** | whole-clip | emotion2vec | `sad` | 0.6500 | 1.0000 | `sad` | 0.6500 | No | No | Identical |
| `sad_sample.wav` | **sad** | segment-1 | emotion2vec | `sad` | 0.6500 | 1.0000 | `sad` | 0.6500 | No | No | Identical |
| `sad_sample.wav` | **sad** | segment-2 | emotion2vec | `sad` | 0.6500 | 0.6547 | `sad` | 0.6500 | No | No | Identical |
| `sad_sample.wav` | **sad** | segment-3 | emotion2vec | `neutral` | 0.6500 | 0.9985 | `sad` | 0.6500 | ⚠️ YES | No | Segment boundary/padding effect or VAD vs whole clip (neutral vs sad) |
| `sample_16k.wav` | **sample** | whole-clip | emotion2vec | `uncertain` | 0.3000 | 0.9903 | `uncertain` | 0.3000 | No | No | Identical |
| `sample_16k.wav` | **sample** | per-segment (none) | emotion2vec | `no_speech` | 0.0000 | 0.0000 | `uncertain` | 0.3000 | ⚠️ YES | ⚠️ YES | Segment boundary/padding effect or VAD vs whole clip (no_speech vs uncertain); Segment confidence calibration/cap (app=0.0000 vs base=0.3000) |
| `1001_DFA_NEU_XX.wav` | **neutral** | whole-clip | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `1001_DFA_NEU_XX.wav` | **neutral** | segment-1 | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `1001_IEO_ANG_HI.wav` | **angry** | whole-clip | emotion2vec | `angry` | 0.6500 | 1.0000 | `angry` | 0.6500 | No | No | Identical |
| `1001_IEO_ANG_HI.wav` | **angry** | segment-1 | emotion2vec | `angry` | 0.6500 | 0.9998 | `angry` | 0.6500 | No | No | Identical |
| `1001_IEO_ANG_LO.wav` | **angry** | whole-clip | emotion2vec | `uncertain` | 0.3000 | 0.9567 | `uncertain` | 0.3000 | No | No | Identical |
| `1001_IEO_ANG_LO.wav` | **angry** | segment-1 | emotion2vec | `uncertain` | 0.3000 | 0.9492 | `uncertain` | 0.3000 | No | No | Identical |
| `1001_IEO_DIS_HI.wav` | **disgust** | whole-clip | emotion2vec | `uncertain` | 0.3000 | 0.9868 | `uncertain` | 0.3000 | No | No | Identical |
| `1001_IEO_DIS_HI.wav` | **disgust** | segment-1 | emotion2vec | `happy` | 0.6500 | 1.0000 | `uncertain` | 0.3000 | ⚠️ YES | ⚠️ YES | Segment boundary/padding effect or VAD vs whole clip (happy vs uncertain); Segment confidence calibration/cap (app=0.6500 vs base=0.3000) |
| `1001_IEO_DIS_LO.wav` | **disgust** | whole-clip | emotion2vec | `uncertain` | 0.3000 | 0.9980 | `uncertain` | 0.3000 | No | No | Identical |
| `1001_IEO_DIS_LO.wav` | **disgust** | segment-1 | emotion2vec | `happy` | 0.6500 | 0.9967 | `uncertain` | 0.3000 | ⚠️ YES | ⚠️ YES | Segment boundary/padding effect or VAD vs whole clip (happy vs uncertain); Segment confidence calibration/cap (app=0.6500 vs base=0.3000) |
| `1001_IEO_FEA_HI.wav` | **fear** | whole-clip | emotion2vec | `uncertain` | 0.3000 | 1.0000 | `uncertain` | 0.3000 | No | No | Identical |
| `1001_IEO_FEA_HI.wav` | **fear** | segment-1 | emotion2vec | `uncertain` | 0.3000 | 1.0000 | `uncertain` | 0.3000 | No | No | Identical |
| `1001_IEO_FEA_LO.wav` | **fear** | whole-clip | emotion2vec | `uncertain` | 0.3000 | 0.9996 | `uncertain` | 0.3000 | No | No | Identical |
| `1001_IEO_FEA_LO.wav` | **fear** | segment-1 | emotion2vec | `uncertain` | 0.3000 | 0.9996 | `uncertain` | 0.3000 | No | No | Identical |
| `1001_IEO_HAP_HI.wav` | **happy** | whole-clip | emotion2vec | `happy` | 0.6500 | 1.0000 | `happy` | 0.6500 | No | No | Identical |
| `1001_IEO_HAP_HI.wav` | **happy** | segment-1 | emotion2vec | `happy` | 0.6500 | 1.0000 | `happy` | 0.6500 | No | No | Identical |
| `1001_IEO_HAP_LO.wav` | **happy** | whole-clip | emotion2vec | `happy` | 0.6500 | 1.0000 | `happy` | 0.6500 | No | No | Identical |
| `1001_IEO_HAP_LO.wav` | **happy** | segment-1 | emotion2vec | `happy` | 0.6500 | 1.0000 | `happy` | 0.6500 | No | No | Identical |
| `1001_IEO_NEU_XX.wav` | **neutral** | whole-clip | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `1001_IEO_NEU_XX.wav` | **neutral** | segment-1 | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `1001_IEO_SAD_HI.wav` | **sad** | whole-clip | emotion2vec | `happy` | 0.5283 | 0.5283 | `happy` | 0.5368 | No | ⚠️ YES | Confidence difference: app=0.5283 vs baseline=0.5368 (app raw=0.5283, cap=0.65) |
| `1001_IEO_SAD_HI.wav` | **sad** | segment-1 | emotion2vec | `happy` | 0.6500 | 0.9383 | `happy` | 0.5368 | No | ⚠️ YES | Segment confidence calibration/cap (app=0.6500 vs base=0.5368) |
| `1001_IEO_SAD_LO.wav` | **sad** | whole-clip | emotion2vec | `sad` | 0.6500 | 0.7587 | `sad` | 0.6500 | No | No | Identical |
| `1001_IEO_SAD_LO.wav` | **sad** | segment-1 | emotion2vec | `neutral` | 0.6500 | 0.7720 | `sad` | 0.6500 | ⚠️ YES | No | Segment boundary/padding effect or VAD vs whole clip (neutral vs sad) |
| `1002_IEO_NEU_XX.wav` | **neutral** | whole-clip | emotion2vec | `neutral` | 0.6500 | 1.0000 | `neutral` | 0.6500 | No | No | Identical |
| `1002_IEO_NEU_XX.wav` | **neutral** | segment-1 | emotion2vec | `neutral` | 0.6500 | 0.9999 | `neutral` | 0.6500 | No | No | Identical |
