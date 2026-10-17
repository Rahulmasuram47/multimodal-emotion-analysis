# Natural-Speech Affective Evaluation Protocol

## 1. Rationale & Purpose
Acted corpora (e.g. CREMA-D, IEMOCAP) fail to reflect natural speech dynamics: actors exaggerate vocal expressions, speak in isolated sentences, and follow theatrical scripts. Real conversation exhibits subtle prosody, spontaneous pauses, conversational overlap, and regional dialectal shifts. This protocol defines a clean, privacy-preserving methodology to gather and evaluate spontaneous speech.

## 2. Dataset Design & Acoustic Specifications
- **Clip Volume & Duration**: 40–60 clips total; duration bounded strictly between 3.0 and 15.0 seconds.
- **Speaker Diversity**: $\ge 8$ consenting adult speakers, with dedicated inclusion of Indian English dialectal contours.
- **Elicitation Task**: Speakers recount a genuine recent personal event (e.g. positive news, a frustrating obstacle, or everyday routine), producing authentic unscripted speech rather than acted prompts.

## 3. Privacy, Anonymity & Consent
- **Consent Text**: *"I consent to having my unscripted voice sample anonymously recorded and evaluated for affective acoustic research. I understand no personal identifiers, full names, or sensitive data will be retained or published."*
- **Anonymized Filename Convention**: `spk<ID>_<CLIP_NUM>.wav` (e.g. `spk01_001.wav`, `spk08_005.wav`).
- **Data Protection**: Personal names, location references, and biometric identities are omitted. Audio remains strictly local and `.gitignore`-protected.

## 4. Annotation Protocol & Blinding
- **Vocabulary**: 5 discrete labels: `neutral`, `happy`, `sad`, `angry`, `other`.
- **Rater Independence**: $\ge 2$ annotators listen to randomized audio clips in complete isolation.
- **Strict Blinding**: Annotators never see model outputs, acoustic predictions, or speaker intent.
- **Speaker Intent Separation**: The speaker's self-reported intent is logged in a separate table. **Intent is NOT ground truth for expressed affect**; communicative expressions may diverge from inner feeling.

## 5. Scoring & Consensus Determination
- **Inter-Rater Reliability**: Evaluated via Cohen's kappa $\kappa$ (2 raters) or Fleiss' $\kappa$ (3+ raters) with bootstrap 95% CIs. Human disagreement reflects natural perceptual ambiguity, not bad data.
- **Consensus Rule**: Majority agreement determines consensus ground truth; even ties are flagged as `disputed`.
- **Three-Way Comparison**: Model evaluated against: (a) unanimous consensus, (b) majority consensus, and (c) speaker intent.
- **Isolation Constraint**: Natural speech metrics are reported independently in `evaluation/results/natural_set.md` and are **never pooled** with CREMA-D or synthetic benchmarks.
