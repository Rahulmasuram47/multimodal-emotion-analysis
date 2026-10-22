# Known Issues

1. **Neutral text vs emotional voice mismatch**: Acted clips with fixed neutral sentences produce a text–voice mismatch (the review-signal gauge sits at 35–62% on such clips) because the text classifier reads neutral while the acoustic classifier detects the acted emotion. This is expected behaviour, not a bug.
2. **Same-speaker merge rule**: Pyannote diarization merges consecutive same-speaker turns when the gap is ≤ 0.25 s with no maximum merged length (`app/diarization.py:196`). Very long monologues may produce a single oversized turn.
3. **Earlier UI results used SUPERB fallback**: Screenshots or results taken before Phase 12 used the Wav2Vec2-SUPERB ER fallback backend (`superb/wav2vec2-base-superb-er`), which scored 12.5% on the 8-clip benchmark. Current results use the emotion2vec backend.
