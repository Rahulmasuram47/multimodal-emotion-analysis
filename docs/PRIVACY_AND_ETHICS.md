# Privacy, Data Governance & Research Ethics Charter

## 1. Data Processing Location & Network Hygiene
- **100% Local CPU Execution**: Audio ingestion, Faster-Whisper transcription, Wav2Vec2/RoBERTa inference, and fusion arbitration run entirely in host memory.
- **Zero Outbound Telemetry**: Static code analysis confirms 0 external HTTP/API network calls during inference (`requests`, `urllib`, `socket` calls absent). Only initial model weight download from Hugging Face occurs if weights are not cached locally.

## 2. Storage & Ephemeral Data Lifecycle
- **Ephemeral Processing**: Uploaded files and recorded microphone buffers reside in temporary memory/disk locations.
- **Immediate Deletion**: Temporary files are unlinked (deleted) immediately upon inference completion or error handling in `finally` blocks.
- **No Involuntary Retention**: Audio buffers and transcripts are never stored in a database or external cloud.
- **Logging Hygiene**: Raw speech transcripts and speaker embeddings are never logged at INFO or DEBUG levels to console outputs or log files.

## 3. Recording Consent & Speaker Anonymization
- **Mandatory Informed Consent**: Live recording and natural-speech data collection require explicit, voluntary, affirmative consent from all speakers.
- **Anonymization**: Files use anonymized keys (e.g. `spk01_001.wav`); names, affiliations, and personal metadata are strictly excluded.

## 4. Intended Use
- **Academic & Educational Research**: Exploration of multimodal conversation dynamics, acoustic prosody, and linguistic sentiment arbitration.
- **Human-in-the-Loop Decision Support**: Strictly advisory indicators for human conversational review.

## 5. Prohibited Uses
This software is strictly forbidden from being deployed or evaluated for:
1. Clinical psychiatric, psychological, or medical diagnostics.
2. Hiring, recruitment, promotion, or workplace performance evaluation.
3. Employee surveillance, eavesdropping, or disciplinary monitoring.
4. Legal interrogation, polygraph testing, credibility scoring, or deception detection.
5. Any automated decision-making impacting personal freedom, livelihoods, or civil rights.
