# Known Issues & Operational Considerations

1. **Model Commit Pinning**: Hugging Face models (`wav2vec2-base-superb-er`, `twitter-roberta-base-sentiment-latest`) resolve to default branch HEAD rather than immutable commit hashes.
2. **Dependency Version Bounds**: `requirements.txt` specifies minimum version bounds (`>=`) rather than locked exact hashes (`==`).
3. **Python Runtime Differences**: Project specification targets Python 3.11 on Windows, while CI workflow (`tests.yml`) executes under Python 3.12 on Linux.
4. **Pyannote Gated Terms**: `pyannote/speaker-diarization-community-1` requires user acceptance of conditions on Hugging Face before first download.
5. **CPU Inference Latency**: Full multimodal pipeline latency scales linearly with audio length on low-spec CPU devices.
6. **Streamlit Cloud Static Root Routing**: Streamlit Cloud intercepts root `/robots.txt`, `/sitemap.xml`, and custom 404s; reverse proxy (Nginx/Cloudflare) needed for root SEO and OG tags.
7. **Free-Tier Host Memory Bounds**: Streamlit Cloud 1 GB memory ceiling is insufficient for concurrent Whisper + Pyannote + Wav2Vec2 + RoBERTa loading; 4-8 GB RAM recommended.
