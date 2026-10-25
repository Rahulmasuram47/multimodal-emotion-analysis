# Deployment & Web Hygiene Guide

This guide details host requirements, security hardening, secrets governance, web hygiene, and runtime latency profiles for deploying the **Multimodal Voice & Conversation Emotion Analysis System**.

---

## 1. Hosting Architecture & Requirements

| Specification | Minimum Bound | Recommended Target | Free-Tier Feasibility |
|---|---|---|---|
| **CPU** | 2 vCPU | 4 vCPU | 2 vCPU is adequate for short clips; shared free-tier CPU is slower than a laptop |
| **RAM** | 4.0 GB (full pipeline) | 8.0 GB | Full pipeline peak measured at about 2,864 MB, which does not fit Streamlit Community Cloud's published limit of about 2.7 GB (February 2024 figure). The reduced hosted-demo configuration (diarization off) peaked at about 1,569 MB and runs there (see section 6) |
| **Disk Storage** | 5.0 GB | 10.0 GB | Required for cached Hugging Face transformers and Whisper models |
| **Network** | HTTPS (TLS 1.2+) | HTTPS (TLS 1.3) | Mandatory for browser microphone recording |

> **Host Sizing Note**:
> The concurrent footprint of emotion2vec_plus_base (the active voice backend), Whisper INT8 (estimated ~150-450 MB), RoBERTa (~500 MB), and Pyannote, together with the PyTorch runtime, was measured at a peak of about 2,864 MB (single measurement, Windows laptop, Python 3.11). The SUPERB Wav2Vec2 fallback (~380 MB) is only loaded if FunASR is unavailable. The two measured peaks (with and without diarization) differ by about 1.3 GB. For hosted deployments of the full pipeline, choose a platform offering at least **4 GB RAM**, for example AWS EC2 `t3.medium` (4 GB) or `t3.large` (8 GB), or Hugging Face Spaces on the default 16 GB CPU hardware (creating a Docker Space requires a paid plan).

---

## 2. HTTPS & Security Hardening
- **HTTPS Enforcement**: WebRTC browser microphone recording (`navigator.mediaDevices.getUserMedia`) is blocked by modern browsers on non-HTTPS origins (except `localhost`). When hosting on custom domains, enforce TLS termination at the reverse proxy (Nginx, Caddy, or Cloudflare) with HTTP-to-HTTPS 301 redirection.
- **HSTS Headers**: Configure `Strict-Transport-Security: max-age=31536000; includeSubDomains; preload`.
- **Upload Protection**: By default, file uploads are restricted to `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg` up to `50 MB` and `600 seconds` (10 minutes). The hosted demo uses a lower duration cap (see section 6). Malformed, empty (0 bytes), or sub-second files are caught by `AudioValidationError` and return human-friendly alerts without leaking tracebacks.

---

## 3. Secret & Credential Governance (`HF_TOKEN`)
- **Server-Side Environment Variables**: Set `HF_TOKEN` in the hosting provider's Secret Manager (Streamlit Cloud Secrets, Hugging Face Secrets, or AWS Parameter Store). The hosted demo runs without `HF_TOKEN`, so diarization is not available there.
- **Zero Token Leakage**:
  - `HF_TOKEN` is never committed to Git, printed to console logs, rendered in Streamlit UI widgets, or transmitted in export JSON bundles.
  - Safe error categorization distinguishes missing credentials, gated-access rejections (403), network timeouts, and local cache hits.
- **Gated Model Acceptance**:
  Pyannote speaker diarization (`pyannote/speaker-diarization-community-1` and `pyannote/segmentation-3.0`) requires accepting user conditions on the Hugging Face Hub under the account associated with `HF_TOKEN` prior to initialization.

---

## 4. Web Hygiene & Static Routing

Streamlit functions as a single-page WebSocket application. Static files and SEO endpoints are handled as follows:
- **Static Assets**: Enabled via `enableStaticServing = true` in `.streamlit/config.toml`. Assets are served under `/app/static/` (including `static/robots.txt` and `static/sitemap.xml`).
- **Root Domain SEO**:
  On containerized deployments behind Nginx or Cloudflare, configure direct root forwarding:
  ```nginx
  location = /robots.txt {
      alias /app/static/robots.txt;
  }
  location = /sitemap.xml {
      alias /app/static/sitemap.xml;
  }
  ```
- **Page Metadata**: `st.set_page_config` sets the page title, icon and About text only. Open Graph tags for social cards are not provided by Streamlit and would have to be injected at the reverse proxy layer.

---

## 5. Performance Benchmarks: Cold Start vs. Warm Turns

- **On-Demand Loading**: Models are **not** loaded at application import or initial HTTP page paint. The landing UI renders before any model is loaded (time-to-first-paint was not formally measured).
- **Resource Caching**: `load_analyzer` uses `@st.cache_resource`, loading weights into memory only on the user's first "Analyze a clip" request.
- **Measured Latency** (latency benchmark: Intel x86_64 CPU, 7.7 GB RAM, Windows 11, Python 3.12.10, PyTorch 2.13.0+cpu; see `evaluation/results/latency.md`). The benchmark times voice emotion, text sentiment and fusion only. Whisper ASR, VAD, diarization, audio decoding and UI rendering are not included:
  - **Cold Start (imports, model loading, first inference)**: 28.31 s total (imports 9.46 s, neural loading 17.08 s, first inference 1.77 s).
  - **Warm Median Latency**:

    | Clip length | Median total | RTF |
    |---|---|---|
    | 5 s | 0.551 s | 0.1102 |
    | 15 s | 2.199 s | 0.1466 |
    | 30 s | 6.734 s | 0.2245 |

  - **Fusion Recompute**: about 0.2 ms in the benchmark; under 5 ms when re-weighting existing predictions without re-running neural models.
  - **End-to-End**: Latency including Whisper and diarization has not been measured. Hosted free-tier CPU is slower than the benchmark machine.

---

## 6. Hosted Demo (Streamlit Community Cloud)

- **Link**: https://multimodal-emotion-analysis.streamlit.app
- **Configuration**:
  - Python 3.12 selected under Advanced settings.
  - `requirements.txt` contains a CPU-only PyTorch index line (`--extra-index-url https://download.pytorch.org/whl/cpu`) and `torchaudio==2.11.0`. The same pinned packages are installed, including pyannote, which is switched off.
  - `packages.txt` installs `ffmpeg` and `libsndfile1`.
  - Environment switches: `DIARIZATION_ENABLED=0` and a 90-second clip cap (variable names in `config.py`). No `HF_TOKEN` is set.
- **Platform Limits**: Streamlit's published figures (as of February 2024, may change) are about 2.7 GB of memory and up to about 2 CPU cores per app. Apps with no traffic for about 12 hours go to sleep and wake when visited. Models are downloaded on the first analysis after a restart.
- **Measured Memory** (whole Streamlit server, single measurements): about 2,864 MB for the full pipeline on a Windows laptop (Python 3.11, torch 2.14.0, after analysing Sad1.wav and one multi-turn clip); about 1,569 MB with diarization switched off. These two runs were not a controlled comparison, so the difference should not be attributed to diarization alone.
- **History**: A first deployment (default GPU build of PyTorch, no ffmpeg, full pipeline) stopped working during analysis. Memory was suspected but the cause was not confirmed from the logs. After the changes above, the test clips tried completed successfully.
- **Limitations**: The hosted demo is close to the platform's memory limit and may be unstable under repeated or concurrent use. It is slower than a laptop and runs a reduced pipeline (no speaker diarization, 90-second cap).
- **Primary Deployment**: The local run is the primary tested deployment. Live demos may also use the local run exposed through a temporary Cloudflare quick tunnel. A `Dockerfile` is provided but was not built or tested (Docker is not installed on the development machine).