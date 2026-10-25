# Deployment & Web Hygiene Guide

This guide details host requirements, security hardening, secrets governance, web hygiene, and runtime latency profiles for deploying the **Multimodal Voice & Conversation Emotion Analysis System**.

---

## 1. Hosting Architecture & Requirements

| Specification | Minimum Bound | Recommended Target | Free-Tier Feasibility |
|---|---|---|---|
| **CPU** | 2 vCPU | 4 vCPU | 2 vCPU adequate for standard clips |
| **RAM** | 4.0 GB | 8.0 GB | **Not feasible on 1 GB free tiers** (Streamlit Community Cloud 1GB limit causes OOM during model load) |
| **Disk Storage** | 5.0 GB | 10.0 GB | Required for cached Hugging Face transformers and Whisper models |
| **Network** | HTTPS (TLS 1.2+) | HTTPS (TLS 1.3) | Mandatory for browser microphone recording |

> **Host Sizing Note**:
> Concurrent memory footprint when loading Wav2Vec2 (`~380 MB`), Whisper INT8 (`~150-450 MB`), RoBERTa (`~500 MB`), and Pyannote (`~1.5 GB`) with PyTorch runtime peaks around 2.5–3.5 GB RAM. For hosted deployments, choose platforms offering at least **4 GB RAM** (e.g., Hugging Face Spaces 16 GB free tier, AWS EC2 `t3.medium`/`t3.large`, or Render Standard).

---

## 2. HTTPS & Security Hardening
- **HTTPS Enforcement**: WebRTC browser microphone recording (`navigator.mediaDevices.getUserMedia`) is blocked by modern browsers on non-HTTPS origins (except `localhost`). When hosting on custom domains, enforce TLS termination at the reverse proxy (Nginx, Caddy, or Cloudflare) with HTTP-to-HTTPS 301 redirection.
- **HSTS Headers**: Configure `Strict-Transport-Security: max-age=31536000; includeSubDomains; preload`.
- **Upload Protection**: File uploads are strictly restricted to `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg` up to `50 MB` and `600 seconds` (10 minutes). Malformed, empty (0 bytes), or sub-second files are caught by `AudioValidationError` and return human-friendly alerts without leaking tracebacks.

---

## 3. Secret & Credential Governance (`HF_TOKEN`)
- **Server-Side Environment Variables**: Set `HF_TOKEN` in the hosting provider's Secret Manager (Streamlit Cloud Secrets, Hugging Face Secrets, or AWS Parameter Store).
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
- **Open Graph & Meta Tags**: Injected at the reverse proxy layer or via Streamlit's `st.set_page_config` metadata for social cards and previews.

---

## 5. Performance Benchmarks: Cold Start vs. Warm Turns

- **On-Demand Loading**: Models are **not** loaded at application import or initial HTTP page paint. The landing UI renders immediately (time-to-first-paint < 1.0s).
- **Resource Caching**: `load_analyzer` uses `@st.cache_resource`, loading weights into memory only on the user's first "Analyze a clip" request.
- **Empirical Measured Latency** (Intel Core Ultra CPU, single-thread reference):
  - **Cold Start (Model Instantiation)**: ~1.2s to 3.5s (displayed with an informative status spinner).
  - **Warm Turn (Inference on 3.0s clip)**: 0.95s total pipeline latency (~0.32x real-time factor).
  - **Fusion Recompute**: < 5 ms (re-weighting existing predictions without re-running neural models).
Measured memory (2026-10-25): the Streamlit server’s peak working set was about 2,864 MB on a Windows laptop (Python 3.11, torch 2.14.0) after analysing Sad1.wav and one multi-turn clip with the default pipeline. This is a single measurement on one machine. Streamlit Community Cloud’s published maximum is about 2.7 GB (February 2024 figure), so the full app is not hosted there. The tested deployment is a local run. Live demos use the local run, optionally exposed through a temporary Cloudflare quick tunnel.