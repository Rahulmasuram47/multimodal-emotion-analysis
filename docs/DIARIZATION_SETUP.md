# Speaker Diarization Setup (HF_TOKEN)

This project uses `pyannote/speaker-diarization-community-1` to separate
"who spoke when" in a multi-speaker recording. Without a Hugging Face
token, diarization is skipped and every segment is labeled `"Unknown"`
— the rest of the pipeline (voice emotion, text sentiment, fusion,
escalation) still runs normally.

This is an **optional enhancement**, not a requirement to run the app.

---

## 1. Create a Hugging Face account

If you don't already have one: https://huggingface.co/join

## 2. Accept the model's gated terms

`pyannote/speaker-diarization-community-1` requires you to accept its
usage terms once, while logged in:

1. Go to https://huggingface.co/pyannote/speaker-diarization-community-1
2. Click **"Agree and access repository"** (you must be logged in)
3. Some versions of pyannote also gate a segmentation sub-model —
   if diarization fails with a "gated repo" error, also visit
   https://huggingface.co/pyannote/segmentation-3.0 and accept there too

Without accepting these terms, the token will be rejected even if it's
valid.

## 3. Create an access token

1. Go to https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Name it anything (e.g. `mini-project-diarization`)
4. Type: **Read** is sufficient — no write access needed
5. Click **Create token** and copy the value (starts with `hf_`)

Treat this token like a password. Never paste it into a commit, a
screenshot, or a chat message you plan to share publicly.

## 4. Add the token to this project

Create a file named exactly `.env` in the project root (same folder as
`config.py`). This file is already listed in `.gitignore` and will
never be committed.

Contents of `.env`:
HF_TOKEN=YOUR_HUGGINGFACE_TOKEN

No quotes, no spaces around the `=`.

The app loads this automatically via `python-dotenv` — the very first
lines of `streamlit_app.py` call `load_dotenv()` before anything else
runs, so `HF_TOKEN` reaches `SpeakerDiarizer` (in `app/diarization.py`)
without any further setup.

## 5. Restart the app fully

`HF_TOKEN` is only read once, when the app process starts. If the app
was already running before you created `.env`, you must **fully stop
and restart it** — a browser refresh alone will not pick up the new
environment variable:

```powershell
# In the terminal running Streamlit: press Ctrl+C to stop it completely
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.fileWatcherType none
```

## 6. Confirm it worked

Open the app's sidebar — a **"Diarization Status"** section shows:

- **✓ Configured (hf_xxxx...xxxx)** — a token was found (only a short
  preview is ever shown, never the full token)
- **✗ Not configured** — no token found; speakers will show as
  "Unknown"

A configured token does not guarantee every run succeeds (network
issues, an unaccepted gated model, or a revoked token can still cause a
failure on a specific run). After each analysis, check the
**Speaker Analysis** section in the results — it shows either:

- ✅ "Speaker diarization ran successfully on this analysis."
- ⚠️ "Speaker diarization failed on this analysis... Reason: ..."

The reason text comes directly from the underlying error and should
tell you exactly what went wrong (missing token, gated terms not
accepted, network failure, etc.).

## 7. First-run download

The first time diarization actually runs, several model files download
from Hugging Face (segmentation model, embedding model, PLDA files).
This takes roughly 30-60 seconds depending on your connection and only
happens once — the files are cached locally afterward
(`~/.cache/huggingface/`) and every later run uses the cached copy, no
internet required.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Sidebar shows "✗ Not configured" | `.env` missing, misspelled, or app wasn't restarted | Re-check step 4-5 |
| "No Hugging Face token found" error at runtime | `.env` exists but wasn't loaded | Confirm `load_dotenv()` is called at the very top of `streamlit_app.py`, before other imports |
| "401 Unauthorized" or "gated repo" error | Terms not accepted, or token revoked | Redo step 2; generate a fresh token if needed |
| Diarization succeeds but everyone shows as one speaker | Normal for single-speaker audio | Test with a real multi-speaker recording |
| Speakers show as "Unknown" despite a valid token | Diarization ran but found no distinguishable turns (e.g. audio too short, very similar voices) | Check the Speaker Analysis warning text for the specific reason |

## Never commit your token

`.env` is gitignored. Before any commit, you can double-check with:

```powershell
git status --short
```

`.env` should **never** appear in this output. If it ever does,
immediately revoke the token at
https://huggingface.co/settings/tokens and generate a new one — do not
just remove it from a future commit, since it would already be in git
history.

---

## 8. Manual Verification Checklist

Before tagging Phase 6 complete or sharing recordings:
- [ ] Confirm `.env` is ignored by git: `git check-ignore -v .env` returns `.gitignore:4:.env`.
- [ ] Confirm no token appears in terminal logs or UI displays: check that only sanitized preview `(hf_...xxxx)` is shown.
- [ ] Confirm single-speaker clips report `Unknown` or `Speaker A` without crashing.
- [ ] Confirm segments with low overlap ratio (<0.35) are marked `"Uncertain Speaker"`.
- [ ] Run a real two-speaker conversation recording and confirm multi-speaker analytics appear (per-speaker talk time, intensity, trend, and escalation initiator).
- [ ] Verify Phase 6 remains untagged until an actual two-speaker recording succeeds with pyannote.
