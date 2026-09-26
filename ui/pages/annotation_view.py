"""
Annotation Helper page — Tab 4 of the Multimodal Emotion Analysis System.

Allows the annotation team (Sahithi, Akshitha, Rahul) to:
  1. Browse unannotated audio files from sample_audio/ and recordings/.
  2. Play each clip directly in the browser.
  3. Get a Whisper draft transcript (never treated as ground truth).
  4. Fill in true_voice_emotion, true_text_sentiment, true_speaker_count,
     true_escalation, true_incongruent, notes, annotator.
  5. Persist the row to data/evaluation_dataset.csv — never silently
     overwriting an existing annotation without confirmation.

Constraints (AGENTS.md):
  - Front-end stays Streamlit; no React / FastAPI.
  - No ML models loaded here except Faster-Whisper for the *draft* transcript.
  - Never fabricate labels or transcripts.
  - Blank = not annotated; never a class.
"""

from __future__ import annotations

import csv
import io
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from config import (
    EVAL_DATASET_PATH,
    RECORDINGS_DIR,
    SAMPLE_AUDIO_DIR,
    SUPPORTED_AUDIO_EXTENSIONS,
    VOICE_EMOTION_OPTIONS,
    TEXT_SENTIMENT_OPTIONS,
    EVAL_MIN_N,
)
from evaluation.dataset import EvaluationDataset, EvaluationSample, EXTENDED_COLUMNS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _list_audio_files(folders: List[str]) -> List[str]:
    """Return sorted list of supported audio files across the given folders."""
    paths: List[str] = []
    for folder in folders:
        p = Path(folder)
        if not p.exists():
            continue
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                paths.append(str(f))
    return paths


def _load_dataset(path: str) -> EvaluationDataset:
    """Load dataset from path, or return empty dataset if file does not exist."""
    if os.path.exists(path):
        try:
            return EvaluationDataset.from_csv(path)
        except Exception as e:
            st.warning(f"Could not load evaluation dataset: {e}. Starting fresh.")
    return EvaluationDataset()


def _row_for_file(dataset: EvaluationDataset, file_path: str) -> Optional[EvaluationSample]:
    """Find existing row for file_path (normalised comparison)."""
    norm = Path(file_path).as_posix()
    for s in dataset.samples:
        if Path(s.filename).as_posix() == norm:
            return s
    return None


def _save_row(dataset: EvaluationDataset, new_sample: EvaluationSample, csv_path: str) -> None:
    """Add or update a row in the dataset and write to CSV (atomic-ish via temp buffer)."""
    # Remove old row for same file if present
    dataset.samples = [
        s for s in dataset.samples
        if Path(s.filename).as_posix() != Path(new_sample.filename).as_posix()
    ]
    dataset.add_sample(new_sample)

    # Write to CSV
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    dataset.to_csv(buf)
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        fh.write(buf.getvalue())


def _get_whisper_draft(audio_path: str) -> str:
    """Return a Whisper draft transcript for *audio_path*.

    Uses the already-cached ConversationAnalyzer's SpeechToText component if
    available in st.cache_resource. Falls back to a note if not loaded.
    Draft is NEVER written as true_transcript — it is editable preview only.
    """
    try:
        # Access the shared cached analyzer if it's already loaded (avoids reloading models)
        from streamlit import runtime
        from app.speech_to_text import SpeechToText

        @st.cache_resource(show_spinner=False)
        def _get_stt() -> SpeechToText:
            return SpeechToText()

        stt = _get_stt()
        result = stt.transcribe(audio_path)
        text = str(result.get("text", "") or "").strip()
        return text if text else ""
    except Exception as e:
        return ""


# ── Main render function ──────────────────────────────────────────────────────

def render_annotation_page() -> None:
    """Render the Annotation Helper tab."""
    st.markdown(
        """
        <div class="section-header-bar">
            <div class="section-title">🏷️ Annotation Helper</div>
            <div class="section-badge">GROUND TRUTH LABELLING TOOL</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Label real recordings for evaluation. Whisper draft transcripts are editable starting points "
        "and are **never** automatically saved as ground truth. "
        "Leave any field blank if you are not confident — blank = excluded from metrics, never a class."
    )

    csv_path = str(Path(EVAL_DATASET_PATH))

    # ── Load dataset ─────────────────────────────────────────────────────────
    if "annotation_dataset_cache" not in st.session_state:
        st.session_state["annotation_dataset_cache"] = None

    dataset: EvaluationDataset = _load_dataset(csv_path)
    # Keep in session to track across saves
    st.session_state["annotation_dataset_cache"] = dataset

    # ── File browser ──────────────────────────────────────────────────────────
    st.markdown("### 📁 Select Audio File")

    audio_files = _list_audio_files([SAMPLE_AUDIO_DIR, RECORDINGS_DIR])

    if not audio_files:
        st.info(
            "No audio files found in `sample_audio/` or `recordings/`. "
            "Add WAV recordings following `docs/DATA_COLLECTION_GUIDE.md`."
        )
        return

    # Annotated vs unannotated breakdown
    annotated = {
        Path(s.filename).as_posix()
        for s in dataset.samples
        if any([
            s.true_voice_emotion,
            s.true_text_sentiment,
            s.true_transcript,
            s.true_speaker_count is not None,
            s.true_escalation,
            s.true_incongruent,
        ])
    }

    def _label_file(fp: str) -> str:
        tag = "✅" if Path(fp).as_posix() in annotated else "⬜"
        return f"{tag}  {Path(fp).name}"

    display_labels = [_label_file(fp) for fp in audio_files]

    col_filter, col_selector = st.columns([1, 3])
    with col_filter:
        show_only_unannotated = st.checkbox("Show unannotated only", value=False)

    if show_only_unannotated:
        filtered = [(lbl, fp) for lbl, fp in zip(display_labels, audio_files)
                    if Path(fp).as_posix() not in annotated]
    else:
        filtered = list(zip(display_labels, audio_files))

    if not filtered:
        st.success("All files in the selected scope are annotated.")
        return

    with col_selector:
        selected_label = st.selectbox(
            "Choose file",
            options=[lbl for lbl, _ in filtered],
            key="annotation_file_selector",
        )

    selected_file = dict(filtered)[selected_label]

    # ── Status banner ─────────────────────────────────────────────────────────
    existing_row = _row_for_file(dataset, selected_file)
    if existing_row:
        st.info(
            f"ℹ️ This file already has annotations from: "
            f"**{existing_row.annotator or 'unknown'}**. "
            "Saving will overwrite after confirmation."
        )

    st.divider()

    # ── Playback ──────────────────────────────────────────────────────────────
    st.markdown("### ▶️ Playback")
    try:
        with open(selected_file, "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/wav")
    except Exception as e:
        st.error(f"Could not load audio file: {e}")
        return

    st.divider()

    # ── Whisper draft ─────────────────────────────────────────────────────────
    st.markdown("### 📝 Whisper Draft Transcript")
    st.caption(
        "This is a machine-generated draft. It is **never** automatically saved as ground truth. "
        "Edit it to produce the transcript you want to annotate, then copy it to the field below."
    )

    draft_key = f"draft_{selected_file}"
    if st.button("🔄 Generate Whisper Draft", key=f"btn_draft_{Path(selected_file).stem}"):
        with st.spinner("Running Faster-Whisper (this may take a moment on first load)..."):
            draft = _get_whisper_draft(selected_file)
        st.session_state[draft_key] = draft

    draft_text = st.session_state.get(draft_key, "")
    editable_draft = st.text_area(
        "Editable draft (for reference only — not saved unless you copy it below)",
        value=draft_text,
        height=80,
        key=f"draft_area_{selected_file}",
    )

    st.divider()

    # ── Annotation form ───────────────────────────────────────────────────────
    st.markdown("### 🏷️ Ground-Truth Labels")
    st.caption("Fill in only the fields you are confident about. Blank = excluded from metrics.")

    pre = existing_row  # may be None

    with st.form(key=f"annotation_form_{Path(selected_file).stem}", clear_on_submit=False):
        col_a, col_b = st.columns(2)

        with col_a:
            emotion_opts = ["(blank — not annotated)"] + VOICE_EMOTION_OPTIONS
            default_emotion = (
                emotion_opts.index(pre.true_voice_emotion)
                if pre and pre.true_voice_emotion in emotion_opts
                else 0
            )
            voice_emotion = st.selectbox(
                "True voice emotion",
                options=emotion_opts,
                index=default_emotion,
                help="Dominant vocal/acoustic emotion. Leave blank if uncertain.",
            )

        with col_b:
            sentiment_opts = ["(blank — not annotated)"] + TEXT_SENTIMENT_OPTIONS
            default_sentiment = (
                sentiment_opts.index(pre.true_text_sentiment)
                if pre and pre.true_text_sentiment in sentiment_opts
                else 0
            )
            text_sentiment = st.selectbox(
                "True text sentiment",
                options=sentiment_opts,
                index=default_sentiment,
                help="Sentiment of the spoken words. Leave blank if no text present.",
            )

        col_c, col_d, col_e = st.columns(3)

        with col_c:
            spk_opts = ["(blank — not annotated)", "1", "2", "3", "4", "5+"]
            default_spk = (
                spk_opts.index(str(pre.true_speaker_count))
                if pre and pre.true_speaker_count is not None and str(pre.true_speaker_count) in spk_opts
                else 0
            )
            speaker_count = st.selectbox(
                "True speaker count",
                options=spk_opts,
                index=default_spk,
                help="Number of distinct speakers in the clip.",
            )

        with col_d:
            esc_opts = ["(blank — not annotated)", "yes", "no"]
            default_esc = (
                esc_opts.index(pre.true_escalation)
                if pre and pre.true_escalation in esc_opts
                else 0
            )
            escalation = st.selectbox(
                "True escalation",
                options=esc_opts,
                index=default_esc,
                help="Does emotional intensity clearly rise from start to end?",
            )

        with col_e:
            inc_opts = ["(blank — not annotated)", "yes", "no"]
            default_inc = (
                inc_opts.index(pre.true_incongruent)
                if pre and pre.true_incongruent in inc_opts
                else 0
            )
            incongruent = st.selectbox(
                "True incongruent",
                options=inc_opts,
                index=default_inc,
                help="Do voice and text express opposite emotional directions?",
            )

        transcript_input = st.text_area(
            "True transcript (verbatim)",
            value=pre.true_transcript or "",
            height=80,
            placeholder="Type or paste the verbatim transcript. Leave blank if unintelligible.",
            help="Ground-truth transcript. Use [inaudible] for unclear sections.",
        )

        col_note, col_ann = st.columns([3, 1])
        with col_note:
            notes_input = st.text_input(
                "Notes",
                value=pre.notes if pre else "",
                placeholder="Optional: recording quality, scenario ID, special conditions, …",
            )
        with col_ann:
            annotator_input = st.text_input(
                "Annotator initials",
                value=pre.annotator if pre else "",
                placeholder="e.g. SR, AK, RV",
                max_chars=20,
            )

        submitted = st.form_submit_button("💾 Save Annotation", type="primary")

    if submitted:
        # ── Confirmation for overwrite ────────────────────────────────────────
        if existing_row and "annotation_confirmed" not in st.session_state:
            st.warning(
                "⚠️ This file already has an annotation. Confirm overwrite?",
            )
            if st.button("Yes, overwrite", key="confirm_overwrite"):
                st.session_state["annotation_confirmed"] = True
                st.rerun()
            if st.button("Cancel", key="cancel_overwrite"):
                st.stop()
        else:
            # Clear confirmation flag for next use
            st.session_state.pop("annotation_confirmed", None)

            # Build EvaluationSample
            def _blank(val: str) -> Optional[str]:
                return None if val.startswith("(blank") else val

            spk_val = _blank(speaker_count)
            spk_int: Optional[int] = None
            if spk_val is not None:
                try:
                    spk_int = int(spk_val.replace("+", ""))
                except ValueError:
                    spk_int = None

            new_sample = EvaluationSample(
                filename=selected_file,
                true_voice_emotion=_blank(voice_emotion),
                true_text_sentiment=_blank(text_sentiment),
                true_speaker_count=spk_int,
                true_transcript=transcript_input.strip() or None,
                true_escalation=_blank(escalation),
                true_incongruent=_blank(incongruent),
                source="recordings" if selected_file.startswith(RECORDINGS_DIR) else "sample_audio",
                annotator=annotator_input.strip(),
                notes=notes_input.strip(),
            )

            try:
                _save_row(dataset, new_sample, csv_path)
                st.session_state["annotation_dataset_cache"] = dataset
                st.success(f"✅ Annotation saved for `{Path(selected_file).name}`.")
                # Reset draft cache for this file
                st.session_state.pop(draft_key, None)
                time.sleep(0.3)
                st.rerun()
            except Exception as e:
                st.error(f"❌ Failed to save annotation: {e}")

    st.divider()

    # ── Progress overview ─────────────────────────────────────────────────────
    st.markdown("### 📊 Annotation Progress")

    total_files = len(audio_files)
    n_annotated = len(annotated)
    n_remaining = total_files - n_annotated

    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    col_p1.metric("Total audio files", total_files)
    col_p2.metric("Annotated", n_annotated)
    col_p3.metric("Remaining", n_remaining)
    col_p4.metric(f"Min N for metrics (EVAL_MIN_N = {EVAL_MIN_N})", f"Need ≥{EVAL_MIN_N}")

    if total_files > 0:
        pct = int(100 * n_annotated / total_files)
        st.progress(pct / 100, text=f"{pct}% annotated")

    # Per-metric readiness
    n_voice = sum(1 for s in dataset.samples if s.true_voice_emotion)
    n_text = sum(1 for s in dataset.samples if s.true_text_sentiment)
    n_trans = sum(1 for s in dataset.samples if s.true_transcript)
    n_spk = sum(1 for s in dataset.samples if s.true_speaker_count is not None)
    n_esc = sum(1 for s in dataset.samples if s.true_escalation)
    n_inc = sum(1 for s in dataset.samples if s.true_incongruent)

    def _ready(n: int) -> str:
        if n == 0:
            return "❌ No data"
        if n < EVAL_MIN_N:
            return f"⚠️ N={n} (need ≥{EVAL_MIN_N})"
        return f"✅ N={n}"

    st.markdown("**Metric readiness:**")
    readiness = {
        "Voice emotion accuracy/F1": _ready(n_voice),
        "Text sentiment accuracy/F1": _ready(n_text),
        "WER / CER": _ready(n_trans),
        "Speaker-count MAE": _ready(n_spk),
        "Escalation P/R/F1": _ready(n_esc),
        "Incongruence P/R/F1": _ready(n_inc),
    }
    for metric, status in readiness.items():
        st.markdown(f"- **{metric}**: {status}")
