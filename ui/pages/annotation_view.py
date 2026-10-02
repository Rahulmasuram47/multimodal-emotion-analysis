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
from evaluation.dataset import (
    EvaluationDataset,
    EvaluationSample,
    EXTENDED_COLUMNS,
    get_dataset_inventory,
)

from scripts.merge_annotations import load_annotator_csvs, analyze_agreement, write_consensus_dataset
import wave


def _get_audio_duration_seconds(file_path: str) -> float:
    try:
        with wave.open(file_path, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return float(frames) / float(rate) if rate > 0 else 0.0
    except Exception:
        return 0.0


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
            <div class="section-badge">GROUND TRUTH LABELLING & VALIDATION TOOL</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Label real recordings for evaluation. Whisper draft transcripts are editable starting points "
        "and are **never** automatically saved as ground truth. "
        "Leave any field blank if you are not confident — blank = excluded from metrics, never a class."
    )

    subtab_annotate, subtab_dashboard, subtab_resolve = st.tabs([
        "🏷️ Annotate Recordings",
        "📊 Dataset Dashboard",
        "🤝 Multi-Annotator Consensus",
    ])

    audio_files = _list_audio_files([SAMPLE_AUDIO_DIR, RECORDINGS_DIR])

    with subtab_annotate:
        col_prof_sel, col_prof_info = st.columns([2, 3])
        with col_prof_sel:
            profile = st.selectbox(
                "Active Annotator Profile",
                options=[
                    "Consensus / Ground Truth (data/evaluation_dataset.csv)",
                    "Sahithi (data/annotations/annotator_SR.csv)",
                    "Akshitha (data/annotations/annotator_AK.csv)",
                    "Rahul (data/annotations/annotator_RV.csv)",
                ],
                key="annotator_profile_choice",
                help="Switch between annotators to record individual labels without overwriting peers.",
            )

        if "annotator_SR" in profile:
            csv_path = "data/annotations/annotator_SR.csv"
            default_init = "SR"
        elif "annotator_AK" in profile:
            csv_path = "data/annotations/annotator_AK.csv"
            default_init = "AK"
        elif "annotator_RV" in profile:
            csv_path = "data/annotations/annotator_RV.csv"
            default_init = "RV"
        else:
            csv_path = str(Path(EVAL_DATASET_PATH))
            default_init = ""

        with col_prof_info:
            st.info(f"Target storage: `{csv_path}`")

        # ── Load dataset ─────────────────────────────────────────────────────────
        dataset: EvaluationDataset = _load_dataset(csv_path)

        # ── File browser ──────────────────────────────────────────────────────────
        st.markdown("### 📁 Select Audio File")

        if not audio_files:
            st.info(
                "No audio files found in `sample_audio/` or `recordings/`. "
                "Add WAV recordings following `docs/DATA_COLLECTION_GUIDE.md`."
            )
        else:
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
                show_only_unannotated = st.checkbox("Show unannotated only", value=False, key="filter_unannotated_chk")

            if show_only_unannotated:
                filtered = [(lbl, fp) for lbl, fp in zip(display_labels, audio_files)
                            if Path(fp).as_posix() not in annotated]
            else:
                filtered = list(zip(display_labels, audio_files))

            if not filtered:
                st.success("All files in the selected scope are annotated.")
            else:
                with col_selector:
                    selected_label = st.selectbox(
                        "Choose file",
                        options=[lbl for lbl, _ in filtered],
                        key="annotation_file_selector",
                    )

                selected_file = dict(filtered)[selected_label]

                # Status banner
                existing_row = _row_for_file(dataset, selected_file)
                if existing_row:
                    st.info(
                        f"ℹ️ This file already has annotations from: "
                        f"**{existing_row.annotator or 'unknown'}**. "
                        "Saving will update this record."
                    )

                st.divider()

                # Playback
                st.markdown("### ▶️ Playback")
                try:
                    with open(selected_file, "rb") as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format="audio/wav")
                except Exception as e:
                    st.error(f"Could not load audio file: {e}")

                st.divider()

                # Whisper draft
                st.markdown("### 📝 Whisper Draft Transcript")
                st.caption(
                    "Machine-generated draft for convenience only. It is **never** automatically saved as ground truth."
                )

                draft_key = f"draft_{selected_file}"
                if st.button("🔄 Generate Whisper Draft", key=f"btn_draft_{Path(selected_file).stem}"):
                    with st.spinner("Running Faster-Whisper..."):
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

                # Annotation form
                st.markdown("### 🏷️ Ground-Truth Labels")
                pre = existing_row

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
                        value=(pre.true_transcript if pre else "") or "",
                        height=80,
                        placeholder="Type or paste the verbatim transcript. Leave blank if unintelligible.",
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
                            value=(pre.annotator if pre else default_init) or default_init,
                            placeholder="e.g. SR, AK, RV",
                            max_chars=20,
                        )

                    submitted = st.form_submit_button("💾 Save Annotation", type="primary")

                if submitted:
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
                        st.success(f"Annotation saved to `{csv_path}` for `{Path(selected_file).name}`.")
                        st.session_state.pop(draft_key, None)
                        time.sleep(0.3)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to save annotation: {e}")

    # ── TAB 2: DATASET DASHBOARD ─────────────────────────────────────────────
    with subtab_dashboard:
        st.markdown("### 📊 Dataset Overview & Health Dashboard")
        consensus_dataset: EvaluationDataset = _load_dataset(str(Path(EVAL_DATASET_PATH)))

        total_clips = len(audio_files)
        total_dur_sec = sum(_get_audio_duration_seconds(f) for f in audio_files)
        mins = int(total_dur_sec // 60)
        secs = int(total_dur_sec % 60)

        # Item 28: Use shared get_dataset_inventory() function
        inv = get_dataset_inventory()

        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        d_col1.metric("Audio Files on Disk", inv.total_audio_files)
        d_col2.metric("Dataset Records", inv.total_dataset_records)
        d_col3.metric("Annotated Clips", inv.annotated_records_count)
        d_col4.metric(f"Min N Target ({EVAL_MIN_N})", f"{inv.annotated_records_count}/{EVAL_MIN_N}")

        with st.expander("ℹ️ Dataset Inventory & Gap Explanation (Item 28)", expanded=False):
            st.markdown(inv.gap_explanation)

        st.divider()

        st.markdown("#### 🎯 Class Balance (Consensus Dataset)")
        b_col1, b_col2, b_col3 = st.columns(3)
        with b_col1:
            st.markdown("**Voice Emotion Balance**")
            v_counts = {emo: 0 for emo in VOICE_EMOTION_OPTIONS}
            for s in consensus_dataset.samples:
                if s.true_voice_emotion in v_counts:
                    v_counts[s.true_voice_emotion] += 1
            for emo, count in v_counts.items():
                st.write(f"- `{emo}`: **{count}**")

        with b_col2:
            st.markdown("**Text Sentiment Balance**")
            t_counts = {sent: 0 for sent in TEXT_SENTIMENT_OPTIONS}
            for s in consensus_dataset.samples:
                if s.true_text_sentiment in t_counts:
                    t_counts[s.true_text_sentiment] += 1
            for sent, count in t_counts.items():
                st.write(f"- `{sent}`: **{count}**")

        with b_col3:
            st.markdown("**Speaker Count Balance**")
            spk_counts = {"1": 0, "2": 0, "3+": 0}
            for s in consensus_dataset.samples:
                if s.true_speaker_count == 1:
                    spk_counts["1"] += 1
                elif s.true_speaker_count == 2:
                    spk_counts["2"] += 1
                elif s.true_speaker_count is not None and s.true_speaker_count >= 3:
                    spk_counts["3+"] += 1
            for spk, count in spk_counts.items():
                st.write(f"- `{spk} speaker(s)`: **{count}**")

        st.divider()
        st.markdown("#### 📋 Missing Annotations Checklist per Clip")
        checklist_data = []
        norm_samples = {Path(s.filename).as_posix(): s for s in consensus_dataset.samples}
        for f in audio_files:
            norm = Path(f).as_posix()
            s = norm_samples.get(norm)
            dur = _get_audio_duration_seconds(f)
            checklist_data.append({
                "File": Path(f).name,
                "Duration": f"{dur:.1f}s",
                "Voice Emotion": f"✅ {s.true_voice_emotion}" if s and s.true_voice_emotion else "❌ Missing",
                "Text Sentiment": f"✅ {s.true_text_sentiment}" if s and s.true_text_sentiment else "❌ Missing",
                "Transcript": "✅ Present" if s and s.true_transcript else "❌ Missing",
                "Speaker Count": f"✅ {s.true_speaker_count}" if s and s.true_speaker_count is not None else "❌ Missing",
                "Escalation": f"✅ {s.true_escalation}" if s and s.true_escalation else "❌ Missing",
                "Annotator": s.annotator if s and s.annotator else "None",
            })
        st.dataframe(checklist_data, use_container_width=True)

    # ── TAB 3: MULTI-ANNOTATOR RESOLVER ──────────────────────────────────────
    with subtab_resolve:
        st.markdown("### 🤝 Multi-Annotator Agreement & Disagreement Resolver")
        st.caption(
            "Load multiple annotators' CSV files (`data/annotations/annotator_*.csv`), calculate Cohen's Kappa, "
            "and resolve conflicts to build a consensus dataset."
        )

        ann_data = load_annotator_csvs("data/annotations")
        if not ann_data or len(ann_data) < 2:
            st.info(
                "Fewer than 2 annotators found in `data/annotations/`. "
                "To use multi-annotator validation, have at least two annotators save records under their respective profiles (e.g., Sahithi SR, Akshitha AK, Rahul RV)."
            )
        else:
            analysis = analyze_agreement(ann_data)
            annotators = analysis["annotators"]
            st.write(f"**Annotator Profiles Loaded ({len(annotators)}):** " + ", ".join(f"`{a}`" for a in annotators))

            # Cohen's Kappa Table
            st.markdown("#### 📐 Inter-Annotator Agreement (Cohen's Kappa)")
            k_col1, k_col2 = st.columns([2, 1])
            with k_col1:
                kappa_rows = []
                for fld, k_val in analysis["avg_field_kappas"].items():
                    k_disp = f"{k_val:.4f}" if k_val is not None else "N/A (insufficient overlap)"
                    interp = "Near Perfect" if k_val and k_val > 0.8 else ("Substantial" if k_val and k_val > 0.6 else "Moderate / Low")
                    kappa_rows.append({"Field": fld, "Average Cohen's Kappa": k_disp, "Agreement Interpretation": interp})
                st.dataframe(kappa_rows, use_container_width=True)

            with k_col2:
                st.metric("Consensus Clips", len(analysis["consensus_records"]))
                st.metric("Disagreements to Resolve", len(analysis["disagreements"]))

            # Disagreement Resolution View
            disagreements = analysis["disagreements"]
            if disagreements:
                st.markdown("#### ⚡ Resolve Disagreements")
                st.write("Conflicting labels detected. Select the agreed-upon consensus ground truth for each file:")
                for idx, dis in enumerate(disagreements):
                    fpath = dis["file"]
                    fname = Path(fpath).name
                    with st.expander(f"⚠️ {fname} — Conflicts in {list(dis['conflicts'].keys())}", expanded=True):
                        st.write(f"**Annotators who labeled this file:** {', '.join(dis['annotators'])}")
                        resolved_fields = {}
                        for field, annotator_votes in dis["conflicts"].items():
                            options = sorted(list(set(annotator_votes.values())))
                            vote_str = ", ".join(f"{a}: '{v}'" for a, v in annotator_votes.items())
                            selected_val = st.radio(
                                f"Consensus for **{field}** (Votes: {vote_str})",
                                options=options,
                                key=f"res_{fname}_{field}_{idx}",
                            )
                            resolved_fields[field] = selected_val

                        if st.button(f"Apply Resolution for {fname}", key=f"btn_res_{fname}_{idx}"):
                            row = dis["partial_row"]
                            row.update(resolved_fields)
                            analysis["consensus_records"][fpath] = row
                            write_consensus_dataset(analysis["consensus_records"], str(Path(EVAL_DATASET_PATH)))
                            st.success(f"Resolved and committed `{fname}` to consensus dataset!")
                            st.rerun()

            if st.button("💾 Export All Consensus Records to evaluation_dataset.csv", type="primary"):
                saved = write_consensus_dataset(analysis["consensus_records"], str(Path(EVAL_DATASET_PATH)))
                st.success(f"Successfully saved {saved} consensus records to `{EVAL_DATASET_PATH}`.")

