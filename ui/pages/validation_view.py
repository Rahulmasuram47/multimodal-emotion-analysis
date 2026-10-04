"""
Multimodal Validation & Empirical Benchmarks page orchestrator (Tab 4).

Reads the latest evaluation results from evaluation/results/latest_evaluation.json
if available. Shows N alongside every metric, 95% bootstrap CIs, and "Insufficient data"
banners when N < EVAL_MIN_N (configurable in config.py).

Tier 3 Fixes:
- Item 28: Uses shared get_dataset_inventory() to align audio files, records, and explain the gap.
- Item 29: Displays 'Insufficient data: N=4, need >=5' plus raw prediction-vs-label table (descriptive only, not a metric).
- Item 30: When all labeled clips are single-speaker, suppresses headline MAE and displays uninformative notice.
- Item 31: Renders the remaining-annotation checklist; defensively guarded against crashes on partial data.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from config import EVAL_DATASET_PATH, EVAL_MIN_N
from evaluation.dataset import format_remaining_annotation_checklist, get_dataset_inventory
from ui.utils import html_block


RESULTS_JSON_PATH = "evaluation/results/latest_evaluation.json"
PREDICTIONS_CSV_PATH = "evaluation/results/evaluation_predictions.csv"
REPORT_MD_PATH = "evaluation/results/EVALUATION_REPORT.md"


def _load_latest_results() -> Optional[Dict[str, Any]]:
    """Load the latest evaluation JSON produced by run_evaluation.py."""
    p = Path(RESULTS_JSON_PATH)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _n_badge(n: int) -> str:
    if n == 0:
        return "❌ No data"
    if n < EVAL_MIN_N:
        return f"⚠️ N={n} (need ≥{EVAL_MIN_N})"
    return f"N={n}"


def _fmt(val: Any, decimals: int = 4) -> str:
    if val is None:
        return "—"
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def _ci_str(ci: Optional[Dict[str, Any]]) -> str:
    """Format bootstrap CI as [lo, hi] string."""
    if ci is None:
        return ""
    lo = ci.get("ci_lower")
    hi = ci.get("ci_upper")
    if lo is None or hi is None:
        return ""
    return f"[{lo:.4f}, {hi:.4f}]"


def _load_prediction_table(modality: str) -> Optional[pd.DataFrame]:
    """Load and filter the raw prediction vs ground truth table from evaluation predictions (Item 29)."""
    p = Path(PREDICTIONS_CSV_PATH)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if modality == "voice":
            col_true = "true_voice_emotion"
            col_pred = "pred_voice_emotion"
            col_match = "voice_match"
        else:
            col_true = "true_text_sentiment"
            col_pred = "pred_text_sentiment"
            col_match = "sentiment_match"

        # Filter to rows that actually have ground truth
        valid_rows = df[df[col_true].notna() & (df[col_true].astype(str).str.strip().str.upper() != "N/A")].copy()
        if valid_rows.empty:
            return None

        # Build clean descriptive dataframe
        res = pd.DataFrame({
            "Recording": valid_rows["filename"].apply(lambda f: Path(str(f)).name),
            "Ground Truth": valid_rows[col_true].astype(str),
            "Model Prediction": valid_rows[col_pred].astype(str),
            "Match": valid_rows[col_match].apply(lambda m: "✅ Yes" if str(m).strip().lower() in ("true", "yes", "1") else "❌ No"),
            "Notes": valid_rows["notes"].fillna("").astype(str),
        })
        return res
    except Exception:
        return None


def _render_classification_section(title: str, data: Dict[str, Any], modality_type: str = "voice") -> None:
    """Render accuracy / F1 / confusion matrix or raw descriptive table when N < EVAL_MIN_N.

    Item 29: Shows 'Insufficient data: N=X, need >=5' plus a raw prediction-vs-label
    table marked 'descriptive only, not a metric'.
    """
    st.markdown(f"#### {title}")
    status = data.get("status", "NO_DATA")
    # Defensively inspect multiple key names (Item 29 bugfix)
    n = int(data.get("evaluated_samples", data.get("n", data.get("total_samples", 0))) or 0)

    if status == "NO_DATA" or n == 0:
        st.info(f"No annotated ground truth for {title}. Add labels via the Annotation Helper tab.")
        return

    # Item 29: Small sample size (0 < n < EVAL_MIN_N)
    if n < EVAL_MIN_N:
        st.warning(
            f"⚠️ **Insufficient data: N={n}, need ≥{EVAL_MIN_N} for statistically valid metrics.** "
            f"The current sample size is too small to report meaningful accuracy or F1 scores."
        )

        df_table = _load_prediction_table(modality_type)
        if df_table is not None and not df_table.empty:
            st.markdown(
                "**Raw Prediction vs. Label Table** *(descriptive only, not a metric)*:"
            )
            st.dataframe(df_table, use_container_width=True, hide_index=True)
            st.caption(
                f"ℹ️ *Descriptive comparison only (N={n}) — not a benchmark metric. "
                f"Requires at least {EVAL_MIN_N} annotated recordings before computing formal accuracy/F1.*"
            )
        else:
            st.caption(f"Raw predictions table not yet generated. Run `python -m evaluation.run_evaluation`.")
        return

    # Sufficient data (n >= EVAL_MIN_N)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", _fmt(data.get("accuracy")))
    col2.metric("Macro F1", _fmt(data.get("f1_macro")))
    col3.metric("Macro Precision", _fmt(data.get("precision_macro")))
    col4.metric("Macro Recall", _fmt(data.get("recall_macro")))

    acc_ci = data.get("accuracy_ci")
    f1_ci = data.get("f1_macro_ci")
    if acc_ci and acc_ci.get("ci_lower") is not None:
        st.caption(
            f"Accuracy 95% CI: {_ci_str(acc_ci)} | "
            f"F1 95% CI: {_ci_str(f1_ci) if f1_ci else '—'} | N={n}"
        )

    # Confusion matrix
    cm = data.get("confusion_matrix", {})
    if cm:
        labels = data.get("labels", list(cm.keys()))
        st.markdown("**Confusion matrix** (rows = true, cols = predicted):")
        rows = {true_lbl: {pred_lbl: cm.get(true_lbl, {}).get(pred_lbl, 0) for pred_lbl in labels}
                for true_lbl in labels}
        df_cm = pd.DataFrame(rows).T.reindex(index=labels, columns=labels, fill_value=0)
        st.dataframe(df_cm, use_container_width=False)


def render_multimodal_validation_section() -> None:
    """Render empirical validation metrics tab (Item 28, 29, 30, 31)."""
    st.markdown(
        html_block("""
<div class="section-header-bar">
<div class="section-title">📊 Multimodal Validation &amp; Empirical Benchmarks</div>
<div class="section-badge">OFFLINE LABELED DATASET EVALUATION</div>
</div>
"""),
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Item 28: Shared inventory function for Annotation Helper and Validation tabs
    # -------------------------------------------------------------------------
    inv = get_dataset_inventory()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Audio Files on Disk", inv.total_audio_files)
    col2.metric("Dataset Records", inv.total_dataset_records)
    col3.metric("Annotated Clips", inv.annotated_records_count)
    col4.metric("Test Fixtures", inv.fixture_files_count)

    with st.expander("ℹ️ Dataset Inventory & Gap Explanation (Item 28)", expanded=False):
        st.markdown(inv.gap_explanation)

    results = _load_latest_results()

    if results is None:
        st.info(
            "No evaluation results found. Run:\n"
            "```\npython -m evaluation.run_evaluation\n```\n"
            "Results will appear here once computed on annotated data."
        )
        _render_annotation_status()
        return

    metadata = results.get("metadata", {})
    evaluated = metadata.get("evaluated_samples_count", 0)
    failed = metadata.get("failed_samples_count", 0)
    has_gt = metadata.get("has_ground_truth", False)
    missing = metadata.get("missing_annotations", [])

    st.caption(
        f"Evaluated: **{evaluated}** | Failed: **{failed}** | Ground truth present: **{'Yes' if has_gt else 'No'}**"
    )

    if missing:
        st.warning(
            "**Pending annotations** (metrics will not display until these are filled):\n"
            + "\n".join(f"- {m}" for m in missing)
        )

    st.divider()

    # ── Voice emotion ──────────────────────────────────────────────────────────
    st.markdown("### 1. Voice Emotion (Acoustic)")
    _render_classification_section(
        "Voice Emotion Metrics",
        results.get("voice_emotion", {}),
        modality_type="voice",
    )

    st.divider()

    # ── Text sentiment ─────────────────────────────────────────────────────────
    st.markdown("### 2. Text Sentiment (Linguistic)")
    _render_classification_section(
        "Text Sentiment Metrics",
        results.get("text_sentiment", {}),
        modality_type="text",
    )

    st.divider()

    # ── STT / WER / CER ───────────────────────────────────────────────────────
    st.markdown("### 3. Speech-to-Text Quality (WER / CER)")
    stt = results.get("speech_to_text", {})
    stt_n = stt.get("total_evaluated", 0)

    if stt.get("status") == "NO_DATA" or stt_n == 0:
        st.info("No reference transcripts annotated. Add via Annotation Helper.")
    elif stt_n < EVAL_MIN_N:
        st.warning(f"⚠️ Insufficient data: N={stt_n} (need ≥{EVAL_MIN_N} for statistically valid WER).")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Average WER", _fmt(stt.get("average_wer")))
        c2.metric("N evaluated", stt_n)
        c3.metric("Total ref words", stt.get("total_reference_words", "—"))

    st.divider()

    # ── Speaker count ─────────────────────────────────────────────────────────
    # Item 30: When all labeled clips are single-speaker, suppress headline MAE
    st.markdown("### 4. Speaker Count (MAE)")
    spk = results.get("speaker_count", {})
    spk_n = int(spk.get("count", spk.get("n", 0)) or 0)
    per_sample = spk.get("per_sample", []) or []

    all_single_speaker = (
        spk_n > 0 and
        all(int(p.get("true_speaker_count", 0)) == 1 for p in per_sample)
    )

    if spk.get("status") == "NO_DATA" or spk_n == 0:
        st.info("No speaker count ground truth annotated.")
    elif all_single_speaker:
        mae_val = _fmt(spk.get("mae"))
        st.info(
            f"ℹ️ **Notice on Speaker Metrics**: All {spk_n} labeled clips are single-speaker "
            f"(true_speaker_count=1); this MAE ({mae_val}) is uninformative for multi-speaker "
            f"separation until multi-speaker conversations (2+ speakers) are annotated and evaluated."
        )
        if per_sample:
            rows = []
            for idx, p in enumerate(per_sample):
                t_c = p.get("true_speaker_count")
                p_c = p.get("predicted_speaker_count")
                diff_val = (
                    abs(int(p_c) - int(t_c))
                    if (t_c is not None and p_c is not None)
                    else abs(int(p.get("diff", 0) or 0))
                )
                rows.append({
                    "Clip #": idx + 1,
                    "True Count": t_c,
                    "Predicted Count": p_c,
                    "Absolute Difference": diff_val,
                })
            df_spk = pd.DataFrame(rows)
            st.dataframe(df_spk, use_container_width=True, hide_index=True)
            st.caption("Descriptive breakdown (headline MAE card suppressed because all labeled clips are single-speaker).")
    elif spk_n < EVAL_MIN_N:
        st.warning(f"⚠️ Insufficient data: N={spk_n} (need ≥{EVAL_MIN_N}).")
    else:
        c1, c2 = st.columns(2)
        c1.metric("MAE", _fmt(spk.get("mae")))
        c2.metric("Exact match accuracy", _fmt(spk.get("exact_match_accuracy")))
        mae_ci = spk.get("mae_ci")
        if mae_ci and mae_ci.get("ci_lower") is not None:
            st.caption(f"MAE 95% CI: {_ci_str(mae_ci)} | N={spk_n}")

    st.divider()

    # ── Ablation comparison ───────────────────────────────────────────────────
    st.markdown("### 5. Modality Ablation Comparison")
    ablation_data = [
        {"Modality Configuration": "Voice Only", "Accuracy": "Insufficient data (N=0)", "Macro F1": "Insufficient data (N=0)"},
        {"Modality Configuration": "Text Only", "Accuracy": "Insufficient data (N=0)", "Macro F1": "Insufficient data (N=0)"},
        {"Modality Configuration": "Rule-Based Fusion", "Accuracy": "Insufficient data (N=0)", "Macro F1": "Insufficient data (N=0)"},
        {"Modality Configuration": "Learned Fusion", "Accuracy": "Insufficient data (N=0)", "Macro F1": "Insufficient data (N=0)"},
    ]
    st.dataframe(pd.DataFrame(ablation_data), use_container_width=True, hide_index=True)

    st.divider()

    # ── Learned fusion ────────────────────────────────────────────────────────
    st.markdown("### 6. Learned Fusion Baseline (Experimental)")
    st.caption(
        "LogisticRegression on concatenated acoustic + linguistic features. "
        "Experimental comparison baseline only — not the production path."
    )
    lf = results.get("learned_fusion", {})
    lf_status = lf.get("status", "INSUFFICIENT_DATA")
    if lf_status == "OK":
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", _fmt(lf.get("accuracy")))
        c2.metric("Macro F1", _fmt(lf.get("f1_macro")))
        c3.metric("Message", lf.get("message", ""))
    else:
        st.info(f"Learned fusion: {lf.get('message', 'Insufficient labelled data.')}")

    # Remaining-annotation checklist
    st.divider()
    st.markdown("### 📋 Remaining-Annotation Checklist")
    import html as _html
    checklist_str = format_remaining_annotation_checklist()
    checklist_lines = [l for l in checklist_str.split("\n") if l.strip() and not l.startswith("=")]
    checklist_content = "<br>".join(_html.escape(l) for l in checklist_lines)
    st.markdown(
        html_block(f"""
<div style="background: var(--panel-2); border: 1px solid var(--border); border-radius: 6px; padding: 1.0rem; font-family: 'IBM Plex Mono', monospace; font-size: 0.80rem; line-height: 1.6; color: var(--text);">
{checklist_content}
</div>
"""),
        unsafe_allow_html=True,
    )

    # Link to full markdown report
    if Path(REPORT_MD_PATH).exists():
        st.divider()
        st.markdown(f"📄 Full report saved at `{REPORT_MD_PATH}`")

    # Mandatory disclaimer
    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** This system performs computational emotion inference from acoustic and linguistic "
        "patterns. It does not diagnose, assess, or make claims about actual psychological or emotional states. "
        "Model confidence reflects classification certainty, not ground-truth correctness."
    )

    _render_annotation_status()


def _render_annotation_status() -> None:
    """Show quick link to the annotation helper."""
    st.divider()
    st.markdown(
        f"🏷️ **Add ground-truth labels** using the **Annotation Helper** tab. "
        f"Evaluation CSV: `{EVAL_DATASET_PATH}` | Min N per metric: **{EVAL_MIN_N}**"
    )
