"""
Multimodal Validation & Empirical Benchmarks page orchestrator (Tab 3).

Reads the latest evaluation results from evaluation/results/latest_evaluation.json
if available. Shows N alongside every metric, 95% bootstrap CIs, and "Insufficient data"
banners when N < EVAL_MIN_N (configurable in config.py).

Does NOT run the full pipeline inline — results are produced by:
    python -m evaluation.run_evaluation
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st

from config import EVAL_MIN_N, EVAL_DATASET_PATH


RESULTS_JSON_PATH = "evaluation/results/latest_evaluation.json"
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


def _render_metric_block(
    label: str,
    data: Dict[str, Any],
    metric_key: str,
    ci_key: Optional[str] = None,
    n_key: str = "n",
) -> None:
    """Render a single metric with N badge and CI."""
    n = int(data.get(n_key, data.get("total_samples", data.get("count", 0))) or 0)
    val = data.get(metric_key)
    ci = data.get(ci_key) if ci_key else None

    badge = _n_badge(n)
    val_str = _fmt(val) if (n >= EVAL_MIN_N and val is not None) else "Insufficient data"
    ci_display = _ci_str(ci) if (n >= EVAL_MIN_N and ci) else ""

    col1, col2, col3 = st.columns([3, 2, 2])
    col1.markdown(f"**{label}**")
    col2.markdown(val_str + (f"  `95% CI {ci_display}`" if ci_display else ""))
    col3.markdown(badge)


def _render_classification_section(title: str, data: Dict[str, Any]) -> None:
    """Render accuracy / F1 / confusion matrix for a classification task."""
    st.markdown(f"#### {title}")
    status = data.get("status", "NO_DATA")
    n = int(data.get("n", data.get("total_samples", 0)) or 0)

    if status == "NO_DATA" or n == 0:
        st.info(f"No annotated ground truth for {title}. Add labels via the Annotation Helper.")
        return

    if n < EVAL_MIN_N:
        st.warning(
            f"Insufficient data: N={n} (need ≥{EVAL_MIN_N} for reliable metrics). "
            "Continue annotating via the Annotation Helper tab."
        )
        return

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
        import pandas as pd
        rows = {true_lbl: {pred_lbl: cm.get(true_lbl, {}).get(pred_lbl, 0) for pred_lbl in labels}
                for true_lbl in labels}
        df_cm = pd.DataFrame(rows).T.reindex(index=labels, columns=labels, fill_value=0)
        st.dataframe(df_cm, use_container_width=False)


def render_multimodal_validation_section() -> None:
    """Render empirical validation metrics tab."""
    st.markdown(
        """
        <div class="section-header-bar">
            <div class="section-title">📊 Multimodal Validation & Empirical Benchmarks</div>
            <div class="section-badge">OFFLINE LABELED DATASET EVALUATION</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
    total = metadata.get("total_dataset_records", 0)
    evaluated = metadata.get("evaluated_samples_count", 0)
    failed = metadata.get("failed_samples_count", 0)
    has_gt = metadata.get("has_ground_truth", False)
    missing = metadata.get("missing_annotations", [])

    # Summary banner
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset records", total)
    col2.metric("Evaluated", evaluated)
    col3.metric("Failed", failed)
    col4.metric("Ground truth present", "Yes" if has_gt else "No")

    if missing:
        st.warning(
            "**Pending annotations** (metrics will not display until these are filled):\n"
            + "\n".join(f"- {m}" for m in missing)
        )

    st.divider()

    # ── Voice emotion ──────────────────────────────────────────────────────────
    st.markdown("### 1. Voice Emotion (Acoustic)")
    _render_classification_section("Voice Emotion Metrics", results.get("voice_emotion", {}))

    st.divider()

    # ── Text sentiment ─────────────────────────────────────────────────────────
    st.markdown("### 2. Text Sentiment (Linguistic)")
    _render_classification_section("Text Sentiment Metrics", results.get("text_sentiment", {}))

    st.divider()

    # ── STT / WER / CER ───────────────────────────────────────────────────────
    st.markdown("### 3. Speech-to-Text Quality (WER / CER)")
    stt = results.get("speech_to_text", {})
    stt_n = stt.get("total_evaluated", 0)

    if stt.get("status") == "NO_DATA" or stt_n == 0:
        st.info("No reference transcripts annotated. Add via Annotation Helper.")
    elif stt_n < EVAL_MIN_N:
        st.warning(f"Insufficient data: N={stt_n} (need ≥{EVAL_MIN_N}).")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Average WER", _fmt(stt.get("average_wer")))
        c2.metric("N evaluated", stt_n)
        c3.metric("Total ref words", stt.get("total_reference_words", "—"))

    st.divider()

    # ── Speaker count ─────────────────────────────────────────────────────────
    st.markdown("### 4. Speaker Count (MAE)")
    spk = results.get("speaker_count", {})
    spk_n = spk.get("count", spk.get("n", 0))

    if spk.get("status") == "NO_DATA" or spk_n == 0:
        st.info("No speaker count ground truth annotated.")
    elif spk_n < EVAL_MIN_N:
        st.warning(f"Insufficient data: N={spk_n} (need ≥{EVAL_MIN_N}).")
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
    ablation_table = results.get("ablation_comparison_table", "")
    if ablation_table:
        st.code(ablation_table, language=None)
    else:
        st.info("Ablation table not available in current results.")

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
