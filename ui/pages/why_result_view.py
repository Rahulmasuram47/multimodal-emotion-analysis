"""
Why-This-Result explainability and transparency page (Tab 3).
"""

import html
from typing import Any, Dict, List
import pandas as pd
import streamlit as st

from ui.utils import html_block, safe_float


def render_why_result_page(
    result: Dict[str, Any],
    segments: List[Dict[str, Any]],
) -> None:
    """Render Tab 3: Detailed Why-This-Result audit, rule evidence trail, and counterfactual sensitivity.

    Item 19: Replaced per-turn expanders (which duplicated the segment breakdown)
    with a single structured summary table of rule IDs, effective weights,
    reliability scores, and counterfactual sensitivity per turn.
    Item 7: Uses html_block to avoid markdown code-block rendering.
    """
    st.markdown(
        html_block("""
<div class="section-header-bar">
<div class="section-title">🔎 Why This Result? — Explainable Evidence Trail</div>
<div class="section-badge">DETERMINISTIC RULE AUDIT &amp; SENSITIVITY</div>
</div>
"""),
        unsafe_allow_html=True,
    )

    st.markdown(
        html_block("""
<div style="background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 1.0rem; margin-bottom: 1.2rem;">
<div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; font-weight: 700; color: var(--text-accent); text-transform: uppercase; margin-bottom: 0.4rem;">
SYSTEM TRANSPARENCY GUARANTEE
</div>
<div style="font-size: 0.82rem; color: var(--text); line-height: 1.55;">
Every pattern label, confidence score, and conversation verdict is produced by <strong>deterministic, explainable rules</strong>.
No black-box machine learning models are used in multimodal arbitration or escalation detection.
Below is the complete evidence trail of every rule evaluated, reliability adjustments applied, and counterfactual sensitivities.
</div>
</div>
"""),
        unsafe_allow_html=True,
    )

    # 1. Conversation Verdict Sensitivity
    verdict = result.get("verdict", {}) or {}
    verdict_cfs = verdict.get("counterfactuals", [])
    if verdict_cfs:
        cfs_items = "".join(f"<li>{html.escape(c)}</li>" for c in verdict_cfs)
        v_label = str(verdict.get("verdict", "unknown")).upper()
        slope_val = safe_float(verdict.get("slope", 0.0))
        net_val = safe_float(verdict.get("net_change", 0.0))
        st.markdown(
            html_block(f"""
<div style="background: var(--panel-2); border-left: 3px solid var(--fused); border-radius: 6px; padding: 0.8rem 1.0rem; margin-bottom: 1.2rem;">
<div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; font-weight: 700; color: var(--fused); text-transform: uppercase; margin-bottom: 0.3rem;">
CONVERSATION VERDICT SENSITIVITY
</div>
<div style="font-size: 0.8rem; color: var(--text); margin-bottom: 0.4rem;">
Current Verdict: <strong>{html.escape(v_label)}</strong>
(slope: {slope_val:+.3f}/turn, net change: {net_val:+.2f})
</div>
<ul style="margin: 0; padding-left: 1.2rem; font-size: 0.78rem; color: var(--muted); line-height: 1.45;">
{cfs_items}
</ul>
</div>
"""),
            unsafe_allow_html=True,
        )

    # 2. Item 19: Single table of rule IDs per turn (no duplicate expanders)
    st.markdown(
        html_block("""
<div class="section-header-bar" style="margin-top: 1.0rem;">
<div class="section-title">📋 Turn-by-Turn Rule Audit Matrix</div>
<div class="section-badge">RULE IDS · WEIGHTS · RELIABILITY · COUNTERFACTUALS</div>
</div>
"""),
        unsafe_allow_html=True,
    )

    if not segments:
        st.info("No segments available to audit.")
        return

    table_rows = []
    for idx, s in enumerate(segments, 1):
        start = safe_float(s.get("start_seconds", 0.0))
        end = safe_float(s.get("end_seconds", 0.0))
        fusion = s.get("fusion", {}) or {}

        pat_label = str(fusion.get("pattern_label", "Standard Arbitration"))
        rule_ids = fusion.get("rule_ids_fired", []) or ["default"]
        rule_str = ", ".join(str(r) for r in rule_ids)

        eff_w = fusion.get("effective_weights", {}) or {}
        w_audio = safe_float(eff_w.get("audio_weight", 0.5))
        w_text = safe_float(eff_w.get("text_weight", 0.5))
        weights_str = f"Audio: {w_audio:.2f} / Text: {w_text:.2f}"

        v_rel = safe_float(fusion.get("audio_reliability", 1.0))
        t_rel = safe_float(fusion.get("text_reliability", 1.0))
        rel_str = f"Voice: {v_rel:.2f} / Text: {t_rel:.2f}"

        cfs = fusion.get("counterfactuals", []) or []
        cfs_summary = "; ".join(cfs) if cfs else "Robust (no sensitivity threshold met)"

        table_rows.append(
            {
                "Turn": f"Turn {idx:02d}",
                "Time Window": f"[{start:.2f}s → {end:.2f}s]",
                "Rule Fired": rule_str,
                "Pattern Label": pat_label,
                "Effective Weights": weights_str,
                "Modality Reliability": rel_str,
                "Counterfactual Sensitivity": cfs_summary,
            }
        )

    df = pd.DataFrame(table_rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )
