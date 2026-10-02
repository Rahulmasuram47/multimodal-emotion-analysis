"""
Why-This-Result explainability and transparency page (Tab 3).
"""

import html
from typing import Any, Dict, List
import streamlit as st

from ui.utils import safe_float


def render_why_result_page(
    result: Dict[str, Any],
    segments: List[Dict[str, Any]],
) -> None:
    """Render Tab 3: Detailed Why-This-Result audit, rule evidence trail, and counterfactual sensitivity."""
    st.markdown(
        """
        <div class="section-header-bar">
            <div class="section-title">🔎 Why This Result? — Explainable Evidence Trail</div>
            <div class="section-badge">DETERMINISTIC RULE AUDIT & SENSITIVITY</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
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
        """,
        unsafe_allow_html=True,
    )

    # 1. Conversation Verdict Sensitivity
    verdict = result.get("verdict", {}) or {}
    verdict_cfs = verdict.get("counterfactuals", [])
    if verdict_cfs:
        st.markdown(
            f"""
            <div style="background: var(--panel-2); border-left: 3px solid var(--fused); border-radius: 6px; padding: 0.8rem 1.0rem; margin-bottom: 1.2rem;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; font-weight: 700; color: var(--fused); text-transform: uppercase; margin-bottom: 0.3rem;">
                    CONVERSATION VERDICT SENSITIVITY
                </div>
                <div style="font-size: 0.8rem; color: var(--text); margin-bottom: 0.4rem;">
                    Current Verdict: <strong>{html.escape(str(verdict.get('verdict', 'unknown')).upper())}</strong>
                    (slope: {safe_float(verdict.get('slope', 0.0)):+.3f}/turn, net change: {safe_float(verdict.get('net_change', 0.0)):+.2f})
                </div>
                <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.78rem; color: var(--muted); line-height: 1.45;">
                    {''.join(f'<li>{html.escape(c)}</li>' for c in verdict_cfs)}
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Segment-by-Segment Rule Execution & Counterfactual Table
    st.markdown(
        """
        <div class="section-header-bar" style="margin-top: 1.0rem;">
            <div class="section-title">📋 Turn-by-Turn Rule Audit & Counterfactuals</div>
            <div class="section-badge">FULL RECORDING TRACE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not segments:
        st.info("No segments available to audit.")
        return

    for s in segments:
        idx = s.get("segment_index", 0) + 1
        start = safe_float(s.get("start_seconds", 0.0))
        end = safe_float(s.get("end_seconds", 0.0))
        speaker = s.get("speaker", "Unknown")
        fusion = s.get("fusion", {}) or {}

        pat_label = str(fusion.get("pattern_label", "Standard Arbitration"))
        explanation = str(fusion.get("pattern_explanation", ""))
        rule_ids = fusion.get("rule_ids_fired", [])
        cfs = fusion.get("counterfactuals", [])

        eff_w = fusion.get("effective_weights", {}) or {}
        w_audio = safe_float(eff_w.get("audio_weight", 0.5))
        w_text = safe_float(eff_w.get("text_weight", 0.5))
        v_rel = safe_float(fusion.get("audio_reliability", 1.0))
        t_rel = safe_float(fusion.get("text_reliability", 1.0))

        with st.expander(f"Turn {idx:02d} [{start:.2f}s → {end:.2f}s] · Rule: {pat_label}", expanded=False):
            st.markdown(
                f"""
                <div style="font-size: 0.8rem; color: var(--text); margin-bottom: 0.5rem; line-height: 1.5;">
                    <strong>Fired Rule:</strong> <code>{', '.join(html.escape(r) for r in rule_ids) if rule_ids else 'default'}</code><br>
                    <strong>Explanation:</strong> {html.escape(explanation)}
                </div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 0.6rem; font-size: 0.75rem; color: var(--muted);">
                    <div style="background: var(--panel-2); padding: 0.5rem 0.75rem; border-radius: 4px;">
                        Effective Audio Weight: <strong style="color: var(--voice);">{w_audio:.2f}</strong> (reliability: {v_rel:.2f})
                    </div>
                    <div style="background: var(--panel-2); padding: 0.5rem 0.75rem; border-radius: 4px;">
                        Effective Text Weight: <strong style="color: var(--text-accent);">{w_text:.2f}</strong> (reliability: {t_rel:.2f})
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if cfs:
                st.markdown(
                    f"""
                    <div style="margin-top: 0.4rem; padding-top: 0.4rem; border-top: 1px dashed var(--border);">
                        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; font-weight: 700; color: var(--text-accent); text-transform: uppercase; margin-bottom: 0.25rem;">
                            Counterfactual Sensitivity:
                        </div>
                        <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.75rem; color: var(--text); line-height: 1.45;">
                            {''.join(f'<li>{html.escape(c)}</li>' for c in cfs)}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
