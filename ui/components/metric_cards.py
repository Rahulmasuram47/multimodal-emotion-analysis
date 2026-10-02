"""
Overview metrics, speaker metrics, and conversational dynamics cards.
"""

import html
from typing import Any, Dict, List, Optional
import altair as alt
import pandas as pd
import streamlit as st

from ui.utils import emotion_icon, html_block, safe_float

# Phase 4: icons for the unified verdict labels (kept separate from

# emotion_icon, which maps emotion words like happy/sad, not verdict
# words like escalating/fluctuating).
_VERDICT_ICONS: Dict[str, str] = {
    "escalating": "📈",
    "de-escalating": "📉",
    "fluctuating": "🔀",
    "stable": "➖",
    "insufficient_data": "❔",
}


def render_overview_metrics(
    language: str,
    language_prob: float,
    duration: float,
    segments: List[Dict[str, Any]],
    skipped_segments: int,
    trend_label: str,
    trend_delta: float,
    escalation_detected: bool,
    latency_val: str,
    verdict_label: Optional[str] = None,
) -> None:
    """Render top system-level metrics.

    Phase 4: an optional 7th "CONVERSATION VERDICT" card is shown when
    verdict_label is provided (result["verdict"]["verdict"]). It sits
    alongside the original 6 cards — including the original raw
    EMOTIONAL TREND and ESCALATION STATUS cards below, which are kept
    unchanged for transparency/ablation rather than removed.
    """
    st.markdown(
        html_block("""
<div class="section-header-bar">
<div class="section-title">📊 Analysis Overview</div>
<div class="section-badge">SYSTEM LEVEL METRICS</div>
</div>
"""),
        unsafe_allow_html=True,
    )


    n_cols = 7 if verdict_label else 6
    cols = st.columns(n_cols)
    m1, m2, m3, m4, m5, m6 = cols[:6]
    m7 = cols[6] if verdict_label else None

    with m1:
        with st.container(key="metric_language"):
            st.metric(
                label="LANGUAGE",
                value=language,
                delta=f"{language_prob:.1%} Conf",
                delta_color="off",
            )

    with m2:
        with st.container(key="metric_duration"):
            st.metric(
                label="AUDIO DURATION",
                value=f"{duration:.2f}s",
                delta=f"{len(segments)} Turns",
                delta_color="off",
            )

    with m3:
        with st.container(key="metric_segments"):
            st.metric(
                label="PROCESSED SEGMENTS",
                value=f"{len(segments)}",
                delta=f"{skipped_segments} Skipped" if skipped_segments > 0 else "100% Valid",
                delta_color="off",
            )

    with m4:
        st.metric(
            label="EMOTIONAL TREND",
            value=f"{emotion_icon(trend_label)} {trend_label}",
            delta=f"{trend_delta:+.4f} Δ",
            delta_color="inverse" if trend_delta > 0.10 else "normal",
        )

    with m5:
        if escalation_detected:
            st.metric(
                label="ESCALATION STATUS",
                value="DETECTED 🚨",
                delta="Elevated Risk",
                delta_color="inverse",
            )
        else:
            st.metric(
                label="ESCALATION STATUS",
                value="NOT DETECTED ✓",
                delta="Stable",
                delta_color="normal",
            )

    with m6:
        st.metric(
            label="PROCESSING TIME",
            value=latency_val,
            delta="Near-real-time",
            delta_color="off",
        )

    if m7 is not None:
        with m7:
            icon = _VERDICT_ICONS.get(verdict_label, "❔")
            display_label = verdict_label.replace("_", " ").title()
            is_negative_direction = verdict_label in ("escalating", "fluctuating")
            st.metric(
                label="CONVERSATION VERDICT",
                value=f"{icon} {display_label}",
                delta="Single unified signal",
                delta_color="inverse" if is_negative_direction else "normal",
            )


def render_speaker_metrics(
    unique_speakers: List[str],
    speaker_labels: List[str],
    diarization_error: str,
) -> None:
    """Render speaker identification statistics.

    Phase 6: shows an explicit success confirmation when diarization ran
    without error, not just a warning on failure — silence previously
    left it ambiguous whether diarization ran at all.
    """
    st.markdown(
        html_block("""
<div class="section-header-bar">
<div class="section-title">🗣️ Speaker Analysis</div>
<div class="section-badge">DIARIZATION OUTPUT</div>
</div>
"""),
        unsafe_allow_html=True,
    )


    if diarization_error:
        st.warning(
            "Speaker diarization failed on this analysis, so speakers show as Unknown. "
            f"Reason: {diarization_error}"
        )
    else:
        st.success("Speaker diarization ran successfully on this analysis.")

    sp1, sp2, sp3 = st.columns(3)
    with sp1:
        st.metric(
            "IDENTIFIED SPEAKERS",
            str(len(unique_speakers)) if unique_speakers else "Unknown",
        )
    with sp2:
        st.metric("LABELED TURNS", str(len(speaker_labels)))
    with sp3:
        speaker_display = (
            ", ".join(unique_speakers)
            if unique_speakers
            else "Unknown"
        )
        st.metric("SPEAKER LABELS", speaker_display)


def render_dynamics_card(
    trend: Dict[str, Any],
    escalation: Dict[str, Any],
    trend_label: str,
    escalation_detected: bool,
    escalation_reason: str,
    verdict: Optional[Dict[str, Any]] = None,
) -> None:
    """Render temporal emotional trajectory chart, the Phase 4 unified
    verdict panel, and the legacy explainable escalation card.

    verdict (optional): result["verdict"] — {"verdict", "reason", "slope",
    "net_change"}. When provided, a "Unified Conversation Verdict" panel
    is shown above the chart, framed as the primary interpretation. The
    original Trend/Escalation explanation below is kept unchanged as
    supporting detail — this is intentional: showing both side by side,
    with the unified verdict framed as authoritative, resolves the old
    "Trend: Increasing / Escalation: Not Detected" contradiction without
    hiding the legacy metrics that the ablation/report still relies on.
    """
    st.markdown(
        html_block("""
<div class="section-header-bar">
<div class="section-title">📈 Emotional Intensity Trajectory</div>
<div class="section-badge">MULTIMODAL TEMPORAL DYNAMICS</div>
</div>
"""),
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------
    # Phase 4: Unified Verdict panel (primary interpretation)
    # -------------------------------------------------------------
    if verdict:
        verdict_label = str(verdict.get("verdict", "")).lower()
        verdict_reason = str(verdict.get("reason", "")).strip()
        icon = _VERDICT_ICONS.get(verdict_label, "❔")
        display_label = verdict_label.replace("_", " ").title() if verdict_label else "Unknown"

        # Item 11: Fewer than 3 turns -> slope/net are n/a
        if verdict_label == "insufficient_data":
            stats_display = "slope=n/a &middot; net change=n/a"
        else:
            slope = safe_float(verdict.get("slope", 0.0))
            net_change = safe_float(verdict.get("net_change", 0.0))
            stats_display = f"slope={slope:+.3f}/turn &middot; net change={net_change:+.2f}"

        verdict_color_var = {
            "escalating": "var(--negative)",
            "fluctuating": "var(--negative)",
            "de-escalating": "var(--positive)",
            "stable": "var(--positive)",
            "insufficient_data": "var(--muted)",
        }.get(verdict_label, "var(--muted)")

        st.markdown(
            html_block(f"""
<div class="modality-card" style="border-left: 3px solid {verdict_color_var}; margin-bottom: 0.8rem;">
<div class="card-meta-label">UNIFIED CONVERSATION VERDICT</div>
<div style="display:flex;align-items:baseline;gap:0.6rem;margin-top:0.2rem;">
<div style="font-family:'IBM Plex Mono',monospace;font-weight:700;font-size:1.1rem;color:{verdict_color_var};">
{icon} {html.escape(display_label)}
</div>
<div style="font-size:0.76rem;color:var(--muted);">
{stats_display}
</div>
</div>
<div style="margin-top:0.4rem;font-size:0.8rem;color:var(--text);line-height:1.5;">
{html.escape(verdict_reason)}
</div>
</div>
"""),
            unsafe_allow_html=True,
        )


    scores = trend.get("intensity_scores", []) or []

    if scores:
        chart_data = pd.DataFrame(
            {
                "Segment": [f"Turn {i+1:02d}" for i in range(len(scores))],
                "Segment_Num": list(range(1, len(scores) + 1)),
                "Intensity": [max(0.0, min(1.0, safe_float(s))) for s in scores],
            }
        )

        base_chart = alt.Chart(chart_data).encode(
            x=alt.X(
                "Segment:O",
                title="Conversation Turn / Segment",
                axis=alt.Axis(
                    labelAngle=0,
                    labelFont="IBM Plex Mono",
                    titleFont="IBM Plex Mono",
                    titlePadding=10,
                ),
            ),
            y=alt.Y(
                "Intensity:Q",
                title="Multimodal Emotional Intensity",
                scale=alt.Scale(domain=[0.0, 1.0]),
                axis=alt.Axis(
                    format=".1f",
                    labelFont="IBM Plex Mono",
                    titleFont="IBM Plex Mono",
                    titlePadding=10,
                ),
            ),
        )

        area_layer = base_chart.mark_area(
            color=alt.Gradient(
                gradient="linear",
                stops=[
                    alt.GradientStop(color="rgba(232, 163, 61, 0.28)", offset=0),
                    alt.GradientStop(color="rgba(232, 163, 61, 0.02)", offset=1),
                ],
                x1=1,
                x2=1,
                y1=1,
                y2=0,
            ),
            opacity=0.8,
        )

        line_layer = base_chart.mark_line(
            color="#E8A33D",
            strokeWidth=2.5,
        )

        point_layer = base_chart.mark_point(
            color="#E8A33D",
            fill="#181D24",
            size=75,
            strokeWidth=2,
        ).encode(
            tooltip=[
                alt.Tooltip("Segment:N", title="Turn"),
                alt.Tooltip("Intensity:Q", title="Intensity Score", format=".4f"),
            ]
        )

        combined_chart = (
            (area_layer + line_layer + point_layer)
            .properties(height=260)
            .configure(
                background="#101318",
            )
            .configure_view(
                strokeWidth=0,
                fill="#181D24",
            )
            .configure_axis(
                labelColor="#8E97A6",
                titleColor="#8E97A6",
                gridColor="rgba(255, 255, 255, 0.06)",
                domainColor="rgba(255, 255, 255, 0.12)",
            )
        )

        st.altair_chart(combined_chart, width="stretch")

    if escalation_reason:
        st.markdown(
            html_block(f"""
<div style="background: var(--panel); border: 1px solid var(--border); border-left: 3px solid {'var(--negative)' if escalation_detected else 'var(--muted)'}; border-radius: 6px; padding: 0.75rem 1rem; margin-top: 0.5rem; font-size: 0.8rem; color: var(--text);">
<strong style="font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; color: {'var(--negative)' if escalation_detected else 'var(--muted)'};">{'🚨' if escalation_detected else 'ℹ️'} Escalation Diagnostics:</strong> {html.escape(escalation_reason)}
</div>
"""),
            unsafe_allow_html=True,
        )

    escalation_wording = "Detected" if escalation_detected else "Not Detected"
    trend_explanation = (
        "Overall intensity increased from the first to the last segment."
        if trend_label.lower() == "increasing"
        else "Overall intensity decreased from the first to the last segment."
        if trend_label.lower() == "decreasing"
        else "There was no strong first-to-last intensity change."
    )
    continuous_note = (
        "Every consecutive segment increased and the total rise met the escalation threshold."
        if escalation_detected
        else "The escalation rule requires at least 3 segments with a strictly increasing intensity sequence and a total rise of at least 20%."
    )

    esc_factors = []
    if escalation_detected:
        esc_factors.append("<strong>Monotonic Trend:</strong> Each consecutive turn intensity strictly exceeded the previous turn.")
        intensity_growth = float(escalation.get("intensity_change", escalation.get("intensity_increase", 0.0)) or 0.0)
        esc_factors.append(f"<strong>Net Intensity Growth:</strong> Final turn intensity exceeds baseline turn by +{intensity_growth:.2f}.")
        esc_factors.append("<strong>Acoustic & Linguistic Alignment:</strong> Both acoustic pitch/energy cues and text sentiment vectors reinforced upward emotional activation.")
    else:
        if trend_label.lower() == "increasing":
            esc_factors.append("<strong>Non-Monotonic Volatility:</strong> Overall intensity rose, but intermittent dips or pauses prevented continuous consecutive escalation.")
        else:
            esc_factors.append("<strong>Stable Pitch & Energy Contours:</strong> Acoustic features remained within baseline ranges without sustained upward trajectory.")
        esc_factors.append("<strong>Multimodal Threshold Check:</strong> Net escalation growth did not exceed the required 0.20 intensity threshold across 3+ turns.")

    esc_factors_html = "".join([f"<li style='margin-bottom: 0.25rem;'>{f}</li>" for f in esc_factors])

    st.markdown(
        html_block(f"""
<div class="modality-card" style="margin-top:0.8rem;">
<div class="card-meta-label">CONVERSATION DYNAMICS & ESCALATION EXPLANATION (LEGACY DETAIL)</div>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
<div>
<div style="font-size:0.8rem;color:var(--muted);">Overall Emotional Trend</div>
<div style="font-family:'IBM Plex Mono',monospace;font-weight:700;color:var(--text);">
{emotion_icon(trend_label)} {html.escape(trend_label)}
</div>
</div>
<div>
<div style="font-size:0.8rem;color:var(--muted);">Continuous Escalation</div>
<div style="font-family:'IBM Plex Mono',monospace;font-weight:700;color:{'var(--negative)' if escalation_detected else 'var(--positive)'};">
{html.escape(escalation_wording)}
</div>
</div>
</div>
<div style="margin-top:0.7rem;font-size:0.76rem;color:var(--muted);line-height:1.5;">
{html.escape(trend_explanation)} {html.escape(continuous_note)}
</div>
<div style="margin-top:0.6rem;padding-top:0.6rem;border-top:1px solid var(--border);font-size:0.74rem;color:var(--muted);">
<strong style="color:var(--text);">Why was escalation {'detected' if escalation_detected else 'not detected'} (legacy rule)?</strong>
<ul style="margin: 0.3rem 0 0 1.2rem; padding: 0;">
{esc_factors_html}
</ul>
</div>
</div>
"""),
        unsafe_allow_html=True,
    )