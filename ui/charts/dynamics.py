"""
Interactive Plotly charts for conversation dynamics, valence-arousal mapping,
agreement ribbons, and speaker comparisons.
"""

import html
from typing import Any, Dict, List, Optional
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ui.utils import compute_ols_slope, html_block, safe_float
from ui.charts.theme import apply_chart_theme, get_chart_theme_tokens


def render_agreement_strip(segments: List[Dict[str, Any]]) -> None:
    """Render a horizontal timeline ribbon showing turn-by-turn modality agreement."""
    if not segments:
        return

    st.markdown(
        html_block("""
<div class="section-header-bar" style="margin-top: 1.0rem;">
<div class="section-title">🚥 Modality Agreement Ribbon</div>
<div class="section-badge">TURN-BY-TURN MODALITY ALIGNMENT</div>
</div>
"""),
        unsafe_allow_html=True,
    )

    blocks_html = []
    for s in segments:
        # Item 8: segment_index is 0-based → display as Turn 01, Turn 02, ...
        raw_idx = s.get("segment_index", 0)
        display_idx = int(raw_idx) + 1  # 1-based
        fusion = s.get("fusion", {}) or {}
        is_agree = bool(fusion.get("agreement", False))
        is_mismatch = bool(fusion.get("mismatch", False))
        is_abstain = bool(fusion.get("abstention", False))

        v_dir = str(fusion.get("audio_direction", "unknown")).upper()
        t_dir = str(fusion.get("text_direction", "unknown")).upper()

        if is_abstain:
            border = "var(--muted)"
            bg = "rgba(148, 163, 184, 0.12)"
            status = "ABSTAIN"
            status_color = "var(--muted)"
        elif is_mismatch:
            border = "var(--negative)"
            bg = "rgba(248, 113, 113, 0.16)"
            status = "MISMATCH"
            status_color = "var(--negative)"
        elif is_agree:
            border = "var(--positive)"
            bg = "rgba(74, 222, 128, 0.14)"
            status = "AGREE"
            status_color = "var(--positive)"
        else:
            border = "var(--warning)"
            bg = "rgba(251, 191, 36, 0.14)"
            status = "WEAK"
            status_color = "var(--warning)"

        # Item 7: each block built without indented leading whitespace
        block = (
            f'<div style="flex: 1 1 80px; min-width: 80px; max-width: 140px;'
            f' background: {bg}; border: 1px solid {border}; border-radius: 6px;'
            f' padding: 6px 8px; text-align: center;">'
            f'<div style="font-family: \'IBM Plex Mono\', monospace; font-size: 0.75rem;'
            f' color: var(--muted); font-weight: 600;">Turn {display_idx:02d}</div>'
            f'<div style="font-family: \'IBM Plex Mono\', monospace; font-size: 0.75rem;'
            f' font-weight: 700; color: {status_color}; margin: 2px 0;">{status}</div>'
            f'<div style="font-size: 0.75rem; color: var(--text); overflow: hidden;'
            f' text-overflow: ellipsis; white-space: nowrap;">{v_dir[:3]}/{t_dir[:3]}</div>'
            f'</div>'
        )
        blocks_html.append(block)

    st.markdown(
        html_block(f"""
<div style="display: flex; gap: 8px; overflow-x: auto; padding: 6px 0; margin-bottom: 1.0rem;">
{''.join(blocks_html)}
</div>
"""),
        unsafe_allow_html=True,
    )




def render_valence_arousal_map(segments: List[Dict[str, Any]]) -> None:
    """Render 2D Russell-style Valence-Arousal emotional trajectory map."""
    if len(segments) < 1:
        return

    st.markdown(
        """
        <div class="section-header-bar" style="margin-top: 1.0rem;">
            <div class="section-title">🧭 Valence-Arousal Trajectory Map</div>
            <div class="section-badge">2D AFFECTIVE TRAJECTORY</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    valence_scores = []
    arousal_scores = []
    hover_texts = []
    turn_labels = []

    sentiment_valence_map = {
        "positive": 0.75,
        "neutral": 0.0,
        "negative": -0.75,
    }

    for idx, s in enumerate(segments):
        sentiment = s.get("text_sentiment", {}) or {}
        sent_label = str(sentiment.get("sentiment", "neutral")).lower()
        sent_conf = safe_float(sentiment.get("confidence", 0.7))
        # Continuous valence scaled by confidence
        base_v = sentiment_valence_map.get(sent_label, 0.0)
        v = base_v * (0.5 + 0.5 * sent_conf)
        valence_scores.append(round(v, 3))

        fusion = s.get("fusion", {}) or {}
        arousal = fusion.get("audio_arousal")
        if arousal is None or not isinstance(arousal, (int, float)):
            intensity = safe_float(s.get("intensity", 0.5))
            arousal = intensity
        arousal_scores.append(round(safe_float(arousal), 3))

        turn_labels.append(f"T{idx+1}")
        hover_texts.append(
            f"<b>Turn {idx+1}</b><br>"
            f"Text Sentiment: {sent_label.title()} (conf: {sent_conf:.0%})<br>"
            f"Arousal: {arousal:.2f}<br>"
            f"Transcript: {html.escape(str(s.get('transcript', ''))[:45])}..."
        )

    fig = go.Figure()

    # Connecting trajectory path with arrows
    fig.add_trace(
        go.Scatter(
            x=valence_scores,
            y=arousal_scores,
            mode="lines+markers+text",
            line=dict(color="#38bdf8", width=2, dash="dot"),
            marker=dict(
                size=12,
                color=list(range(len(segments))),
                colorscale="Viridis",
                showscale=False,
                line=dict(color="#ffffff", width=1),
            ),
            text=turn_labels,
            textposition="top center",
            hovertext=hover_texts,
            hoverinfo="text",
            name="Emotional Path",
        )
    )

    theme_tokens = get_chart_theme_tokens()
    # Reference quadrant lines (X=0, Y=0.5)
    fig.add_hline(y=0.5, line_width=1, line_dash="dash", line_color=theme_tokens["zerolinecolor"])
    fig.add_vline(x=0.0, line_width=1, line_dash="dash", line_color=theme_tokens["zerolinecolor"])

    apply_chart_theme(fig, height=360, margin=dict(l=40, r=40, t=30, b=40), show_legend=False)
    fig.update_xaxes(
        title="Valence (Negative ← Neutral → Positive)",
        range=[-1.05, 1.05],
    )
    fig.update_yaxes(
        title="Arousal / Activation (Calm → Excited)",
        range=[0.0, 1.05],
    )

    st.plotly_chart(fig, use_container_width=True)


def render_intensity_trajectory_chart(
    segments: List[Dict[str, Any]],
    verdict: Optional[Dict[str, Any]] = None,
    escalation: Optional[Dict[str, Any]] = None,
) -> None:
    """Render fitted OLS slope, tolerance band, and escalation segments.

    Item 11: When fewer than 3 turns are present, the OLS slope and
    tolerance band are hidden, no 'stable' claim is made, and the legend
    shows 'Too few turns to assess a trend'. The slope shown in the legend
    comes from verdict["slope"] (the same value used by the verdict panel)
    so all three displays (verdict, legend, exports) are always consistent.
    """
    if len(segments) < 2:
        return

    st.markdown(
        html_block("""
<div class="section-header-bar" style="margin-top: 1.0rem;">
<div class="section-title">📈 Intensity Trajectory with Fitted Slope &amp; Tolerance Band</div>
<div class="section-badge">OLS REGRESSION &amp; TOLERANCE DYNAMICS</div>
</div>
"""),
        unsafe_allow_html=True,
    )

    intensities = [safe_float(s.get("intensity", 0.0)) for s in segments]
    n = len(intensities)
    x = np.arange(1, n + 1)
    y = np.array(intensities)

    import config
    # Item 11: slope is shown only for >= 3 turns (same threshold as verdict)
    min_turns_for_slope = getattr(config, "VERDICT_MIN_SEGMENTS", 3)
    enough_turns = n >= min_turns_for_slope

    # Use verdict["slope"] as SSOT (same value displayed in verdict panel)
    v_dict = verdict or {}
    if enough_turns:
        if v_dict.get("verdict") not in (None, "", "insufficient_data"):
            slope = safe_float(v_dict.get("slope", 0.0))
        else:
            slope, _intercept, _net = compute_ols_slope(intensities)
        intercept_arr = np.polyfit(x, y, 1)
        intercept = intercept_arr[1]
        fitted_y = slope * x + intercept
        dip_tol = getattr(config, "ESCALATION_DIP_TOLERANCE", 0.08)
        upper_band = np.clip(fitted_y + dip_tol, 0.0, 1.0)
        lower_band = np.clip(fitted_y - dip_tol, 0.0, 1.0)
    else:
        slope = 0.0

    fig = go.Figure()

    if enough_turns:
        # Tolerance band
        fig.add_trace(
            go.Scatter(
                x=list(x) + list(x)[::-1],
                y=list(upper_band) + list(lower_band)[::-1],
                fill="toself",
                fillcolor="rgba(56, 189, 248, 0.10)",
                line=dict(color="rgba(255, 255, 255, 0)"),
                hoverinfo="skip",
                showlegend=True,
                name=f"Tolerance Band (±{dip_tol:.2f})",
            )
        )

    # Actual intensity line
    turn_names = [f"Turn {i}" for i in x]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line=dict(color="#f472b6", width=2.5),
            marker=dict(size=8, color="#f472b6"),
            name="Segment Intensity",
            hovertext=[f"<b>{turn_names[i-1]}</b><br>Intensity: {y[i-1]:.1%}" for i in range(1, n + 1)],
            hoverinfo="text",
        )
    )

    if enough_turns:
        # OLS Fitted Trend Line — slope is SSOT from verdict dict
        fig.add_trace(
            go.Scatter(
                x=x,
                y=fitted_y,
                mode="lines",
                line=dict(color="#38bdf8", width=2, dash="dash"),
                name=f"OLS Slope ({slope:+.3f}/turn)",
            )
        )
    else:
        # Item 11: fewer than min_turns — no slope claim, clear notice
        fig.add_annotation(
            text=f"Too few turns to assess a trend (need ≥ {min_turns_for_slope})",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=11, color="#8E97A6"),
        )

    apply_chart_theme(fig, height=340, margin=dict(l=40, r=20, t=30, b=40), show_legend=True)
    fig.update_xaxes(
        title="Conversation Turn",
        tickvals=list(x),
        ticktext=turn_names,
    )
    fig.update_yaxes(
        title="Emotional Intensity",
        range=[0.0, 1.05],
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10),
        ),
    )

    st.plotly_chart(fig, use_container_width=True)




def render_speaker_comparison_chart(speaker_analytics: List[Dict[str, Any]]) -> None:
    """Render comparative analytics across speakers when 2+ speakers are present."""
    if not speaker_analytics or len(speaker_analytics) < 2:
        return

    st.markdown(
        """
        <div class="section-header-bar" style="margin-top: 1.0rem;">
            <div class="section-title">👥 Speaker-by-Speaker Comparison</div>
            <div class="section-badge">INTER-SPEAKER DYNAMICS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    speakers = [str(sa.get("speaker", "Unknown")) for sa in speaker_analytics]
    talk_shares = [safe_float(sa.get("talk_time_share", 0.0)) * 100 for sa in speaker_analytics]
    mean_intensities = [safe_float(sa.get("mean_intensity", 0.0)) * 100 for sa in speaker_analytics]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Talk Time Share (%)",
            x=speakers,
            y=talk_shares,
            marker_color="#38bdf8",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Mean Intensity (%)",
            x=speakers,
            y=mean_intensities,
            marker_color="#f87171",
        )
    )

    apply_chart_theme(fig, height=280, margin=dict(l=40, r=20, t=30, b=40), show_legend=True)
    fig.update_layout(
        barmode="group",
        yaxis=dict(title="Percentage (%)", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)
