"""
Segment-level timeline breakdown and turn-by-turn expander accordion.
"""

import html
from typing import Any, Dict, List
import streamlit as st

from ui.components.incongruence_card import render_incongruence_card
from ui.components.modality_cards import render_channel_meters_html
from ui.utils import direction_label, emotion_code_label, emotion_icon, html_block, safe_float


import re
import textwrap


# ---------------------------------------------------------------------------
# Disclaimer shown ONCE at page level (item 14)
# ---------------------------------------------------------------------------
_INCONGRUENCE_PAGE_DISCLAIMER = (
    "⚠️ This system identifies computational disagreement between acoustic "
    "signals and linguistic sentiment. It does NOT detect deception, sarcasm, "
    "or psychological states. Modality disagreement is an analytical signal "
    "for further review, not a finding about intent."
)


def _render_why_this_result(fusion: Dict[str, Any]) -> None:
    """Phase 5: 'Why this result?' expander — shows the declarative
    pattern rule that fired, the evidence behind it, and the effective
    (reliability-weighted) fusion weights actually used for this segment.

    Reads keys added by app/fusion.py's fuse() in Phase 5. Uses .get()
    with safe defaults throughout so this never breaks on an older
    cached result dict that predates Phase 5.

    Item 15: "(widened short-segment window)" shown only when
    was_widened AND duration < SHORT_SEG_SECONDS. "(low ASR confidence)"
    shown only when text_reliability < LOW_ASR_CONFIDENCE_DISPLAY_THRESHOLD.
    """
    import config as _config

    pattern_label = str(fusion.get("pattern_label", "")).strip()
    pattern_explanation = str(fusion.get("pattern_explanation", "")).strip()
    rule_ids = fusion.get("rule_ids_fired", []) or []

    audio_arousal = fusion.get("audio_arousal")
    text_arousal = fusion.get("text_arousal", "unknown")

    audio_reliability = safe_float(fusion.get("audio_reliability", 1.0))
    text_reliability = safe_float(fusion.get("text_reliability", 1.0))

    # Item 15: widening badge only when actually widened AND short
    was_widened = bool(fusion.get("acoustic_window_widened", audio_reliability < 1.0))
    short_seg_threshold = getattr(_config, "SHORT_SEG_SECONDS", 1.0)
    segment_duration = safe_float(fusion.get("_segment_duration", 99.0))  # 99 = unknown → hide badge
    audio_rel_note = (
        " (widened short-segment window)"
        if was_widened and segment_duration < short_seg_threshold
        else ""
    )

    # Item 15: low ASR confidence badge uses config threshold
    low_asr_threshold = getattr(_config, "LOW_ASR_CONFIDENCE_DISPLAY_THRESHOLD", 0.65)
    text_rel_note = (
        " (low ASR confidence)"
        if text_reliability < low_asr_threshold
        else ""
    )

    base_weights = fusion.get("base_weights", {}) or {}
    effective_weights = fusion.get("effective_weights", {}) or {}
    base_audio_w = safe_float(base_weights.get("audio_weight", 0.0))
    base_text_w = safe_float(base_weights.get("text_weight", 0.0))
    eff_audio_w = safe_float(effective_weights.get("audio_weight", base_audio_w))
    eff_text_w = safe_float(effective_weights.get("text_weight", base_text_w))

    weights_shifted = (
        abs(eff_audio_w - base_audio_w) > 0.001
        or abs(eff_text_w - base_text_w) > 0.001
    )

    if not pattern_label:
        return  # nothing to show for an older/incomplete fusion dict

    rule_ids_block = ""
    if rule_ids:
        rule_ids_str = ", ".join(html.escape(str(r)) for r in rule_ids)
        rule_ids_block = (
            f'<div style="font-size: 0.68rem; color: var(--muted); margin-bottom: 0.6rem;">'
            f'Rule ID(s) fired: <code>{rule_ids_str}</code></div>'
        )

    arousal_display = f"{safe_float(audio_arousal):.2f}" if isinstance(audio_arousal, (int, float)) else "n/a"
    weight_note = (
        " (adjusted from base due to segment reliability)"
        if weights_shifted
        else " (equal to base weights — both modalities fully reliable)"
    )

    cfs = fusion.get("counterfactuals", [])
    cfs_block = ""
    if cfs:
        cfs_items = "".join(f"<li style='margin-bottom:0.25rem;'>{html.escape(c)}</li>" for c in cfs)
        cfs_block = (
            '<div style="margin-top:0.6rem; border-top:1px dashed var(--border); padding-top:0.4rem;">'
            '<div style="font-family: \'IBM Plex Mono\', monospace; font-size:0.75rem; font-weight:700;'
            ' color:var(--text-accent); text-transform:uppercase; margin-bottom:0.25rem;">'
            'Counterfactual Sensitivity:</div>'
            f'<ul style="margin:0; padding-left:1.1rem; font-size:0.75rem; color:var(--text); line-height:1.4;">{cfs_items}</ul>'
            '</div>'
        )

    content = (
        f'<div style="font-family: \'IBM Plex Mono\', monospace; font-size: 0.78rem; font-weight: 700;'
        f' color: var(--fused); text-transform: uppercase; margin-bottom: 0.3rem;">'
        f'Pattern: {html.escape(pattern_label)}</div>'
        f'<div style="font-size: 0.78rem; color: var(--text); line-height: 1.5; margin-bottom: 0.6rem;">'
        f'{html.escape(pattern_explanation)}</div>'
        f'{rule_ids_block}'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:0.6rem;font-size:0.75rem;color:var(--muted);margin-bottom:0.5rem;">'
        f'<div>Voice arousal: <strong style="color:var(--text);">{arousal_display}</strong></div>'
        f'<div>Text arousal: <strong style="color:var(--text);">{html.escape(str(text_arousal))}</strong> (not modeled)</div>'
        f'</div>'
        f'<div style="font-size:0.75rem;color:var(--muted);">'
        f'<div style="margin-bottom:0.2rem;">Base weights: audio {base_audio_w:.2f} / text {base_text_w:.2f}</div>'
        f'<div>Effective weights used: <strong style="color:var(--text);">audio {eff_audio_w:.2f} / text {eff_text_w:.2f}</strong>{weight_note}</div>'
        f'<div style="margin-top:0.3rem;">'
        f'Reliability — voice: {audio_reliability:.2f}{audio_rel_note}, text: {text_reliability:.2f}{text_rel_note}'
        f'</div></div>'
        f'{cfs_block}'
    )

    with st.expander("🔎 Why this result?", expanded=False):
        st.markdown(content, unsafe_allow_html=True)


def render_segment_timeline(segments: List[Dict[str, Any]]) -> None:
    """Render the per-turn expander list with fine-grained multimodal details.

    Item 14: the long modality-incongruence disclaimer is shown ONCE at
    the top of the timeline (not repeated per turn). The per-turn
    incongruence card omits it.
    """
    st.markdown(
        html_block("""
<div class="section-header-bar">
<div class="section-title">🔍 Segment-Level Multimodal Breakdown</div>
<div class="section-badge">TURN-BY-TURN INFERENCE</div>
</div>
"""),
        unsafe_allow_html=True,
    )

    if not segments:
        st.info("No conversational segments extracted from this recording.")
        return

    # Item 14: single disclaimer at page level
    st.caption(_INCONGRUENCE_PAGE_DISCLAIMER)

    for segment in segments:
        idx = segment.get("segment_index", "?")
        start = safe_float(segment.get("start_seconds", 0.0))
        end = safe_float(segment.get("end_seconds", 0.0))
        # Item 13: store raw transcript text (no speaker prefix)
        raw_text = str(segment.get("transcript", "")).strip()
        voice = segment.get("voice_emotion", {}) or {}
        sentiment = segment.get("text_sentiment", {}) or {}
        fusion = segment.get("fusion", {}) or {}
        intensity = safe_float(segment.get("intensity", 0.0))
        speaker = str(segment.get("speaker", "")).strip()

        v_emotion_raw = str(voice.get("emotion", "unknown")).lower()
        v_emotion_display = emotion_code_label(v_emotion_raw)  # item 9
        v_conf = safe_float(voice.get("confidence", 0.0))
        t_sentiment_raw = str(sentiment.get("sentiment", "unknown")).lower()
        t_sentiment_display = emotion_code_label(t_sentiment_raw)  # item 9
        t_conf = safe_float(sentiment.get("confidence", 0.0))
        f_conf = safe_float(fusion.get("combined_confidence", 0.0))

        agreement = bool(fusion.get("agreement", False))
        mismatch = bool(fusion.get("mismatch", False))
        interpretation = str(fusion.get("interpretation", "")).strip()
        audio_dir = direction_label(fusion.get("audio_direction", "unknown"))
        text_dir = direction_label(fusion.get("text_direction", "unknown"))

        # Item 8 (also applied here): segment_index is 0-based
        title_prefix = f"Turn {int(idx)+1:02d}" if isinstance(idx, int) else f"Turn {idx}"
        speaker_tag = f" · {speaker}" if speaker and speaker != "Unknown" else ""

        expander_title = (
            f"{title_prefix}{speaker_tag} · [{start:.2f}s → {end:.2f}s] · "
            f"Voice: {v_emotion_display.upper()} | Text: {t_sentiment_display.upper()} | Intensity: {intensity:.0%}"
        )

        # Per-speaker color assignment
        speaker_colors = {
            "Speaker A": "#38bdf8",
            "Speaker B": "#f472b6",
            "Speaker C": "#a78bfa",
            "Speaker D": "#fbbf24",
        }
        spk_color = speaker_colors.get(speaker, "#94a3b8")
        pat_label = html.escape(str(fusion.get("pattern_label", "Multimodal Arbitration")))

        with st.expander(expander_title, expanded=True):
            # Chat-style turn header with emotion chips & speaker tag
            # Tier 4 (Item 32): Low voice reliability chip
            import config as _config
            low_v_conf_thresh = getattr(_config, "LOW_VOICE_CONFIDENCE_THRESHOLD", 0.50)
            low_v_rel_disp_thresh = getattr(_config, "LOW_VOICE_RELIABILITY_DISPLAY_THRESHOLD", 0.65)
            v_rel = safe_float(segment.get("voice_reliability", voice.get("voice_reliability", fusion.get("audio_reliability", 1.0))))
            is_low_voice_rel = bool(
                segment.get("low_voice_reliability", False)
                or voice.get("low_voice_reliability", False)
                or segment.get("voice_collapse_detected", False)
                or voice.get("voice_collapse_detected", False)
                or voice.get("is_low_voice_confidence", False)
                or v_rel < low_v_rel_disp_thresh
                or v_conf < low_v_conf_thresh
            )
            low_voice_chip_html = (
                f'<span style="font-family:\'IBM Plex Mono\',monospace; font-size:0.75rem; '
                f'background:rgba(239,68,68,0.15); color:var(--warning); padding:2px 8px; border-radius:4px; border:1px solid rgba(239,68,68,0.3);">'
                f'⚠️ LOW VOICE RELIABILITY</span>'
                if is_low_voice_rel else ""
            )

            st.markdown(
                html_block(f"""
<div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; margin-bottom:0.5rem;">
<div style="display:flex; align-items:center; gap:0.5rem;">
<span style="display:inline-block; width:10px; height:10px; border-radius:50%; background:{spk_color};"></span>
<span style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem; font-weight:700; color:{spk_color};">
{html.escape(speaker if speaker and speaker != 'Unknown' else 'Speaker')}
</span>
<span style="font-size:0.75rem; color:var(--muted);">[{start:.2f}s → {end:.2f}s]</span>
</div>
<div style="display:flex; gap:6px; flex-wrap:wrap;">
<span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; background:rgba(251,191,36,0.15); color:var(--voice); padding:2px 8px; border-radius:4px;">🎙️ {v_emotion_display.upper()}</span>
<span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; background:rgba(56,189,248,0.15); color:var(--text-accent); padding:2px 8px; border-radius:4px;">💬 {t_sentiment_display.upper()}</span>
<span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; background:rgba(226,232,240,0.12); color:var(--fused); padding:2px 8px; border-radius:4px;">⚡ {pat_label}</span>
{low_voice_chip_html}
</div>
</div>
"""),
                unsafe_allow_html=True,
            )

            # Intensity mini-bar
            st.markdown(
                html_block(f"""
<div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.4rem;">
<span style="font-family:'IBM Plex Mono',monospace; font-size:0.75rem; color:var(--muted); min-width:90px;">Intensity: {intensity:.0%}</span>
<div style="flex-grow:1; background:rgba(255,255,255,0.08); height:6px; border-radius:3px; overflow:hidden;">
<div style="background:linear-gradient(90deg, #38bdf8, #f87171); width:{max(0.0, min(1.0, intensity))*100:.1f}%; height:100%; border-radius:3px;"></div>
</div>
</div>
"""),
                unsafe_allow_html=True,
            )

            # Item 22: Expose intensity v2 subcomponents per segment
            int_comp = segment.get("intensity_components", {}) or {}
            if int_comp.get("formula_version") == "v2":
                p_arousal = safe_float(int_comp.get("prosodic_arousal", 0.0))
                v_arousal = safe_float(int_comp.get("voice_arousal", 0.0))
                t_neg = safe_float(int_comp.get("text_negativity", 0.0))
                weights = int_comp.get("weights", {}) or {}
                w_p = safe_float(weights.get("prosody", 0.35))
                w_v = safe_float(weights.get("voice_arousal", 0.35))
                w_t = safe_float(weights.get("text_negativity", 0.30))
                st.markdown(
                    html_block(f"""
<div style="display:flex; gap:0.6rem; flex-wrap:wrap; font-family:'IBM Plex Mono',monospace; font-size:0.70rem; color:var(--muted); margin-bottom:0.6rem; padding:2px 6px; background:rgba(255,255,255,0.02); border-radius:4px;">
<span><strong>Intensity v2:</strong></span>
<span title="Loudness, pitch variation, voiced ratio, speech rate">Prosody (w={w_p:.0%}): <strong style="color:var(--text);">{p_arousal:.1%}</strong></span>
<span>·</span>
<span title="Expected-value voice-emotion arousal from class probabilities">Voice Arousal (w={w_v:.0%}): <strong style="color:var(--voice);">{v_arousal:.1%}</strong></span>
<span>·</span>
<span title="RoBERTa negative sentiment class probability">Text Negativity (w={w_t:.0%}): <strong style="color:var(--critical);">{t_neg:.1%}</strong></span>
</div>
"""),
                    unsafe_allow_html=True,
                )

            # Item 13: strip speaker prefix from raw_text before display
            # Store clean text (no "Speaker A:" prefix) for display
            clean_text = raw_text
            if speaker and speaker != "Unknown":
                clean_text = re.sub(rf"^{re.escape(speaker)}\s*:\s*", "", clean_text, flags=re.IGNORECASE)

            if clean_text:
                # Show speaker badge separately; never embed prefix in the text
                speaker_label_html = (
                    f'<span style="font-family: \'IBM Plex Mono\', monospace; font-size: 0.75rem; '
                    f'color: var(--text-accent); font-weight: 700; text-transform: uppercase;">Transcript:</span> '
                )
                st.markdown(
                    html_block(f"""
<div class="segment-transcript-box" style="border-left: 3px solid {spk_color};">
{speaker_label_html}"{html.escape(clean_text)}"
</div>
"""),
                    unsafe_allow_html=True,
                )

            # Item 15: use config threshold for low ASR confidence badge
            low_asr_threshold = getattr(_config, "LOW_ASR_CONFIDENCE_DISPLAY_THRESHOLD", 0.65)
            seg_text_rel = safe_float(segment.get("text_reliability", fusion.get("text_reliability", 1.0)))
            seg_low_conf = bool(segment.get("is_low_confidence", False) or seg_text_rel < low_asr_threshold)
            if seg_low_conf:
                st.markdown(
                    html_block(f"""
<div style="background: rgba(234, 179, 8, 0.12); border: 1px solid rgba(234, 179, 8, 0.4); border-radius: 6px; padding: 0.4rem 0.65rem; margin-bottom: 0.6rem; font-size: 0.72rem; color: var(--text);">
⚠️ <strong>Low ASR Confidence (Text Reliability {seg_text_rel:.0%}):</strong> Sentiment ran on an uncertain transcript.
</div>
"""),
                    unsafe_allow_html=True,
                )

            # Tier 4 (Item 32): Low voice reliability warning callout
            if is_low_voice_rel:
                collapse_reason = "Acoustic domain shift (class collapse detected)" if segment.get("voice_collapse_detected") or voice.get("voice_collapse_detected") else f"Voice confidence ({v_conf:.0%}) < {low_v_conf_thresh:.0%}"
                st.markdown(
                    html_block(f"""
<div style="background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.35); border-radius: 6px; padding: 0.4rem 0.65rem; margin-bottom: 0.6rem; font-size: 0.72rem; color: var(--text);">
⚠️ <strong>Low Voice Reliability ({v_rel:.0%}):</strong> {collapse_reason}; voice weight reduced in multimodal fusion.
</div>
"""),
                    unsafe_allow_html=True,
                )


            card_col1, card_col2, card_col3 = st.columns(3)

            with card_col1:
                st.markdown(
                    html_block(f"""
<div class="segment-card voice">
<div class="card-meta-label" style="color: var(--voice);">🎙️ VOICE EMOTION</div>
<div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 0.35rem;">
{emotion_icon(v_emotion_raw)} {html.escape(v_emotion_display.upper())}
</div>
<div style="font-size: 0.72rem; color: var(--muted);">
Confidence: <strong style="color: var(--voice);">{v_conf:.1%}</strong>
</div>
</div>
"""),
                    unsafe_allow_html=True,
                )

            with card_col2:
                st.markdown(
                    html_block(f"""
<div class="segment-card text">
<div class="card-meta-label" style="color: var(--text-accent);">💬 TEXT SENTIMENT</div>
<div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 0.35rem;">
{emotion_icon(t_sentiment_raw)} {html.escape(t_sentiment_display.upper())}
</div>
<div style="font-size: 0.72rem; color: var(--muted);">
Confidence: <strong style="color: var(--text-accent);">{t_conf:.1%}</strong>
</div>
</div>
"""),
                    unsafe_allow_html=True,
                )

            with card_col3:
                st.markdown(
                    html_block(f"""
<div class="segment-card fusion">
<div class="card-meta-label" style="color: var(--fused);">⚡ MULTIMODAL FUSION</div>
<div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 0.35rem;">
INTENSITY: {intensity:.1%}
</div>
<div style="font-size: 0.72rem; color: var(--muted);">
Combined Conf: <strong style="color: var(--fused);">{f_conf:.1%}</strong>
</div>
</div>
"""),
                    unsafe_allow_html=True,
                )

            st.markdown(
                render_channel_meters_html(
                    voice_label=v_emotion_raw,
                    voice_confidence=v_conf,
                    text_label=t_sentiment_raw,
                    text_confidence=t_conf,
                    fused_confidence=f_conf,
                ),
                unsafe_allow_html=True,
            )

            if agreement:
                badge_class = "ok"
                badge_text = "✓ Voice & Text Modalities Agree"
            elif mismatch:
                badge_class = "warn"
                badge_text = "⚠ Voice / Text Modality Mismatch Detected"
            else:
                badge_class = "info"
                badge_text = "ℹ Low-Confidence Disagreement"

            # Item 14: only show "different directions" line on real direction mismatch
            dirs_differ = (
                audio_dir.lower() != text_dir.lower()
                and audio_dir.lower() not in ("", "unknown")
                and text_dir.lower() not in ("", "unknown")
            )
            direction_line = (
                f'<div style="font-size: 0.72rem; color: var(--muted); margin-top: 0.25rem;">'
                f'Voice and text point in different directions: Voice → {html.escape(audio_dir)}, Text → {html.escape(text_dir)}'
                f'</div>'
                if dirs_differ and not agreement
                else ""
            )

            st.markdown(
                html_block(f"""
<div style="background: var(--panel-2); border: 1px solid var(--border); border-radius: 6px; padding: 0.65rem 0.85rem; margin-top: 0.4rem;">
<div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.35rem;">
<span class="status-badge {badge_class}">{badge_text}</span>
<span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--muted);">
Voice Dir: <strong style="color: var(--text);">{html.escape(audio_dir)}</strong> &nbsp;|&nbsp;
Text Dir: <strong style="color: var(--text);">{html.escape(text_dir)}</strong>
</span>
</div>
{direction_line}
<div style="font-size: 0.75rem; color: var(--muted); line-height: 1.4;">
<strong>Interpretation:</strong> {html.escape(interpretation) if interpretation else 'Standard multimodal arbitration.'}
</div>
</div>
"""),
                unsafe_allow_html=True,
            )

            # Phase 5: "Why this result?" — declarative rule trace and
            # effective (reliability-weighted) fusion weights for this
            # specific segment.
            _render_why_this_result(fusion)

            # Item 14: incongruence card called WITHOUT repeating disclaimer
            render_incongruence_card(
                acoustic_label=v_emotion_raw,
                acoustic_confidence=v_conf,
                text_label=t_sentiment_raw,
                text_confidence=t_conf,
                title=f"{title_prefix} MODALITY INCONGRUENCE",
                show_disclaimer=False,  # already shown at page level
            )