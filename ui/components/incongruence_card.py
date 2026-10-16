"""
Modality incongruence conflict card rendering with safety disclaimers.
Extracted from v1.0-baseline:streamlit_app.py with exact logic and styling parity.

Item 14: The long heuristic_notice disclaimer is shown at per-turn card level
by default (show_disclaimer=True). Pass show_disclaimer=False to suppress it
when the calling context has already shown it once at page level.
"""

import html
from typing import Any, Dict, Optional
import streamlit as st

from app.incongruence_analysis import detect_modality_incongruence
from ui.utils import html_block


def render_incongruence_card(
    acoustic_label: str,
    acoustic_confidence: float,
    text_label: str,
    text_confidence: float,
    title: str = "MODALITY INCONGRUENCE",
    show_disclaimer: bool = True,
    fusion: Optional[Dict[str, Any]] = None,
) -> None:
    """Render a transparent, safe, explainable Modality Incongruence Card.

    Parameters
    ----------
    show_disclaimer:
        When False the long heuristic_notice footer is suppressed. Use this
        when the disclaimer has already been rendered at the page level so it
        does not repeat for every turn (item 14).
    fusion:
        Optional segment fusion result dict. When rule_ids_fired contains any
        'mismatch_*', conflict strength is derived from fusion so it never
        evaluates to 'NONE'.
    """
    inc = detect_modality_incongruence(
        acoustic_label=acoustic_label,
        acoustic_confidence=acoustic_confidence,
        text_label=text_label,
        text_confidence=text_confidence,
    )

    fusion = fusion or {}
    rule_ids = fusion.get("rule_ids_fired", []) or []
    has_mismatch_rule = any(str(r).startswith("mismatch_") for r in rule_ids) or bool(fusion.get("mismatch", False))

    raw_strength = inc.conflict_strength.upper()
    if has_mismatch_rule and raw_strength == "NONE":
        # Derive conflict strength from confidence or default to MEDIUM if high confidence
        if acoustic_confidence >= 0.60 and text_confidence >= 0.60:
            raw_strength = "HIGH"
        elif acoustic_confidence >= 0.40 and text_confidence >= 0.40:
            raw_strength = "MEDIUM"
        else:
            raw_strength = "LOW"
    explanation = inc.explanation
    if has_mismatch_rule and inc.conflict_strength.upper() == "NONE":
        explanation = str(fusion.get("interpretation") or "").strip() or f"Modality mismatch detected between acoustic tone ('{acoustic_label}') and text sentiment ('{text_label}')."

    strength = raw_strength
    if strength == "HIGH":
        badge_bg = "rgba(228, 107, 107, 0.18)"
        badge_color = "var(--negative)"
        border_color = "var(--negative)"
    elif strength == "MEDIUM":
        badge_bg = "rgba(232, 163, 61, 0.18)"
        badge_color = "var(--voice)"
        border_color = "var(--voice)"
    elif strength == "LOW":
        badge_bg = "rgba(79, 179, 169, 0.18)"
        badge_color = "var(--text-accent)"
        border_color = "var(--text-accent)"
    else:
        badge_bg = "rgba(154, 163, 178, 0.12)"
        badge_color = "var(--muted)"
        border_color = "var(--border)"

    disclaimer_html = (
        f'<div style="font-size:0.70rem; color:var(--muted-dark); line-height:1.35;'
        f' font-style:italic; border-top:1px dashed var(--border); padding-top:0.4rem;">'
        f'⚠️ {html.escape(inc.heuristic_notice)}'
        f'</div>'
        if show_disclaimer
        else ""
    )

    st.markdown(
        html_block(f"""
<div class="modality-card" style="border-left: 3px solid {border_color}; margin-top: 0.6rem; margin-bottom: 0.6rem;">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 0.5rem; flex-wrap:wrap; gap:0.4rem;">
<div class="card-meta-label" style="letter-spacing: 0.1em; color: var(--text);">
⚡ {html.escape(title)}
</div>
<span style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; font-weight:700; background:{badge_bg}; color:{badge_color}; padding:2px 8px; border-radius:4px;">
CONFLICT: {strength}
</span>
</div>
<div style="display:grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-bottom: 0.6rem;">
<div style="background:var(--panel-2); padding:0.5rem 0.75rem; border-radius:6px; border:1px solid var(--border);">
<div style="font-size:0.68rem; color:var(--muted); text-transform:uppercase; font-family:'IBM Plex Mono',monospace;">Acoustic Signal (Voice)</div>
<div style="font-size:0.95rem; font-weight:700; color:var(--voice); font-family:'IBM Plex Mono',monospace;">
{html.escape(inc.acoustic_label.upper())} — {inc.acoustic_confidence:.0%}
</div>
</div>
<div style="background:var(--panel-2); padding:0.5rem 0.75rem; border-radius:6px; border:1px solid var(--border);">
<div style="font-size:0.68rem; color:var(--muted); text-transform:uppercase; font-family:'IBM Plex Mono',monospace;">Text Signal (Sentiment)</div>
<div style="font-size:0.95rem; font-weight:700; color:var(--text-accent); font-family:'IBM Plex Mono',monospace;">
{html.escape(inc.text_label.upper())} — {inc.text_confidence:.0%}
</div>
</div>
</div>
<div style="font-size:0.78rem; color:var(--text); line-height:1.45; margin-bottom: 0.4rem;">
<strong style="color:var(--muted);">Possible interpretation:</strong> {html.escape(explanation)}
</div>
{disclaimer_html}
</div>
"""),
        unsafe_allow_html=True,
    )
