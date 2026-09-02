r"""
Streamlit dashboard for the Multimodal Voice & Conversation Emotion Analysis System.

Run from the project root:

.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.fileWatcherType none
"""

import html
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from app.conversation_analysis import ConversationAnalyzer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Multimodal Emotion Analysis System",
    page_icon="ðŸŽ›ï¸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# DESIGN SYSTEM
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Inter:wght@300;400;500;600;700&display=swap');

    :root {
        --bg: #101318;
        --panel: #181D24;
        --panel-2: #1D232C;
        --panel-hover: #222933;
        --border: rgba(255, 255, 255, 0.08);
        --border-strong: rgba(255, 255, 255, 0.15);
        --text: #F1F3F5;
        --muted: #8E97A6;
        --muted-dark: #636B78;
        --voice: #E8A33D;
        --voice-glow: rgba(232, 163, 61, 0.15);
        --text-accent: #4FB3A9;
        --text-glow: rgba(79, 179, 169, 0.15);
        --positive: #58B889;
        --positive-bg: rgba(88, 184, 137, 0.12);
        --negative: #E46B6B;
        --negative-bg: rgba(228, 107, 107, 0.12);
        --neutral: #9AA3B2;
        --neutral-bg: rgba(154, 163, 178, 0.10);
        --fused: #D7DBE2;
        --card-radius: 8px;
    }

    /* Global Base */
    .stApp {
        background:
            radial-gradient(circle at 85% 5%, rgba(232, 163, 61, 0.045), transparent 28%),
            radial-gradient(circle at 15% 15%, rgba(79, 179, 169, 0.04), transparent 25%),
            var(--bg);
        color: var(--text);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .stApp p,
    .stApp label,
    .stApp .stMarkdown,
    .stApp .stCaption,
    .stApp div[data-testid="stText"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* NOTE (bug fix): we deliberately do NOT force a font-family onto
       <span> elements globally. Streamlit renders its native icons
       (expander arrow, file-uploader "browse"/"x" controls, status
       icons) as <span> elements whose glyph depends on an icon-ligature
       font. A previous rule here (`span:not(.material-icons)`) caught
       spans that don't carry that exact class in this Streamlit
       version, which silently broke their icon font and printed the
       raw ligature text on screen ("arrow_downward", "add",
       "check_circle", etc). Our own custom badges/labels/chips below
       already set font-family via their own explicit classes, so they
       are unaffected by leaving native spans alone. */

    /* Native Streamlit chrome: header bar, toolbar, top decoration strip.
       Fixes the white bar above the title -- Streamlit's header keeps its
       default light background unless explicitly recolored. */
    [data-testid="stHeader"] {
        background: var(--bg) !important;
        box-shadow: none;
    }

    [data-testid="stToolbar"] {
        background: transparent !important;
    }

    [data-testid="stDecoration"] {
        background: linear-gradient(90deg, var(--voice), var(--text-accent)) !important;
        height: 3px;
    }

    header[data-testid="stHeader"] svg {
        fill: var(--muted) !important;
    }

    /* Layout Constraints */
    .block-container {
        max-width: 1400px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* =====================================================
       HEADER & BRAND
       ===================================================== */

    .brand-container {
        background: linear-gradient(180deg, rgba(24, 29, 36, 0.95), rgba(20, 24, 30, 0.85));
        border: 1px solid var(--border);
        border-radius: var(--card-radius);
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }

    .brand-top-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.75rem;
    }

    .brand-title-group {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .brand-beacon {
        width: 10px;
        height: 10px;
        background: var(--voice);
        border-radius: 50%;
        box-shadow: 0 0 12px var(--voice);
        animation: pulse-beacon 2.5s infinite ease-in-out;
    }

    @keyframes pulse-beacon {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.45; transform: scale(0.85); }
    }

    .main-title {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        color: var(--text);
        margin: 0;
        line-height: 1.2;
    }

    .system-badge {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--muted);
        background: var(--panel-2);
        padding: 0.25rem 0.6rem;
        border-radius: 4px;
        border: 1px solid var(--border);
    }

    .subtitle {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.82rem;
        color: var(--muted);
        letter-spacing: 0.02em;
        margin: 0.4rem 0 0.65rem 1.4rem;
    }

    .header-chips-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.75rem;
        padding-top: 0.75rem;
        border-top: 1px solid var(--border);
    }

    .header-chip {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.68rem;
        letter-spacing: 0.03em;
        color: var(--muted);
        background: rgba(255, 255, 255, 0.03);
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }

    .header-chip strong {
        color: var(--text);
    }

    /* =====================================================
       SECTION HEADERS
       ===================================================== */

    .section-header-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.45rem;
        margin-top: 1.6rem;
        margin-bottom: 1rem;
    }

    .section-title {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--muted);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .section-badge {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.62rem;
        color: var(--muted-dark);
        letter-spacing: 0.06em;
    }

    /* =====================================================
       PIPELINE WORKFLOW INDICATOR
       ===================================================== */

    .pipeline-container {
        display: grid;
        grid-template-columns: repeat(7, 1fr);
        gap: 0.5rem;
        margin-bottom: 1.25rem;
    }

    .pipeline-step {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0.65rem 0.5rem;
        text-align: center;
        transition: all 0.2s ease;
    }

    .pipeline-step:hover {
        border-color: var(--border-strong);
        background: var(--panel-2);
    }

    .step-number {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.6rem;
        font-weight: 700;
        color: var(--muted);
        letter-spacing: 0.05em;
        margin-bottom: 0.2rem;
    }

    .step-name {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.7rem;
        font-weight: 600;
        color: var(--text);
        line-height: 1.2;
    }

    .step-modality {
        font-size: 0.62rem;
        color: var(--muted);
        margin-top: 0.2rem;
    }

    /* =====================================================
       UPLOAD & CONTROLS
       ===================================================== */

    .upload-wrapper {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: var(--card-radius);
        padding: 1.25rem;
        margin-bottom: 1.25rem;
    }

    /* Bug fix (round 2): the previous version only targeted specific
       testids (stFileUploaderFile, stFileUploaderFileIcon, etc.), but
       this Streamlit version renders some inner wrapper elements â€”
       tags, name/size text, icon buttons â€” with their own default
       light background that those specific rules never reached, so
       a white box kept showing through after upload. Instead of
       guessing every exact testid, we blank EVERY descendant's
       background/text color first (this rule), then re-apply the
       two dark panels we actually want (the rules that follow it).
       Because both have equal CSS specificity, source order decides
       the winner, so the specific rules below correctly override
       this blanket reset. */
    [data-testid="stFileUploader"] *,
    [data-testid="stFileUploader"] *::before,
    [data-testid="stFileUploader"] *::after {
        background-color: transparent !important;
        color: var(--text) !important;
    }

    [data-testid="stFileUploader"] section {
        background-color: var(--panel-2) !important;
        border: 1px dashed var(--border-strong) !important;
        border-radius: 6px;
        padding: 1.5rem 1rem !important;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: var(--voice) !important;
        background-color: rgba(232, 163, 61, 0.03) !important;
    }

    [data-testid="stFileUploader"] small {
        color: var(--muted) !important;
    }

    [data-testid="stFileUploader"] button {
        background-color: var(--panel) !important;
        color: var(--text) !important;
        border: 1px solid var(--border-strong) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.75rem !important;
        border-radius: 4px !important;
    }

    /* The uploaded-file row (filename + size pill) â€” the dark panel
       that should visibly sit inside the dropzone once a file is
       selected. */
    [data-testid="stFileUploaderFile"] {
        background-color: var(--panel) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: 6px !important;
        padding: 0.5rem 0.7rem !important;
    }

    [data-testid="stFileUploaderFile"] small {
        color: var(--muted) !important;
    }

    [data-testid="stFileUploaderFileIcon"] {
        background-color: var(--panel-2) !important;
        border-radius: 4px;
    }

    [data-testid="stFileUploaderDeleteBtn"] svg,
    [data-testid="stFileUploaderFileIcon"] svg {
        fill: var(--muted) !important;
    }

    .format-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: 0.6rem;
    }

    .format-pill {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.64rem;
        padding: 0.18rem 0.45rem;
        border-radius: 4px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid var(--border);
        color: var(--muted);
    }

    .format-pill.recommended {
        border-color: rgba(232, 163, 61, 0.35);
        color: var(--voice);
        background: rgba(232, 163, 61, 0.08);
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        min-height: 44px;
        background: var(--voice);
        color: #101318;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        border: none;
        border-radius: 6px;
        transition: all 0.15s ease-in-out;
        box-shadow: 0 2px 10px rgba(232, 163, 61, 0.2);
    }

    .stButton > button:hover {
        background: #F3B658;
        color: #101318;
        transform: translateY(-1px);
        box-shadow: 0 4px 14px rgba(232, 163, 61, 0.35);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    .stButton > button:disabled {
        background: rgba(232, 163, 61, 0.2);
        color: rgba(241, 243, 245, 0.35);
        box-shadow: none;
        cursor: not-allowed;
    }
/* =====================================================
   DOWNLOAD BUTTONS
   Fix Streamlit default white download buttons
   ===================================================== */

[data-testid="stDownloadButton"] {
    width: 100% !important;
}

[data-testid="stDownloadButton"] button {
    width: 100% !important;
    min-height: 44px !important;
    background: var(--panel) !important;
    background-color: var(--panel) !important;
    color: var(--text) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
}

[data-testid="stDownloadButton"] button:hover {
    background: var(--panel-2) !important;
    background-color: var(--panel-2) !important;
    color: var(--text) !important;
    border-color: var(--voice) !important;
    box-shadow: 0 2px 10px rgba(232, 163, 61, 0.12) !important;
}

[data-testid="stDownloadButton"] button:focus,
[data-testid="stDownloadButton"] button:active {
    background: var(--panel-2) !important;
    background-color: var(--panel-2) !important;
    color: var(--text) !important;
    border-color: var(--voice) !important;
}

[data-testid="stDownloadButton"] button *,
[data-testid="stDownloadButton"] button p,
[data-testid="stDownloadButton"] button span {
    background: transparent !important;
    background-color: transparent !important;
    color: var(--text) !important;
}

[data-testid="stDownloadButton"] button svg {
    fill: var(--muted) !important;
}
    /* =====================================================
       METRICS & OVERVIEW CARDS
       ===================================================== */

    [data-testid="stMetric"] {
        background: linear-gradient(160deg, var(--panel), #14181F);
        border: 1px solid var(--border);
        border-radius: var(--card-radius);
        padding: 0.9rem 1.1rem;
        min-height: 100px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }

    [data-testid="stMetricLabel"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.65rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.1em;
        color: var(--muted) !important;
        text-transform: uppercase;
    }

    [data-testid="stMetricValue"] {
        font-family: 'IBM Plex Mono', monospace !important;
        color: var(--text) !important;
        font-size: 1.3rem !important;
        font-weight: 700 !important;
    }

    /* Bug fix (round 2): the previous rule used :not(:has(svg)) to
       recolor only the arrow-less "off" deltas. :has() isn't
       supported in every browser â€” when it isn't, the WHOLE selector
       is treated as invalid and the browser drops the entire rule,
       silently leaving these deltas unstyled (default dark text on
       a dark panel = invisible). Rather than depend on that
       pseudo-class, the three "off" metrics are each wrapped in a
       st.container(key=...) below, and we target those keyed
       classes directly here â€” no :has() needed anywhere. */
    .st-key-metric_language [data-testid="stMetricDelta"],
    .st-key-metric_duration [data-testid="stMetricDelta"],
    .st-key-metric_segments [data-testid="stMetricDelta"] {
        color: var(--muted) !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }

    .st-key-metric_language [data-testid="stMetricDelta"] *,
    .st-key-metric_duration [data-testid="stMetricDelta"] *,
    .st-key-metric_segments [data-testid="stMetricDelta"] * {
        color: inherit !important;
    }

    /* =====================================================
       SNAPSHOT & INSIGHT CARDS
       ===================================================== */

    .modality-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: var(--card-radius);
        padding: 1rem 1.15rem;
        height: 100%;
        position: relative;
        overflow: hidden;
    }

    .modality-card.voice-accent {
        border-top: 3px solid var(--voice);
    }

    .modality-card.text-accent {
        border-top: 3px solid var(--text-accent);
    }

    .modality-card.fusion-accent {
        border-top: 3px solid var(--fused);
    }

    .modality-card.escalation-alert {
        border: 1px solid var(--negative);
        background: linear-gradient(160deg, var(--panel), rgba(228, 107, 107, 0.08));
    }

    .card-meta-label {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.4rem;
    }

    .card-highlight-value {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--text);
        display: flex;
        align-items: center;
        gap: 0.45rem;
        margin-bottom: 0.25rem;
    }

    .card-subtext {
        font-size: 0.75rem;
        color: var(--muted);
        line-height: 1.4;
    }

    /* Status Chips / Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.66rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        padding: 0.3rem 0.6rem;
        border-radius: 4px;
    }

    .status-badge.ok {
        color: var(--positive);
        background: var(--positive-bg);
        border: 1px solid rgba(88, 184, 137, 0.25);
    }

    .status-badge.warn {
        color: var(--voice);
        background: var(--voice-glow);
        border: 1px solid rgba(232, 163, 61, 0.25);
    }

    .status-badge.danger {
        color: var(--negative);
        background: var(--negative-bg);
        border: 1px solid rgba(228, 107, 107, 0.3);
    }

    .status-badge.info {
        color: var(--neutral);
        background: var(--neutral-bg);
        border: 1px solid rgba(154, 163, 178, 0.2);
    }

    /* =====================================================
       AUDIO PLAYER & TRANSCRIPT
       ===================================================== */

    .audio-player-box {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: var(--card-radius);
        padding: 0.85rem 1rem;
        margin-bottom: 1rem;
    }

    /* Real container (replaces the old broken cross-call div) that
       actually wraps the audio player. Scoped to this one container
       only via the key="audio_player_box" argument in the Python code. */
    .st-key-audio_player_box {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--card-radius) !important;
        padding: 0.85rem 1rem !important;
    }

    audio {
        width: 100%;
        border-radius: 6px;
        filter: invert(0.9) hue-rotate(180deg);
        height: 38px;
    }

    .transcript-container {
        background: var(--panel);
        border: 1px solid var(--border);
        border-left: 3px solid var(--text-accent);
        border-radius: var(--card-radius);
        padding: 1.15rem 1.35rem;
        margin-bottom: 0.85rem;
        font-size: 0.92rem;
        line-height: 1.7;
        color: var(--text);
    }

    /* =====================================================
       METERS & PROGRESS BARS
       ===================================================== */

    .meter-container {
        margin: 0.5rem 0;
    }

    .meter-row {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        margin-bottom: 0.45rem;
    }

    .meter-label-tag {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.62rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        width: 4rem;
        flex-shrink: 0;
    }

    .meter-bar-track {
        flex: 1;
        height: 7px;
        background: rgba(255, 255, 255, 0.06);
        border-radius: 4px;
        overflow: hidden;
    }

    .meter-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }

    .meter-value-tag {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.72rem;
        color: var(--text);
        width: 3.2rem;
        text-align: right;
        flex-shrink: 0;
        font-weight: 600;
    }

    /* =====================================================
       CONVERSATION SEGMENTS / EXPANDERS
       ===================================================== */

    [data-testid="stExpander"] {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--card-radius) !important;
        margin-bottom: 0.85rem !important;
        overflow: hidden;
    }

    [data-testid="stExpander"] summary {
        background: var(--panel) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.84rem !important;
        padding: 0.65rem 1rem !important;
    }

    [data-testid="stExpander"] summary:hover {
        background: var(--panel-2) !important;
    }

    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span:not([data-testid="stIconMaterial"]) {
        color: var(--text) !important;
        font-weight: 600;
    }

    /* =====================================================
       NATIVE ALERTS, STATUS, TOAST, EXCEPTION
       (st.info / st.warning / st.error / st.exception / st.status /
       st.toast â€” none of these had any dark-theme styling before,
       so they were rendering in Streamlit's default light colors.
       Same reset-then-reapply approach as the file uploader fix:
       blank every descendant first, then set one consistent dark
       panel look. The emoji/icon each of these renders keeps its
       own native color regardless, so warning/error/info are still
       visually distinguishable at a glance. )
       ===================================================== */

    [data-testid="stAlert"],
    [data-testid="stException"] {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--card-radius) !important;
    }

    [data-testid="stAlert"] *,
    [data-testid="stException"] * {
        color: var(--text) !important;
        background-color: transparent !important;
    }

    [data-testid="stException"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
    }

    [data-testid="stStatusWidget"] {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--card-radius) !important;
    }

    [data-testid="stStatusWidget"] * {
        color: var(--text) !important;
        background-color: transparent !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }

    [data-testid="stToast"] {
        background: var(--panel) !important;
        border: 1px solid var(--border-strong) !important;
        border-radius: var(--card-radius) !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
    }

    [data-testid="stToast"] * {
        color: var(--text) !important;
        background-color: transparent !important;
    }

    .segment-transcript-box {
        background: var(--panel-2);
        border-left: 2px solid var(--muted);
        padding: 0.75rem 1rem;
        border-radius: 0 4px 4px 0;
        margin-bottom: 0.85rem;
        font-size: 0.88rem;
        line-height: 1.55;
        color: var(--text);
    }

    .segment-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 0.75rem;
        margin-top: 0.6rem;
        margin-bottom: 0.6rem;
    }

    .segment-card {
        background: var(--panel-2);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 0.75rem 0.85rem;
    }

    .segment-card.voice {
        border-top: 2px solid var(--voice);
    }

    .segment-card.text {
        border-top: 2px solid var(--text-accent);
    }

    .segment-card.fusion {
        border-top: 2px solid var(--fused);
    }

    /* =====================================================
       EMPTY STATE
       ===================================================== */

    .empty-state-card {
        background: linear-gradient(180deg, var(--panel), #14181F);
        border: 1px solid var(--border);
        border-radius: var(--card-radius);
        padding: 2rem 1.5rem;
        text-align: center;
        margin: 1.5rem 0;
    }

    .empty-steps-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-top: 1.5rem;
        text-align: left;
    }

    .empty-step-box {
        background: var(--panel-2);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 1rem 1.15rem;
    }

    .empty-step-num {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.68rem;
        font-weight: 700;
        color: var(--voice);
        margin-bottom: 0.35rem;
    }

    .empty-step-heading {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--text);
        margin-bottom: 0.3rem;
    }

    .empty-step-desc {
        font-size: 0.76rem;
        color: var(--muted);
        line-height: 1.45;
    }

    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background-color: #12151B !important;
        border-right: 1px solid var(--border) !important;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }

    .sidebar-heading {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.66rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--muted);
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.4rem;
        margin-top: 1.2rem;
        margin-bottom: 0.75rem;
    }

    .sidebar-spec-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.3rem 0;
        font-size: 0.74rem;
        border-bottom: 1px dashed rgba(255, 255, 255, 0.04);
    }

    .sidebar-spec-key {
        font-family: 'IBM Plex Mono', monospace !important;
        color: var(--muted);
        font-size: 0.68rem;
    }

    .sidebar-spec-val {
        font-family: 'IBM Plex Mono', monospace !important;
        color: var(--text);
        font-size: 0.7rem;
        font-weight: 600;
    }

    /* =====================================================
       FOOTER
       ===================================================== */

    .dashboard-footer {
        text-align: center;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.65rem;
        letter-spacing: 0.06em;
        color: var(--muted);
        padding-top: 2rem;
        margin-top: 2.5rem;
        border-top: 1px solid var(--border);
    }

    /* =====================================================
       RESPONSIVENESS
       ===================================================== */

    @media (max-width: 900px) {
        .pipeline-container {
            grid-template-columns: repeat(3, 1fr);
        }

        .segment-grid {
            grid-template-columns: 1fr;
        }

        .empty-steps-grid {
            grid-template-columns: 1fr;
        }
    }

    @media (max-width: 600px) {
        .pipeline-container {
            grid-template-columns: repeat(2, 1fr);
        }

        .main-title {
            font-size: 1.2rem;
        }

        .subtitle {
            font-size: 0.74rem;
            margin-left: 0.5rem;
        }

        .block-container {
            padding-left: 0.75rem;
            padding-right: 0.75rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD ML PIPELINE ONCE (CACHED)
# =========================================================

@st.cache_resource(show_spinner=False)
def load_analyzer() -> ConversationAnalyzer:
    """Load and initialize the complete ML pipeline once per process."""
    return ConversationAnalyzer()


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert any value to float with fallback."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def emotion_icon(emotion: str) -> str:
    """Return a descriptive icon for emotion/sentiment categories."""
    key = str(emotion).lower().strip()
    icons = {
        "angry": "ðŸ˜ ",
        "ang": "ðŸ˜ ",
        "sad": "ðŸ˜Ÿ",
        "happy": "ðŸ˜Š",
        "hap": "ðŸ˜Š",
        "neutral": "ðŸ˜",
        "neu": "ðŸ˜",
        "positive": "â†—",
        "negative": "â†˜",
        "increasing": "â†—",
        "decreasing": "â†˜",
        "stable": "â†’",
    }
    return icons.get(key, "ðŸŽ™ï¸")


def direction_label(direction: str) -> str:
    """Format an emotional direction label for clean display."""
    if not direction:
        return "Unknown"
    return str(direction).replace("_", " ").title()


def derive_segment_summary(segments: List[Dict[str, Any]]) -> Tuple[str, str, int, int]:
    """
    Derive summary statistics strictly from actual segment outputs.
    Preserves real pipeline outputs without hallucinating or inventing data.
    """
    voice_labels = []
    text_labels = []
    agreement_count = 0
    mismatch_count = 0

    for segment in segments:
        voice = segment.get("voice_emotion", {}) or {}
        sentiment = segment.get("text_sentiment", {}) or {}
        fusion = segment.get("fusion", {}) or {}

        voice_label = str(voice.get("emotion", "")).strip().lower()
        text_label = str(sentiment.get("sentiment", "")).strip().lower()

        if voice_label:
            voice_labels.append(voice_label)
        if text_label:
            text_labels.append(text_label)

        if fusion.get("agreement", False):
            agreement_count += 1
        elif fusion.get("mismatch", False):
            mismatch_count += 1

    dominant_voice = (
        pd.Series(voice_labels).value_counts().idxmax()
        if voice_labels
        else "unknown"
    )

    dominant_text = (
        pd.Series(text_labels).value_counts().idxmax()
        if text_labels
        else "unknown"
    )

    return dominant_voice, dominant_text, agreement_count, mismatch_count


def render_channel_meters_html(
    voice_label: str,
    voice_confidence: float,
    text_label: str,
    text_confidence: float,
    fused_confidence: float,
) -> str:
    """Generate HTML for dual-modality and fused confidence meters."""
    voice_pct = max(0.0, min(1.0, safe_float(voice_confidence))) * 100
    text_pct = max(0.0, min(1.0, safe_float(text_confidence))) * 100
    fused_pct = max(0.0, min(1.0, safe_float(fused_confidence))) * 100

    icon = emotion_icon(voice_label)
    safe_voice = html.escape(str(voice_label).upper())
    safe_text = html.escape(str(text_label).upper())

    return f"""
    <div class="meter-container">
        <div class="meter-row">
            <div class="meter-label-tag" style="color: var(--voice);">VOICE</div>
            <div class="meter-bar-track">
                <div class="meter-bar-fill" style="width: {voice_pct:.1f}%; background: var(--voice);"></div>
            </div>
            <div class="meter-value-tag">{voice_pct:.0f}%</div>
        </div>
        <div class="meter-row">
            <div class="meter-label-tag" style="color: var(--text-accent);">TEXT</div>
            <div class="meter-bar-track">
                <div class="meter-bar-fill" style="width: {text_pct:.1f}%; background: var(--text-accent);"></div>
            </div>
            <div class="meter-value-tag">{text_pct:.0f}%</div>
        </div>
        <div class="meter-row">
            <div class="meter-label-tag" style="color: var(--fused);">FUSED</div>
            <div class="meter-bar-track">
                <div class="meter-bar-fill" style="width: {fused_pct:.1f}%; background: var(--fused);"></div>
            </div>
            <div class="meter-value-tag">{fused_pct:.0f}%</div>
        </div>
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--muted); margin-top: 0.4rem; padding-top: 0.35rem; border-top: 1px dashed var(--border);">
            {icon} {safe_voice} + {safe_text} &nbsp;â†’&nbsp; <strong style="color: var(--text);">COMBINED CONFIDENCE: {fused_pct:.1f}%</strong>
        </div>
    </div>
    """


# =========================================================
# SIDEBAR (SYSTEM ARCHITECTURE & INPUT GUIDELINES)
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;">
            <div style="width: 8px; height: 8px; border-radius: 50%; background: var(--voice); box-shadow: 0 0 8px var(--voice);"></div>
            <div style="font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 0.85rem; color: var(--text);">
                AI ANALYSIS CONSOLE
            </div>
        </div>
        <div style="font-size: 0.72rem; color: var(--muted); margin-bottom: 1.25rem;">
            B.Tech Mini-Project AI Pipeline
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-heading">Models & Architecture</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="sidebar-spec-row">
            <span class="sidebar-spec-key">Acoustic Emotion</span>
            <span class="sidebar-spec-val" style="color: var(--voice);">Wav2Vec2-SUPERB ER</span>
        </div>
        <div class="sidebar-spec-row">
            <span class="sidebar-spec-key">Speech-to-Text</span>
            <span class="sidebar-spec-val" style="color: var(--text);">Whisper STT</span>
        </div>
        <div class="sidebar-spec-row">
            <span class="sidebar-spec-key">Text Sentiment</span>
            <span class="sidebar-spec-val" style="color: var(--text-accent);">RoBERTa NLP</span>
        </div>
        <div class="sidebar-spec-row">
            <span class="sidebar-spec-key">Speaker ID</span>
            <span class="sidebar-spec-val" style="color: var(--positive);">Pyannote Diarization</span>
        </div>
        <div class="sidebar-spec-row">
            <span class="sidebar-spec-key">Fusion Strategy</span>
            <span class="sidebar-spec-val" style="color: var(--fused);">Weighted Rule-Based</span>
        </div>
        <div class="sidebar-spec-row">
            <span class="sidebar-spec-key">Dynamics Engine</span>
            <span class="sidebar-spec-val">Trend & Escalation</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-heading">Audio Input Specs</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="sidebar-spec-row">
            <span class="sidebar-spec-key">Supported</span>
            <span class="sidebar-spec-val">WAV, FLAC, MP3, OGG, M4A</span>
        </div>
        <div class="sidebar-spec-row">
            <span class="sidebar-spec-key">Target Rate</span>
            <span class="sidebar-spec-val">16,000 Hz Mono</span>
        </div>
        <div class="sidebar-spec-row">
            <span class="sidebar-spec-key">Recommended</span>
            <span class="sidebar-spec-val" style="color: var(--voice);">WAV (PCM 16-bit)</span>
        </div>
        <div class="sidebar-spec-row">
            <span class="sidebar-spec-key">Hardware Engine</span>
            <span class="sidebar-spec-val">CPU Inference</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# HEADER COMPONENT
# =========================================================

st.markdown(
    """
    <div class="brand-container">
        <div class="brand-top-row">
            <div class="brand-title-group">
                <div class="brand-beacon"></div>
                <h1 class="main-title">MULTIMODAL VOICE & CONVERSATION EMOTION ANALYSIS SYSTEM</h1>
            </div>
            <div class="system-badge">AI ANALYTICS ENGINE Â· V1.0</div>
        </div>
        <div class="subtitle">Voice + Speech + Text â†’ Unified Conversation Emotion Analysis</div>
        <div class="header-chips-row">
            <div class="header-chip">ðŸŽ™ï¸ <strong>Voice Emotion:</strong> Wav2Vec2-SUPERB ER</div>
            <div class="header-chip">ðŸ“ <strong>Speech Recognition:</strong> Whisper STT</div>
            <div class="header-chip">ðŸ’¬ <strong>Text Sentiment:</strong> RoBERTa</div>
            <div class="header-chip">ðŸ—£ï¸ <strong>Speaker ID:</strong> Pyannote Diarization</div>
            <div class="header-chip">âš¡ <strong>Fusion:</strong> Multi-Evidence Arbitration</div>
            <div class="header-chip">ðŸ“ˆ <strong>Dynamics:</strong> Trend & Escalation Detection</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# INFORMATIONAL PIPELINE FLOW
# =========================================================

st.markdown(
    """
    <div class="pipeline-container">
        <div class="pipeline-step">
            <div class="step-number">STAGE 01</div>
            <div class="step-name">Audio Input</div>
            <div class="step-modality">WAV / MP3 Preprocessing</div>
        </div>
        <div class="pipeline-step" style="border-top: 2px solid var(--voice);">
            <div class="step-number" style="color: var(--voice);">STAGE 02</div>
            <div class="step-name">Voice Emotion</div>
            <div class="step-modality">Acoustic Wav2Vec2</div>
        </div>
        <div class="pipeline-step" style="border-top: 2px solid var(--text-accent);">
            <div class="step-number" style="color: var(--text-accent);">STAGE 03</div>
            <div class="step-name">Speech-to-Text</div>
            <div class="step-modality">Whisper Transcription</div>
        </div>
        <div class="pipeline-step" style="border-top: 2px solid var(--text-accent);">
            <div class="step-number" style="color: var(--text-accent);">STAGE 04</div>
            <div class="step-name">Text Sentiment</div>
            <div class="step-modality">RoBERTa Classifier</div>
        </div>
        <div class="pipeline-step" style="border-top: 2px solid var(--neutral);">
            <div class="step-number" style="color: var(--neutral);">STAGE 05</div>
            <div class="step-name">Speaker ID</div>
            <div class="step-modality">Pyannote Diarization</div>
        </div>
        <div class="pipeline-step" style="border-top: 2px solid var(--fused);">
            <div class="step-number" style="color: var(--fused);">STAGE 06</div>
            <div class="step-name">Multimodal Fusion</div>
            <div class="step-modality">Decision Arbitration</div>
        </div>
        <div class="pipeline-step" style="border-top: 2px solid var(--positive);">
            <div class="step-number" style="color: var(--positive);">STAGE 07</div>
            <div class="step-name">Conversation Analytics</div>
            <div class="step-modality">Trend & Escalation</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown(
    """
    <div class="section-header-bar">
        <div class="section-title">ðŸ“‚ Audio Ingestion & Analysis</div>
        <div class="section-badge">SUPPORTED: WAV Â· FLAC Â· MP3 Â· OGG Â· M4A</div>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_col, action_col = st.columns([3, 1])

with upload_col:
    uploaded_file = st.file_uploader(
        "Upload audio recording for multimodal analysis",
        type=["wav", "flac", "mp3", "ogg", "m4a"],
        help="Upload spoken audio or conversational recording. WAV format (16kHz) is recommended for highest acoustic fidelity.",
        label_visibility="collapsed",
    )

with action_col:
    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
    analyze_clicked = st.button(
        "ANALYZE AUDIO â–¶",
        type="primary",
        disabled=uploaded_file is None,
    )

    reset_clicked = st.button("NEW ANALYSIS â†»", use_container_width=True)
    if reset_clicked:
        st.rerun()

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        file_ext = Path(uploaded_file.name).suffix.upper().replace(".", "")
        st.markdown(
            f"""
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--muted); margin-top: 0.5rem; line-height: 1.4;">
                <div>FILE: <strong style="color: var(--text);">{html.escape(uploaded_file.name)}</strong></div>
                <div>SIZE: <strong style="color: var(--text);">{file_size_mb:.2f} MB</strong> Â· FORMAT: <strong style="color: var(--voice);">{file_ext}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )



def build_key_finding(
    dominant_voice: str,
    dominant_text: str,
    segments: List[Dict[str, Any]],
    trend_label: str,
    escalation_detected: bool,
) -> str:
    """Create a plain-English summary strictly from actual pipeline outputs."""
    voice = dominant_voice.title() if dominant_voice != "unknown" else "Unknown"
    text = dominant_text.title() if dominant_text != "unknown" else "Unknown"

    if escalation_detected:
        dynamic = "Continuous escalation was detected."
    elif trend_label.lower() == "increasing":
        dynamic = "Overall emotional intensity increased, but continuous escalation was not detected."
    elif trend_label.lower() == "decreasing":
        dynamic = "Overall emotional intensity decreased across the analyzed conversation."
    else:
        dynamic = "Overall emotional intensity remained relatively stable."

    if segments:
        first = str((segments[0].get("fusion") or {}).get("audio_direction", "")).lower()
        last = str((segments[-1].get("fusion") or {}).get("audio_direction", "")).lower()
        if first and last and first != last:
            transition = f"The dominant multimodal direction moved from {first} toward {last}."
        else:
            transition = "The multimodal emotional direction was broadly consistent across the conversation."
    else:
        transition = "No segment-level transition could be determined."

    return (
        f"Voice is predominantly {voice.lower()} while text sentiment is predominantly "
        f"{text.lower()}. {dynamic} {transition}"
    )


def render_fusion_explainer() -> None:
    """Explain the actual rule-based fusion configuration used by the project."""
    with st.expander("â“˜ How Multimodal Fusion Works", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                '<div class="modality-card voice-accent">'
                '<div class="card-meta-label">VOICE MODEL</div>'
                '<div class="card-highlight-value">Wav2Vec2-SUPERB ER</div>'
                '<div class="card-subtext">4 acoustic emotion classes: neutral, happy, angry, sad.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                '<div class="modality-card text-accent">'
                '<div class="card-meta-label">TEXT MODEL</div>'
                '<div class="card-highlight-value">CardiffNLP RoBERTa</div>'
                '<div class="card-subtext">3 sentiment classes: negative, neutral, positive.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                '<div class="modality-card fusion-accent">'
                '<div class="card-meta-label">FUSION CONFIGURATION</div>'
                '<div class="card-highlight-value">50% Voice + 50% Text</div>'
                '<div class="card-subtext">Confidence threshold: 40%. Agreement and mismatch are checked by emotional direction.</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div style="background:var(--panel-2);border:1px solid var(--border);
                        border-radius:6px;padding:0.9rem 1rem;margin-top:0.8rem;
                        color:var(--muted);font-size:0.78rem;line-height:1.6;">
                <strong style="color:var(--text);">Decision logic:</strong>
                voice emotions are mapped to positive/negative/neutral direction,
                text sentiment is already directional, then the system checks whether
                both modalities agree. Combined confidence is calculated from the
                configured 50/50 confidence-weighted combination.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_system_notes() -> None:
    """Show implementation notes without changing backend behavior."""
    with st.expander("â“˜ System Notes & Interpretation Guide", expanded=False):
        st.markdown(
            """
            <div style="color:var(--muted);font-size:0.78rem;line-height:1.7;">
            <strong style="color:var(--text);">Supported audio:</strong> WAV, FLAC, MP3, OGG, M4A<br>
            <strong style="color:var(--text);">Processing target:</strong> 16 kHz mono<br>
            <strong style="color:var(--text);">Voice classes:</strong> neutral, happy, angry, sad<br>
            <strong style="color:var(--text);">Text classes:</strong> negative, neutral, positive<br>
            <strong style="color:var(--text);">Fusion:</strong> transparent weighted rule-based decision fusion<br>
            <strong style="color:var(--text);">Escalation:</strong> requires at least 3 segments and continuously increasing intensity<br>
            <strong style="color:var(--text);">Interpretation:</strong> model predictions can vary with recording quality,
            background noise, speaker characteristics, and transcription accuracy.
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# RESULTS RENDERING FUNCTION
# =========================================================

def display_results(result: Dict[str, Any], audio_bytes: bytes = None, audio_format: str = "audio/wav") -> None:
    """Display comprehensive results from the multimodal conversation analyzer."""
    result = result or {}

    language = str(result.get("language", "Unknown")).upper()
    language_prob = safe_float(result.get("language_probability", 0.0))
    duration = safe_float(result.get("duration_seconds", 0.0))
    segments = result.get("segments", []) or []
    escalation = result.get("escalation", {}) or {}
    trend = result.get("emotional_trend", {}) or {}
    skipped_segments = int(result.get("skipped_segments", 0))

    escalation_detected = bool(escalation.get("escalation_detected", False))
    escalation_reason = str(escalation.get("reason", "")).strip()

    dominant_voice, dominant_text, agree_count, mismatch_count = derive_segment_summary(segments)

    trend_label = str(trend.get("trend", "unknown")).replace("_", " ").title()
    trend_delta = safe_float(trend.get("change", 0.0))

    # -----------------------------------------------------
    # KEY FINDING
    # -----------------------------------------------------
    finding = build_key_finding(
        dominant_voice=dominant_voice,
        dominant_text=dominant_text,
        segments=segments,
        trend_label=trend_label,
        escalation_detected=escalation_detected,
    )
    st.markdown(
        f"""
        <div class="modality-card fusion-accent" style="margin-bottom:1rem;">
            <div class="card-meta-label">KEY FINDING</div>
            <div style="font-size:0.98rem;font-weight:600;color:var(--text);line-height:1.55;">
                {html.escape(finding)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # 1. OVERVIEW METRICS
    # -----------------------------------------------------
    st.markdown(
        """
        <div class="section-header-bar">
            <div class="section-title">ðŸ“Š Analysis Overview</div>
            <div class="section-badge">SYSTEM LEVEL METRICS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4, m5 = st.columns(5)

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
            delta=f"{trend_delta:+.4f} Î”",
            delta_color="inverse" if trend_delta > 0.10 else "normal",
        )

    with m5:
        if escalation_detected:
            st.metric(
                label="ESCALATION STATUS",
                value="DETECTED ðŸš¨",
                delta="Elevated Risk",
                delta_color="inverse",
            )
        else:
            st.metric(
                label="ESCALATION STATUS",
                value="NOT DETECTED âœ“",
                delta="Stable",
                delta_color="normal",
            )

    # -----------------------------------------------------
    # 2. CHANNEL SNAPSHOT & MODALITY ALIGNMENT
    # -----------------------------------------------------
    st.markdown(
        """
        <div class="section-header-bar">
            <div class="section-title">ðŸŽšï¸ Modality Snapshot & Fusion Arbitration</div>
            <div class="section-badge">VOICE VS TEXT ALIGNMENT</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    snap_col1, snap_col2, snap_col3 = st.columns(3)

    with snap_col1:
        st.markdown(
            f"""
            <div class="modality-card voice-accent">
                <div class="card-meta-label">Acoustic Modality (Voice)</div>
                <div class="card-highlight-value" style="color: var(--voice);">
                    <span>{emotion_icon(dominant_voice)}</span>
                    <span>{html.escape(dominant_voice.title())}</span>
                </div>
                <div class="card-subtext">
                    Dominant acoustic emotion extracted via Wav2Vec2-SUPERB acoustic emotion classification.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with snap_col2:
        st.markdown(
            f"""
            <div class="modality-card text-accent">
                <div class="card-meta-label">Linguistic Modality (Text)</div>
                <div class="card-highlight-value" style="color: var(--text-accent);">
                    <span>{emotion_icon(dominant_text)}</span>
                    <span>{html.escape(dominant_text.title())}</span>
                </div>
                <div class="card-subtext">
                    Dominant textual sentiment classified via CardiffNLP RoBERTa transformer embeddings.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with snap_col3:
        total_segs = max(1, len(segments))
        agree_pct = (agree_count / total_segs) * 100
        st.markdown(
            f"""
            <div class="modality-card fusion-accent">
                <div class="card-meta-label">Cross-Modal Alignment</div>
                <div class="card-highlight-value" style="color: var(--fused);">
                    <span>âš¡ {agree_count}/{total_segs} Agree</span>
                </div>
                <div class="card-subtext">
                    Modality agreement: <strong style="color: var(--positive);">{agree_pct:.0f}%</strong> Â·
                    Mismatches: <strong style="color: var(--voice);">{mismatch_count}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


    # -----------------------------------------------------
    # SPEAKER SUMMARY
    # -----------------------------------------------------
    speaker_labels = [
        str(s.get("speaker", "")).strip()
        for s in segments
        if str(s.get("speaker", "")).strip()
        and str(s.get("speaker", "")).strip() != "Unknown"
    ]
    unique_speakers = list(dict.fromkeys(speaker_labels))
    diarization_error = str(
        result.get("diarization_error", "") or ""
    ).strip()

    st.markdown(
        """
        <div class="section-header-bar">
            <div class="section-title">ðŸ—£ï¸ Speaker Analysis</div>
            <div class="section-badge">DIARIZATION OUTPUT</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if diarization_error:
        st.warning(
            "Speaker diarization failed, so speakers show as Unknown. "
            f"Reason: {diarization_error}"
        )

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

    # -----------------------------------------------------
    # 3. AUDIO PLAYBACK & FULL TRANSCRIPT
    # -----------------------------------------------------
    st.markdown(
        """
        <div class="section-header-bar">
            <div class="section-title">ðŸŽ™ï¸ Audio Recording & Transcript</div>
            <div class="section-badge">WHISPER RECOGNITION</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    audio_col, transcript_col = st.columns([1, 2])

    with audio_col:
        with st.container(key="audio_player_box"):
            st.markdown(
                """
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: var(--muted); text-transform: uppercase; margin-bottom: 0.4rem; font-weight: 600;">
                    Audio Playback
                </div>
                """,
                unsafe_allow_html=True,
            )
            if audio_bytes:
                st.audio(audio_bytes, format=audio_format)
            else:
                st.caption("No audio preview stream available.")

    with transcript_col:
        full_transcript = str(result.get("full_transcript", "")).strip()
        if full_transcript:
            escaped_transcript = html.escape(full_transcript).replace("\n", "<br>")
            st.markdown(
                f"""
                <div class="transcript-container">
                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.64rem; font-weight: 700; color: var(--text-accent); text-transform: uppercase; margin-bottom: 0.4rem; letter-spacing: 0.08em;">
                        COMPLETE SPEECH TRANSCRIPT
                    </div>
                    {escaped_transcript}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("No speech transcript detected in audio stream.")

    # -----------------------------------------------------
    # 4. EMOTIONAL TREND & INTENSITY TRAJECTORY
    # -----------------------------------------------------
    st.markdown(
        """
        <div class="section-header-bar">
            <div class="section-title">ðŸ“ˆ Emotional Intensity Trajectory</div>
            <div class="section-badge">MULTIMODAL TEMPORAL DYNAMICS</div>
        </div>
        """,
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
                # Fixes the white bands: configure_view only fills the
                # plot rectangle itself, not the full chart canvas â€”
                # the padding where axis labels/titles live was left
                # at Vega-Lite's default white background.
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
            f"""
            <div style="background: var(--panel); border: 1px solid var(--border); border-left: 3px solid {'var(--negative)' if escalation_detected else 'var(--muted)'}; border-radius: 6px; padding: 0.75rem 1rem; margin-top: 0.5rem; font-size: 0.8rem; color: var(--text);">
                <strong style="font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; color: {'var(--negative)' if escalation_detected else 'var(--muted)'};">{'ðŸš¨' if escalation_detected else 'â„¹ï¸'} Escalation Diagnostics:</strong> {html.escape(escalation_reason)}
            </div>
            """,
            unsafe_allow_html=True,
        )


    # -----------------------------------------------------
    # TREND INTERPRETATION
    # -----------------------------------------------------
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
    st.markdown(
        f"""
        <div class="modality-card" style="margin-top:0.8rem;">
            <div class="card-meta-label">CONVERSATION DYNAMICS</div>
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
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------
    # 5. CONVERSATION SEGMENTS (FINE-GRAINED DETAILS)
    # -----------------------------------------------------
    st.markdown(
        """
        <div class="section-header-bar">
            <div class="section-title">ðŸ” Segment-Level Multimodal Breakdown</div>
            <div class="section-badge">TURN-BY-TURN INFERENCE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not segments:
        st.info("No conversational segments extracted from this recording.")
        return

    for segment in segments:
        idx = segment.get("segment_index", "?")
        start = safe_float(segment.get("start_seconds", 0.0))
        end = safe_float(segment.get("end_seconds", 0.0))
        text = str(segment.get("transcript", "")).strip()
        voice = segment.get("voice_emotion", {}) or {}
        sentiment = segment.get("text_sentiment", {}) or {}
        fusion = segment.get("fusion", {}) or {}
        intensity = safe_float(segment.get("intensity", 0.0))
        speaker = str(segment.get("speaker", "")).strip()

        v_emotion = str(voice.get("emotion", "unknown")).lower()
        v_conf = safe_float(voice.get("confidence", 0.0))
        t_sentiment = str(sentiment.get("sentiment", "unknown")).lower()
        t_conf = safe_float(sentiment.get("confidence", 0.0))
        f_conf = safe_float(fusion.get("combined_confidence", 0.0))

        agreement = bool(fusion.get("agreement", False))
        mismatch = bool(fusion.get("mismatch", False))
        interpretation = str(fusion.get("interpretation", "")).strip()
        audio_dir = direction_label(fusion.get("audio_direction", "unknown"))
        text_dir = direction_label(fusion.get("text_direction", "unknown"))

        # Header title
        if isinstance(idx, int):
            title_prefix = f"Turn {idx:02d}"
        else:
            title_prefix = f"Turn {idx}"

        speaker_tag = f" Â· {speaker}" if speaker and speaker != "Unknown" else ""

        expander_title = (
            f"{title_prefix}{speaker_tag} Â· [{start:.2f}s â†’ {end:.2f}s] Â· "
            f"Voice: {v_emotion.upper()} | Text: {t_sentiment.upper()} | Intensity: {intensity:.0%}"
        )

        with st.expander(expander_title, expanded=True):
            if text:
                speaker_prefix = f"{html.escape(speaker)}: " if speaker and speaker != "Unknown" else ""
                st.markdown(
                    f"""
                    <div class="segment-transcript-box">
                        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: var(--text-accent); font-weight: 700; text-transform: uppercase;">Transcript:</span> "{speaker_prefix}{html.escape(text)}"
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 3-Column Modality breakdown
            card_col1, card_col2, card_col3 = st.columns(3)

            with card_col1:
                st.markdown(
                    f"""
                    <div class="segment-card voice">
                        <div class="card-meta-label" style="color: var(--voice);">ðŸŽ™ï¸ VOICE EMOTION</div>
                        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 0.35rem;">
                            {emotion_icon(v_emotion)} {html.escape(v_emotion.upper())}
                        </div>
                        <div style="font-size: 0.72rem; color: var(--muted);">
                            Confidence: <strong style="color: var(--voice);">{v_conf:.1%}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with card_col2:
                st.markdown(
                    f"""
                    <div class="segment-card text">
                        <div class="card-meta-label" style="color: var(--text-accent);">ðŸ’¬ TEXT SENTIMENT</div>
                        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 0.35rem;">
                            {emotion_icon(t_sentiment)} {html.escape(t_sentiment.upper())}
                        </div>
                        <div style="font-size: 0.72rem; color: var(--muted);">
                            Confidence: <strong style="color: var(--text-accent);">{t_conf:.1%}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with card_col3:
                st.markdown(
                    f"""
                    <div class="segment-card fusion">
                        <div class="card-meta-label" style="color: var(--fused);">âš¡ MULTIMODAL FUSION</div>
                        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 0.35rem;">
                            INTENSITY: {intensity:.1%}
                        </div>
                        <div style="font-size: 0.72rem; color: var(--muted);">
                            Combined Conf: <strong style="color: var(--fused);">{f_conf:.1%}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Confidence meters
            st.markdown(
                render_channel_meters_html(
                    voice_label=v_emotion,
                    voice_confidence=v_conf,
                    text_label=t_sentiment,
                    text_confidence=t_conf,
                    fused_confidence=f_conf,
                ),
                unsafe_allow_html=True,
            )

            # Modality arbitration badge
            if agreement:
                badge_class = "ok"
                badge_text = "âœ“ Voice & Text Modalities Agree"
            elif mismatch:
                badge_class = "warn"
                badge_text = "âš  Voice / Text Modality Mismatch Detected"
            else:
                badge_class = "info"
                badge_text = "â„¹ Low-Confidence Disagreement"

            st.markdown(
                f"""
                <div style="background: var(--panel-2); border: 1px solid var(--border); border-radius: 6px; padding: 0.65rem 0.85rem; margin-top: 0.4rem;">
                    <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.35rem;">
                        <span class="status-badge {badge_class}">{badge_text}</span>
                        <span style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--muted);">
                            Voice Dir: <strong style="color: var(--text);">{html.escape(audio_dir)}</strong> &nbsp;|&nbsp;
                            Text Dir: <strong style="color: var(--text);">{html.escape(text_dir)}</strong>
                        </span>
                    </div>
                    <div style="font-size: 0.75rem; color: var(--muted); line-height: 1.4;">
                        <strong>Interpretation:</strong> {html.escape(interpretation) if interpretation else 'Standard multimodal arbitration.'}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# =========================================================
# MAIN EXECUTION FLOW & EMPTY STATE
# =========================================================

if analyze_clicked and uploaded_file is not None:
    suffix = Path(uploaded_file.name).suffix or ".wav"
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
            temp_audio.write(uploaded_file.getbuffer())
            temp_path = temp_audio.name

        with st.status("Executing Multimodal Analysis Pipeline...", expanded=True) as status_box:
            st.write("Initializing neural audio feature extractor and NLP tokenizer...")
            analyzer = load_analyzer()

            def pipeline_progress(stage: str, current: int, total: int) -> None:
                if total > 1:
                    status_box.update(
                        label=f"âš™ï¸ Multimodal Pipeline: {stage} ({current}/{total})...",
                        state="running",
                    )
                else:
                    status_box.update(
                        label=f"âš™ï¸ Multimodal Pipeline: {stage}...",
                        state="running",
                    )

            result = analyzer.analyze(temp_path, progress_callback=pipeline_progress)
            status_box.update(
                label="âœ“ Multimodal Analysis Completed Successfully",
                state="complete",
                expanded=False,
            )

        st.toast("Pipeline execution completed successfully.", icon="âœ…")

        display_results(
            result=result,
            audio_bytes=uploaded_file.getvalue(),
            audio_format=uploaded_file.type or "audio/wav",
        )

    except Exception as exc:
        st.error("Multimodal analysis pipeline failed.")
        st.exception(exc)

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass

elif uploaded_file is None:
    # Polished Empty State
    st.markdown(
        """
        <div class="empty-state-card">
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--voice); font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase;">
                AWAITING AUDIO INPUT
            </div>
            <div style="font-size: 1.15rem; font-weight: 700; color: var(--text); margin: 0.4rem 0;">
                Multimodal Voice & Conversation Emotion Analysis Engine
            </div>
            <div style="font-size: 0.82rem; color: var(--muted); max-width: 650px; margin: 0 auto;">
                Upload a spoken conversation or voice sample above to perform simultaneous acoustic emotion analysis, automatic speech transcription, sentiment classification, decision-level fusion, and escalation detection.
            </div>
            <div class="empty-steps-grid">
                <div class="empty-step-box">
                    <div class="empty-step-num">STEP 01</div>
                    <div class="empty-step-heading">Audio Ingestion</div>
                    <div class="empty-step-desc">Upload WAV, MP3, FLAC, OGG, or M4A audio. Preprocessed and resampled to 16 kHz mono.</div>
                </div>
                <div class="empty-step-box" style="border-top: 2px solid var(--voice);">
                    <div class="empty-step-num" style="color: var(--voice);">STEP 02</div>
                    <div class="empty-step-heading">Dual Modality Inference</div>
                    <div class="empty-step-desc">Wav2Vec2 captures acoustic emotional patterns while Whisper + RoBERTa transcribes and scores linguistic sentiment.</div>
                </div>
                <div class="empty-step-box" style="border-top: 2px solid var(--text-accent);">
                    <div class="empty-step-num" style="color: var(--text-accent);">STEP 03</div>
                    <div class="empty-step-heading">Fusion & Escalation</div>
                    <div class="empty-step-desc">Weighted rule-based fusion evaluates agreement/mismatch and tracks conversation intensity dynamics.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# EXPLAINABILITY & EXPORT
# =========================================================

if "result" in globals() and isinstance(result, dict) and result:
    render_fusion_explainer()
    render_system_notes()

    export_col1, export_col2, export_col3 = st.columns([1, 1, 2])
    with export_col1:
        st.download_button(
            "DOWNLOAD JSON REPORT",
            data=__import__("json").dumps(result, indent=2, default=str),
            file_name="multimodal_emotion_analysis_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with export_col2:
        rows = []
        for s in result.get("segments", []) or []:
            v = s.get("voice_emotion", {}) or {}
            t = s.get("text_sentiment", {}) or {}
            f = s.get("fusion", {}) or {}
            rows.append(
                {
                    "segment": s.get("segment_index", ""),
                    "start_seconds": s.get("start_seconds", ""),
                    "end_seconds": s.get("end_seconds", ""),
                    "speaker": s.get("speaker", ""),
                    "transcript": s.get("transcript", ""),
                    "voice_emotion": v.get("emotion", ""),
                    "voice_confidence": v.get("confidence", ""),
                    "text_sentiment": t.get("sentiment", ""),
                    "text_confidence": t.get("confidence", ""),
                    "fused_confidence": f.get("combined_confidence", ""),
                    "agreement": f.get("agreement", False),
                    "mismatch": f.get("mismatch", False),
                    "intensity": s.get("intensity", ""),
                }
            )
        csv_data = pd.DataFrame(rows).to_csv(index=False) if rows else ""
        st.download_button(
            "DOWNLOAD CSV REPORT",
            data=csv_data,
            file_name="multimodal_emotion_segments.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with export_col3:
        st.caption("Export contains the actual model outputs and segment-level analysis; no synthetic results are added.")


# =========================================================
# DASHBOARD FOOTER
# =========================================================

st.markdown(
    """
    <div class="dashboard-footer">
        MULTIMODAL VOICE & CONVERSATION EMOTION ANALYSIS SYSTEM &nbsp;Â·&nbsp;
        B.TECH MINI-PROJECT AI ANALYTICS ENGINE &nbsp;Â·&nbsp;
        CPU INFERENCE OPTIMIZED
    </div>
    """,
    unsafe_allow_html=True,
)
