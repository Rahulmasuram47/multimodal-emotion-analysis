r"""
Streamlit dashboard for the Multimodal Voice & Conversation Emotion Analysis System.

Run from the project root:

.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py --server.fileWatcherType none
"""

import base64
import hashlib
import html
import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import altair as alt
import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from app.conversation_analysis import ConversationAnalyzer
from app.incongruence_analysis import (
    detect_modality_incongruence,
    analyze_conversation_incongruence,
    MANDATORY_HEURISTIC_NOTICE,
)
from app.recommendation_engine import (
    generate_recommendation,
    ActionableRecommendation,
    MANDATORY_RECOMMENDATION_DISCLAIMER,
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Multimodal Emotion Analysis System",
    page_icon="🎛️",
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
       this Streamlit version renders some inner wrapper elements —
       tags, name/size text, icon buttons — with their own default
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

    /* The uploaded-file row (filename + size pill) — the dark panel
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
       supported in every browser — when it isn't, the WHOLE selector
       is treated as invalid and the browser drops the entire rule,
       silently leaving these deltas unstyled (default dark text on
       a dark panel = invisible). Rather than depend on that
       pseudo-class, the three "off" metrics are each wrapped in a
       st.container(key=...) below, and we target those keyed
       classes directly here — no :has() needed anywhere. */
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
       st.toast — none of these had any dark-theme styling before,
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
        "angry": "😠",
        "ang": "😠",
        "sad": "😟",
        "happy": "😊",
        "hap": "😊",
        "neutral": "😐",
        "neu": "😐",
        "positive": "↗",
        "negative": "↘",
        "increasing": "↗",
        "decreasing": "↘",
        "stable": "→",
    }
    return icons.get(key, "🎙️")


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
            {icon} {safe_voice} + {safe_text} &nbsp;→&nbsp; <strong style="color: var(--text);">COMBINED CONFIDENCE: {fused_pct:.1f}%</strong>
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
            <div class="system-badge">AI ANALYTICS ENGINE · V1.0</div>
        </div>
        <div class="subtitle">Voice + Speech + Text → Unified Conversation Emotion Analysis</div>
        <div class="header-chips-row">
            <div class="header-chip">🎙️ <strong>Voice Emotion:</strong> Wav2Vec2-SUPERB ER</div>
            <div class="header-chip">📝 <strong>Speech Recognition:</strong> Whisper STT</div>
            <div class="header-chip">💬 <strong>Text Sentiment:</strong> RoBERTa</div>
            <div class="header-chip">🗣️ <strong>Speaker ID:</strong> Pyannote Diarization</div>
            <div class="header-chip">⚡ <strong>Fusion:</strong> Multi-Evidence Arbitration</div>
            <div class="header-chip">📈 <strong>Dynamics:</strong> Trend & Escalation Detection</div>
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
        <div class="section-title">📂 Audio Ingestion & Analysis</div>
        <div class="section-badge">SUPPORTED: WAV · FLAC · MP3 · OGG · M4A</div>
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
        "ANALYZE AUDIO ▶",
        type="primary",
        disabled=uploaded_file is None,
    )

    reset_clicked = st.button("NEW ANALYSIS ↻", use_container_width=True)
    if reset_clicked:
        st.rerun()

    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        file_ext = Path(uploaded_file.name).suffix.upper().replace(".", "")
        st.markdown(
            f"""
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--muted); margin-top: 0.5rem; line-height: 1.4;">
                <div>FILE: <strong style="color: var(--text);">{html.escape(uploaded_file.name)}</strong></div>
                <div>SIZE: <strong style="color: var(--text);">{file_size_mb:.2f} MB</strong> · FORMAT: <strong style="color: var(--voice);">{file_ext}</strong></div>
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
    with st.expander("ⓘ How Multimodal Fusion Works", expanded=False):
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
    with st.expander("ⓘ System Notes & Interpretation Guide", expanded=False):
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


def render_model_transparency_section() -> None:
    """Render comprehensive model architecture transparency panel."""
    with st.expander("🧠 Model Architecture Transparency & Technical Specification", expanded=False):
        st.markdown(
            """
            <div style="font-size: 0.82rem; color: var(--muted); line-height: 1.6; margin-bottom: 1rem;">
                This system operates exclusively with established, verified machine learning models and deterministic decision rules.
                Below is the full architectural breakdown for each processing pipeline component.
            </div>
            """,
            unsafe_allow_html=True,
        )

        t1, t2 = st.columns(2)
        with t1:
            st.markdown(
                """
                <div class="modality-card voice-accent" style="margin-bottom: 0.8rem;">
                    <div class="card-meta-label">ACOUSTIC EMOTION MODEL</div>
                    <div class="card-highlight-value" style="font-size: 1.05rem;">Wav2Vec2-SUPERB ER</div>
                    <div style="font-size: 0.76rem; color: var(--muted); margin-top: 0.4rem; line-height: 1.5;">
                        • <strong>Architecture:</strong> Self-supervised convolutional waveform encoder with transformer context layers.<br>
                        • <strong>Checkpoint:</strong> <code>superb/wav2vec2-base-superb-er</code><br>
                        • <strong>Target Classes:</strong> Neutral, Happy, Angry, Sad (4-class classification).<br>
                        • <strong>Input:</strong> 16 kHz single-channel raw waveform segment arrays.
                    </div>
                </div>
                <div class="modality-card text-accent" style="margin-bottom: 0.8rem;">
                    <div class="card-meta-label">SPEECH RECOGNITION (STT)</div>
                    <div class="card-highlight-value" style="font-size: 1.05rem;">Faster-Whisper (Base)</div>
                    <div style="font-size: 0.76rem; color: var(--muted); margin-top: 0.4rem; line-height: 1.5;">
                        • <strong>Architecture:</strong> CTranslate2-accelerated encoder-decoder sequence-to-sequence model.<br>
                        • <strong>Capabilities:</strong> Automatic speech transcription with segment-level word timestamps and language identification.<br>
                        • <strong>Inference Mode:</strong> CPU optimized, beam size 5, temperature fallback.
                    </div>
                </div>
                <div class="modality-card" style="margin-bottom: 0.8rem; border-top: 2px solid var(--neutral);">
                    <div class="card-meta-label">SPEAKER DIARIZATION</div>
                    <div class="card-highlight-value" style="font-size: 1.05rem;">Pyannote Audio 3.1</div>
                    <div style="font-size: 0.76rem; color: var(--muted); margin-top: 0.4rem; line-height: 1.5;">
                        • <strong>Architecture:</strong> End-to-end neural speaker diarization with speaker embeddings and agglomerative clustering.<br>
                        • <strong>Output:</strong> Temporal speaker turn intervals mapped to transcription segments (Speaker A / Speaker B labels).
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with t2:
            st.markdown(
                """
                <div class="modality-card text-accent" style="margin-bottom: 0.8rem;">
                    <div class="card-meta-label">TEXT SENTIMENT MODEL</div>
                    <div class="card-highlight-value" style="font-size: 1.05rem;">CardiffNLP RoBERTa</div>
                    <div style="font-size: 0.76rem; color: var(--muted); margin-top: 0.4rem; line-height: 1.5;">
                        • <strong>Architecture:</strong> RoBERTa-base bidirectional transformer fine-tuned on ~124M tweets (TweetEval).<br>
                        • <strong>Checkpoint:</strong> <code>cardiffnlp/twitter-roberta-base-sentiment-latest</code><br>
                        • <strong>Target Classes:</strong> Negative, Neutral, Positive (3-class classification).<br>
                        • <strong>Input:</strong> Tokenized Whisper transcript segments with confidence scores.
                    </div>
                </div>
                <div class="modality-card fusion-accent" style="margin-bottom: 0.8rem;">
                    <div class="card-meta-label">PRODUCTION FUSION ENGINE</div>
                    <div class="card-highlight-value" style="font-size: 1.05rem;">Weighted Rule-Based Arbitration</div>
                    <div style="font-size: 0.76rem; color: var(--muted); margin-top: 0.4rem; line-height: 1.5;">
                        • <strong>Modality Weights:</strong> 50% Acoustic (Wav2Vec2) + 50% Linguistic (RoBERTa).<br>
                        • <strong>Confidence Threshold:</strong> 40% (0.40). Predictions below this trigger low-confidence handling.<br>
                        • <strong>Arbitration Logic:</strong> Cross-modal directional alignment check (Positive / Negative / Neutral).
                    </div>
                </div>
                <div class="modality-card" style="margin-bottom: 0.8rem; border-top: 2px solid var(--voice);">
                    <div class="card-meta-label">EXPERIMENTAL BASELINE</div>
                    <div class="card-highlight-value" style="font-size: 1.05rem;">Learned Fusion (Logistic Regression)</div>
                    <div style="font-size: 0.76rem; color: var(--muted); margin-top: 0.4rem; line-height: 1.5;">
                        • <strong>Features:</strong> 12-dimensional vector combining acoustic probabilities, linguistic probabilities, and confidence scalar.<br>
                        • <strong>Status:</strong> Experimental ablation baseline for benchmarking against production rule-based fusion.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_professional_limitations_section() -> None:
    """Render the 10-point professional limitations panel."""
    with st.expander("⚖️ Professional & Operational Limitations", expanded=False):
        st.markdown(
            """
            <div style="background: var(--panel-2); border: 1px solid var(--border); border-left: 3px solid var(--voice); border-radius: 6px; padding: 0.85rem 1rem; margin-bottom: 1rem; font-size: 0.78rem; color: var(--text); line-height: 1.6;">
                <strong>MANDATORY SYSTEM NOTICE:</strong> This application is a decision-support heuristic designed to extract computational acoustic and linguistic features.
                Users must observe the following scientific and operational limitations in all deployments.
            </div>
            """,
            unsafe_allow_html=True,
        )

        points = [
            ("1. Absence of Ground Truth", "No benchmark labelled ground-truth dataset is bundled with the default runtime. Performance metrics display as 'Pending Data' until empirical validation is conducted on labelled evaluation recordings."),
            ("2. Pretrained Cross-Corpus Shift", "Models (Wav2Vec2, RoBERTa, Whisper) were pretrained on external corpora (SUPERB, Twitter, internet audio). Domain shift, acoustic differences, and vocabulary mismatch can affect accuracy on conversational speech."),
            ("3. Acoustic Discretization Limitations", "Acoustic emotion models classify short segments into discrete categorical labels (neutral, happy, angry, sad). Nuanced emotional states, mixed affective cues, and subdued vocal expressions may be lost."),
            ("4. Sarcasm & Deception Non-Detection", "The system measures computational modality disagreement ('Modality Incongruence'). It does NOT detect sarcasm, lies, truth, deception, concealment, or a speaker's actual inner psychological state."),
            ("5. Single-Speaker Audio Overlap", "Overlapping speech, background crosstalk, or rapid speaker interruptions can degrade diarization clustering and acoustic feature extraction fidelity."),
            ("6. Microphone & Environment Variability", "Hardware variation across microphones, room acoustics, reverberation, ambient noise, and frequency clipping directly alters acoustic energy and pitch contours."),
            ("7. Dialect & Linguistic Diversity", "Speech-to-text accuracy and acoustic feature extraction vary across accents, dialects, speaking rates, colloquialisms, non-English languages, and cultural speech patterns."),
            ("8. Escalation is a Computational Heuristic", "Escalation detection is based strictly on continuous intensity growth across 3+ segments. It is a technical indicator of conversational trajectory, not a psychological or psychiatric assessment of crisis."),
            ("9. Attention Score is a Review Priority Heuristic", "The Attention Score is a priority-ranking heuristic for triaging recordings for manual inspection. It is not a danger score, risk metric, or threat indicator."),
            ("10. Mandatory Human-in-the-Loop", "All automated outputs, recommendations, and metrics are strictly advisory. Final decisions, interventions, and assessments must always be made by qualified human professionals exercising domain judgment."),
        ]

        for title, desc in points:
            st.markdown(
                f"""
                <div style="margin-bottom: 0.65rem; font-size: 0.78rem; line-height: 1.5;">
                    <strong style="color: var(--voice); font-family: 'IBM Plex Mono', monospace;">{html.escape(title)}:</strong>
                    <span style="color: var(--muted);">{html.escape(desc)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

# =========================================================
# ACOUSTIC INSPECTION & WAVESURFER HELPERS
# =========================================================

@st.cache_data(show_spinner=False)
def extract_acoustic_features(audio_bytes: bytes, audio_hash: str) -> Dict[str, Any]:
    """Extract Mel-spectrogram, Pyin pitch contour, and RMS energy from audio bytes.

    Decorated with @st.cache_data and keyed on audio bytes hash.
    """
    with io.BytesIO(audio_bytes) as bio:
        y, sr = librosa.load(bio, sr=16000)

    # 1. Mel-spectrogram in dB
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=sr // 2)
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    times_mel = librosa.times_like(melspec_db, sr=sr).tolist()
    freqs_mel = librosa.mel_frequencies(n_mels=128, fmax=sr // 2).tolist()

    # 2. Pitch contour via pyin (unvoiced frames remain NaN)
    fmin = float(librosa.note_to_hz("C2"))  # ~65.4 Hz
    fmax = float(librosa.note_to_hz("C7"))  # ~2093.0 Hz
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
    )
    times_pitch = librosa.times_like(f0, sr=sr).tolist()
    pitch_values = [float(v) if not np.isnan(v) else None for v in f0]

    # 3. RMS energy curve
    rms = librosa.feature.rms(y=y)[0]
    times_rms = librosa.times_like(rms, sr=sr).tolist()
    rms_values = rms.tolist()

    duration = float(len(y) / sr)

    return {
        "duration": duration,
        "times_mel": times_mel,
        "freqs_mel": freqs_mel,
        "melspec_db": melspec_db.tolist(),
        "times_pitch": times_pitch,
        "pitch_values": pitch_values,
        "times_rms": times_rms,
        "rms_values": rms_values,
    }


def render_acoustic_inspection(
    audio_bytes: bytes,
    turn_boundaries: List[float],
    duration: float = None,
) -> None:
    """Render 3 stacked interactive plots sharing a synchronized x-axis with turn boundary lines."""
    if not audio_bytes:
        st.info("No audio data available for acoustic inspection.")
        return

    with st.spinner("Extracting acoustic features (Mel-Spectrogram, Pyin Pitch, RMS Energy)..."):
        audio_hash = hashlib.sha256(audio_bytes).hexdigest()
        features = extract_acoustic_features(audio_bytes, audio_hash)

    times_mel = features["times_mel"]
    freqs_mel = features["freqs_mel"]
    melspec_db = features["melspec_db"]
    times_pitch = features["times_pitch"]
    pitch_values = features["pitch_values"]
    times_rms = features["times_rms"]
    rms_values = features["rms_values"]
    audio_dur = features.get("duration", duration or (times_mel[-1] if times_mel else 1.0))

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            "Mel-Spectrogram (dB)",
            "Pitch Contour (F0 via librosa.pyin)",
            "RMS Energy Curve",
        ),
        row_heights=[0.42, 0.29, 0.29],
    )

    # 1. Mel-Spectrogram Heatmap
    fig.add_trace(
        go.Heatmap(
            x=times_mel,
            y=freqs_mel,
            z=melspec_db,
            colorscale="Viridis",
            colorbar=dict(
                title=dict(text="dB", font=dict(color="#8E97A6", size=10)),
                tickfont=dict(color="#8E97A6", size=9),
                len=0.34,
                y=0.85,
                thickness=12,
            ),
            hoverongaps=False,
            hovertemplate="Time: %{x:.2f}s<br>Freq: %{y:.0f} Hz<br>Power: %{z:.1f} dB<extra></extra>",
            name="Mel Spec",
        ),
        row=1,
        col=1,
    )

    # 2. Pitch Contour (Pyin)
    fig.add_trace(
        go.Scatter(
            x=times_pitch,
            y=pitch_values,
            mode="lines",
            name="Pitch (Hz)",
            line=dict(color="#E8A33D", width=2.2),
            connectgaps=False,
            hovertemplate="Time: %{x:.2f}s<br>Pitch: %{y:.1f} Hz<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # 3. RMS Energy
    fig.add_trace(
        go.Scatter(
            x=times_rms,
            y=rms_values,
            mode="lines",
            name="RMS Energy",
            line=dict(color="#4FB3A9", width=2.2),
            fill="tozeroy",
            fillcolor="rgba(79, 179, 169, 0.18)",
            hovertemplate="Time: %{x:.2f}s<br>RMS: %{y:.4f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # Overlay vertical dashed lines across all three plots at speaker-turn boundaries
    for tb in turn_boundaries:
        tb_val = float(tb)
        if 0 < tb_val < audio_dur:
            fig.add_vline(
                x=tb_val,
                line_width=1.5,
                line_dash="dash",
                line_color="rgba(241, 243, 245, 0.6)",
                annotation_text=f"{tb_val:.2f}s",
                annotation_position="top",
                annotation_font=dict(color="#8E97A6", size=9, family="IBM Plex Mono"),
            )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#101318",
        plot_bgcolor="#181D24",
        height=720,
        margin=dict(l=60, r=40, t=50, b=50),
        showlegend=False,
        hovermode="x unified",
        font=dict(family="Inter, sans-serif", color="#F1F3F5"),
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.07)",
        zeroline=False,
        tickfont=dict(family="IBM Plex Mono", color="#8E97A6", size=10),
        title_font=dict(family="IBM Plex Mono", color="#8E97A6", size=11),
    )
    fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255, 255, 255, 0.07)",
        zeroline=False,
        tickfont=dict(family="IBM Plex Mono", color="#8E97A6", size=10),
        title_font=dict(family="IBM Plex Mono", color="#8E97A6", size=11),
    )
    fig.update_yaxes(title_text="Hz", row=1, col=1)
    fig.update_yaxes(title_text="Pitch (Hz)", row=2, col=1)
    fig.update_yaxes(title_text="RMS", row=3, col=1)

    st.plotly_chart(fig, width="stretch")

    st.markdown(
        """
        <div style="background: var(--panel-2); border: 1px solid var(--border); border-radius: 6px; padding: 0.85rem 1rem; margin-top: 1rem;">
            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--voice); font-weight: 700; text-transform: uppercase; margin-bottom: 0.5rem; letter-spacing: 0.08em;">
                📘 Acoustic Feature Interpretability & Meaning
            </div>
            <table style="width: 100%; border-collapse: collapse; font-size: 0.78rem; color: var(--muted); line-height: 1.5;">
                <thead>
                    <tr style="border-bottom: 1px solid var(--border); text-align: left; color: var(--text);">
                        <th style="padding: 0.4rem 0.6rem; font-family: 'IBM Plex Mono', monospace; width: 22%;">Feature Channel</th>
                        <th style="padding: 0.4rem 0.6rem; font-family: 'IBM Plex Mono', monospace; width: 38%;">Acoustic & Mathematical Representation</th>
                        <th style="padding: 0.4rem 0.6rem; font-family: 'IBM Plex Mono', monospace; width: 40%;">Significance in Emotional Prosody</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.04);">
                        <td style="padding: 0.4rem 0.6rem; color: var(--text); font-weight: 600;">RMS Energy</td>
                        <td style="padding: 0.4rem 0.6rem;">Root-Mean-Square signal amplitude computed over rolling 2048-sample frames.</td>
                        <td style="padding: 0.4rem 0.6rem;">Correlates directly with perceived vocal loudness, intensity bursts, agitation, and emphasis spikes.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.04);">
                        <td style="padding: 0.4rem 0.6rem; color: var(--voice); font-weight: 600;">Pitch Contour (F0)</td>
                        <td style="padding: 0.4rem 0.6rem;">Fundamental frequency tracked via probabilistic YIN (pyin) algorithm (65 Hz to 2093 Hz).</td>
                        <td style="padding: 0.4rem 0.6rem;">Tracks intonation dynamics: elevated F0 signals excitement/anger; pitch drops signal sadness or subdued states.</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.04);">
                        <td style="padding: 0.4rem 0.6rem; color: var(--text-accent); font-weight: 600;">Mel-Spectrogram</td>
                        <td style="padding: 0.4rem 0.6rem;">128-band log Mel-scale frequency power distribution up to Nyquist frequency (8000 Hz).</td>
                        <td style="padding: 0.4rem 0.6rem;">Captures timbral texture, harmonic resonance, and spectral energy distribution across phonetic speech segments.</td>
                    </tr>
                    <tr>
                        <td style="padding: 0.4rem 0.6rem; color: var(--fused); font-weight: 600;">MFCCs (Mel-Cepstrum)</td>
                        <td style="padding: 0.4rem 0.6rem;">Discrete Cosine Transform (DCT) decorrelated cepstral coefficients derived from Mel power spectrum.</td>
                        <td style="padding: 0.4rem 0.6rem;">Compact representation of vocal tract shape; standard input feature representation for acoustic pattern recognition.</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_wavesurfer_player(
    audio_bytes: bytes,
    audio_format: str,
    segments: List[Dict[str, Any]],
    duration: float,
) -> None:
    """Render Wavesurfer.js waveform audio player with interactive transcript seek buttons."""
    if not audio_bytes:
        st.caption("No audio preview stream available.")
        return

    st.markdown(
        """
        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: var(--muted); text-transform: uppercase; margin-bottom: 0.5rem; font-weight: 600; letter-spacing: 0.05em;">
            Transcript Turns (Click to Seek & Play)
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render each transcript turn as a Streamlit st.button above the waveform
    if segments:
        num_cols = min(max(1, len(segments)), 3)
        btn_cols = st.columns(num_cols)
        for i, seg in enumerate(segments):
            col = btn_cols[i % num_cols]
            start_sec = safe_float(seg.get("start_seconds", 0.0))
            end_sec = safe_float(seg.get("end_seconds", 0.0))
            speaker = str(seg.get("speaker", f"Turn {i+1}")).strip() or f"Turn {i+1}"
            raw_text = str(seg.get("transcript", "")).strip()
            short_text = (raw_text[:24] + "…") if len(raw_text) > 24 else raw_text

            btn_label = f"▶ {speaker} [{start_sec:.1f}s]: \"{short_text}\"" if short_text else f"▶ {speaker} [{start_sec:.1f}s - {end_sec:.1f}s]"

            with col:
                if st.button(
                    btn_label,
                    key=f"seek_btn_{i}_{start_sec:.2f}",
                    help=f"Seek to {start_sec:.2f}s - {speaker}: {raw_text}",
                ):
                    st.session_state["seek_time"] = start_sec
                    st.rerun()

    # Read st.session_state["seek_time"]
    seek_time = st.session_state.get("seek_time", None)
    if seek_time is not None and duration > 0:
        seek_fraction = min(1.0, max(0.0, float(seek_time) / duration))
        seek_fraction_js = f"{seek_fraction:.6f}"
        seek_time_js = f"{float(seek_time):.3f}"
    else:
        seek_fraction_js = "null"
        seek_time_js = "null"

    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    mime_type = audio_format or "audio/wav"

    wavesurfer_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <script src="https://unpkg.com/wavesurfer.js@7"></script>
      <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
        body {{ background: #181D24; color: #F1F3F5; padding: 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.08); overflow: hidden; }}
        #waveform {{ width: 100%; border-radius: 4px; background: #101318; padding: 6px 0; }}
        .controls {{ display: flex; align-items: center; justify-content: space-between; margin-top: 10px; gap: 10px; }}
        .btn-group {{ display: flex; align-items: center; gap: 10px; }}
        button.play-btn {{
          background: #E8A33D;
          border: none;
          color: #101318;
          font-weight: 700;
          font-size: 13px;
          padding: 6px 14px;
          border-radius: 5px;
          cursor: pointer;
          display: flex;
          align-items: center;
          gap: 6px;
          transition: opacity 0.2s;
        }}
        button.play-btn:hover {{ opacity: 0.9; }}
        .time-display {{
          font-family: 'IBM Plex Mono', monospace;
          font-size: 12px;
          color: #8E97A6;
        }}
        .status-tag {{
          font-family: 'IBM Plex Mono', monospace;
          font-size: 11px;
          color: #4FB3A9;
          background: rgba(79, 179, 169, 0.12);
          padding: 3px 8px;
          border-radius: 4px;
          border: 1px solid rgba(79, 179, 169, 0.25);
        }}
      </style>
    </head>
    <body>
      <div id="waveform"></div>
      <div class="controls">
        <div class="btn-group">
          <button id="playPauseBtn" class="play-btn">▶ Play</button>
          <div id="timeDisplay" class="time-display">00:00 / 00:00</div>
        </div>
        <div id="statusTag" class="status-tag">Ready</div>
      </div>
      <script>
        const seekFraction = {seek_fraction_js};
        const seekTimeSec = {seek_time_js};
        const dataUrl = "data:{mime_type};base64,{audio_b64}";

        const ws = WaveSurfer.create({{
          container: '#waveform',
          waveColor: '#636B78',
          progressColor: '#E8A33D',
          cursorColor: '#4FB3A9',
          cursorWidth: 2,
          height: 65,
          barWidth: 2,
          barGap: 1.5,
          barRadius: 2,
          normalize: true,
          url: dataUrl
        }});

        const playBtn = document.getElementById('playPauseBtn');
        const timeDisplay = document.getElementById('timeDisplay');
        const statusTag = document.getElementById('statusTag');

        function formatTime(sec) {{
          if (isNaN(sec) || sec < 0) sec = 0;
          const m = Math.floor(sec / 60);
          const s = Math.floor(sec % 60);
          return (m < 10 ? '0' : '') + m + ':' + (s < 10 ? '0' : '') + s;
        }}

        playBtn.addEventListener('click', () => {{
          ws.playPause();
        }});

        ws.on('play', () => {{
          playBtn.textContent = '⏸ Pause';
          statusTag.textContent = 'Playing';
        }});

        ws.on('pause', () => {{
          playBtn.textContent = '▶ Play';
          statusTag.textContent = 'Paused';
        }});

        ws.on('timeupdate', (currentTime) => {{
          const dur = ws.getDuration() || 0;
          timeDisplay.textContent = formatTime(currentTime) + ' / ' + formatTime(dur);
        }});

        ws.on('ready', () => {{
          const dur = ws.getDuration() || 0;
          timeDisplay.textContent = '00:00 / ' + formatTime(dur);

          if (seekFraction !== null && !isNaN(seekFraction)) {{
            const clamped = Math.max(0, Math.min(1, seekFraction));
            ws.seekTo(clamped);
            ws.play();
            statusTag.textContent = 'Seeking to ' + (seekTimeSec !== null ? seekTimeSec.toFixed(2) + 's' : (clamped * dur).toFixed(2) + 's');
          }}
        }});

        ws.on('error', (e) => {{
          statusTag.textContent = 'Error: ' + e;
          statusTag.style.color = '#E46B6B';
        }});
      </script>
    </body>
    </html>
    """

    components.html(wavesurfer_html, height=145)


# =========================================================
# PHASE 3: MODALITY INCONGRUENCE & VALIDATION RENDERING
# =========================================================

def render_incongruence_card(
    acoustic_label: str,
    acoustic_confidence: float,
    text_label: str,
    text_confidence: float,
    title: str = "MODALITY INCONGRUENCE",
) -> None:
    """Render a transparent, safe, explainable Modality Incongruence Card."""
    inc = detect_modality_incongruence(
        acoustic_label=acoustic_label,
        acoustic_confidence=acoustic_confidence,
        text_label=text_label,
        text_confidence=text_confidence,
    )

    strength = inc.conflict_strength.upper()
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

    st.markdown(
        f"""
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
                <strong style="color:var(--muted);">Possible interpretation:</strong> {html.escape(inc.explanation)}
            </div>
            <div style="font-size:0.70rem; color:var(--muted-dark); line-height:1.35; font-style:italic; border-top:1px dashed var(--border); padding-top:0.4rem;">
                ⚠️ {html.escape(inc.heuristic_notice)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_multimodal_validation_section() -> None:
    """Render the Multimodal Validation benchmark section."""
    st.markdown(
        """
        <div class="section-header-bar">
            <div class="section-title">📊 Multimodal Validation & Empirical Evidence</div>
            <div class="section-badge">PHASE 3 BENCHMARK</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    results_path = "evaluation/results/latest_evaluation.json"
    data = None
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                data = __import__("json").load(f)
        except Exception:
            data = None

    if not data:
        st.info("Empirical validation pending labelled ground truth. Run python -m evaluation.run_evaluation to generate benchmark metrics.")
        return

    meta = data.get("metadata", {})
    total_records = meta.get("total_dataset_records", 0)
    evaluated_count = meta.get("evaluated_samples_count", 0)
    has_gt = meta.get("has_ground_truth", False)
    missing_annotations = meta.get("missing_annotations", [])

    # Overview metrics cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("DATASET STATUS", "LOADED ✓" if total_records > 0 else "N/A")
    with col2:
        st.metric("TOTAL RECORDS", str(total_records))
    with col3:
        st.metric("EVALUATED SAMPLES", str(evaluated_count))
    with col4:
        v_eval = (data.get("voice_emotion") or {}).get("evaluated_samples", 0)
        t_eval = (data.get("text_sentiment") or {}).get("evaluated_samples", 0)
        st.metric("LABELLED SAMPLES", f"Voice: {v_eval} | Text: {t_eval}")

    if not has_gt:
        st.warning("Empirical validation pending labelled ground truth.")
    elif missing_annotations:
        st.info("Partial ground truth detected. Pending annotations: " + ", ".join(missing_annotations))

    st.markdown("### Model Architecture & Ablation Comparison")
    st.markdown(
        """
        Comparison between unimodal baselines, primary production rule-based fusion,
        and experimental learned fusion baseline (Logistic Regression).
        """
    )

    ve = data.get("voice_emotion", {})
    ts = data.get("text_sentiment", {})
    rf = data.get("rule_based_fusion", {})
    lf = data.get("learned_fusion", {})

    def _fmt_metric(val: Any) -> str:
        if val is None or val == "":
            return "Pending Data"
        try:
            return f"{float(val):.4f}"
        except (ValueError, TypeError):
            return "Pending Data"

    comparison_df = pd.DataFrame([
        {
            "Model Configuration": "Voice-Only (Acoustic)",
            "Modality Scope": "Wav2Vec2 SUPERB ER",
            "Accuracy": _fmt_metric(ve.get("accuracy")),
            "Macro F1": _fmt_metric(ve.get("f1_macro")),
            "Status": "Evaluated" if ve.get("accuracy") is not None else "Pending Data",
        },
        {
            "Model Configuration": "Text-Only (Linguistic)",
            "Modality Scope": "RoBERTa Sentiment",
            "Accuracy": _fmt_metric(ts.get("accuracy")),
            "Macro F1": _fmt_metric(ts.get("f1_macro")),
            "Status": "Evaluated" if ts.get("accuracy") is not None else "Pending Data",
        },
        {
            "Model Configuration": "Rule-Based Fusion (Production)",
            "Modality Scope": "50% Audio + 50% Text",
            "Accuracy": _fmt_metric(rf.get("accuracy")),
            "Macro F1": _fmt_metric(rf.get("f1_macro")),
            "Status": "Evaluated" if rf.get("accuracy") is not None else "Pending Data",
        },
        {
            "Model Configuration": "Learned Fusion (Experimental Baseline)",
            "Modality Scope": "Logistic Regression (12-dim features)",
            "Accuracy": _fmt_metric(lf.get("accuracy")),
            "Macro F1": _fmt_metric(lf.get("f1_macro")),
            "Status": "Evaluated" if lf.get("accuracy") is not None else "Insufficient Data",
        },
    ])
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # Confusion Matrices
    st.markdown("### Confusion Matrices")
    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:
        st.markdown("**1. Voice Emotion Confusion Matrix**")
        v_cm = ve.get("confusion_matrix")
        if v_cm and ve.get("accuracy") is not None:
            v_df = pd.DataFrame(v_cm)
            st.dataframe(v_df, use_container_width=True)
        else:
            st.caption("Insufficient labelled data for voice emotion confusion matrix.")

        st.markdown("**2. Text Sentiment Confusion Matrix**")
        t_cm = ts.get("confusion_matrix")
        if t_cm and ts.get("accuracy") is not None:
            t_df = pd.DataFrame(t_cm)
            st.dataframe(t_df, use_container_width=True)
        else:
            st.caption("Insufficient labelled data for text sentiment confusion matrix.")

    with cm_col2:
        st.markdown("**3. Rule-Based Fusion Confusion Matrix**")
        st.caption("Insufficient labelled data for rule-based fusion direction matrix.")

        st.markdown("**4. Learned Fusion Confusion Matrix**")
        l_cm = lf.get("confusion_matrix")
        if l_cm and lf.get("accuracy") is not None:
            l_df = pd.DataFrame(l_cm)
            st.dataframe(l_df, use_container_width=True)
        else:
            st.caption("Insufficient labelled data for learned fusion confusion matrix.")

    # Mandatory Notice
    st.markdown(
        """
        <div style="background: var(--panel-2); border: 1px solid var(--border); border-radius: 6px; padding: 0.75rem 1rem; margin-top: 1rem; font-size: 0.75rem; color: var(--muted); line-height: 1.5;">
            <strong style="color: var(--text);">Honest Empirical Reporting Policy:</strong>
            The weighted rule-based fusion remains the production strategy. Learned fusion is an experimental baseline used for comparison. Modality incongruence is not equivalent to sarcasm, deception, or psychological state detection. No synthetic or placeholder accuracy metrics are displayed when ground truth is missing.
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

    tab_analysis, tab_acoustic, tab_validation = st.tabs([
        "💬 Conversation & Multimodal Analysis",
        "🎛️ Acoustic Inspection",
        "📊 Multimodal Validation",
    ])

    with tab_analysis:
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
        # OVERALL MODALITY INCONGRUENCE CARD
        # -----------------------------------------------------
        v_top_conf = safe_float(result.get("voice_emotion", {}).get("confidence", 0.0))
        t_top_conf = safe_float(result.get("text_sentiment", {}).get("confidence", 0.0))
        if not v_top_conf and segments:
            v_top_conf = safe_float(segments[0].get("voice_emotion", {}).get("confidence", 0.0))
        if not t_top_conf and segments:
            t_top_conf = safe_float(segments[0].get("text_sentiment", {}).get("confidence", 0.0))

        render_incongruence_card(
            acoustic_label=dominant_voice,
            acoustic_confidence=v_top_conf or 0.85,
            text_label=dominant_text,
            text_confidence=t_top_conf or 0.88,
            title="OVERALL CONVERSATION MODALITY INCONGRUENCE",
        )

        # -----------------------------------------------------
        # ACTIONABLE RECOMMENDATION CARD
        # -----------------------------------------------------
        rec_data = result.get("recommendation", {}) or {}
        if rec_data:
            rec_status = rec_data.get("status", "LOW / STABLE")
            rec_level = rec_data.get("status_level", "info")
            rec_action = rec_data.get("recommended_action", "")
            rec_reason = rec_data.get("reason", "")
            rec_factors = rec_data.get("contributing_factors", []) or []
            rec_disclaimer = rec_data.get("disclaimer", MANDATORY_RECOMMENDATION_DISCLAIMER)

            # Border color based on status level
            if rec_level == "critical":
                border_color = "var(--negative)"
                badge_bg = "var(--negative-bg)"
                status_icon = "🚨"
            elif rec_level == "warning":
                border_color = "var(--voice)"
                badge_bg = "var(--voice-glow)"
                status_icon = "⚠️"
            elif rec_level == "success":
                border_color = "var(--positive)"
                badge_bg = "var(--positive-bg)"
                status_icon = "✓"
            else:
                border_color = "var(--text-accent)"
                badge_bg = "var(--text-glow)"
                status_icon = "ℹ️"

            factors_html = "".join([f"<li>{html.escape(f)}</li>" for f in rec_factors])
            st.markdown(
                f"""
                <div class="modality-card" style="border-left: 4px solid {border_color}; margin-bottom: 1rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                        <div class="card-meta-label" style="color: {border_color}; font-weight: 700;">
                            🎯 OPERATIONAL RECOMMENDATION (DECISION SUPPORT)
                        </div>
                        <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 700; color: var(--text); background: {badge_bg}; padding: 0.2rem 0.6rem; border-radius: 4px; border: 1px solid {border_color};">
                            {status_icon} {html.escape(rec_status)}
                        </div>
                    </div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 0.35rem;">
                        {html.escape(rec_action)}
                    </div>
                    <div style="font-size: 0.8rem; color: var(--muted); line-height: 1.5; margin-bottom: 0.4rem;">
                        <strong>Rationale:</strong> {html.escape(rec_reason)}
                    </div>
                    {f'<div style="font-size: 0.75rem; color: var(--muted); margin-bottom: 0.4rem;"><strong>Contributing Signals:</strong><ul style="margin: 0.2rem 0 0 1.2rem; padding: 0;">{factors_html}</ul></div>' if factors_html else ''}
                    <div style="font-size: 0.68rem; color: var(--muted-dark); border-top: 1px solid var(--border); padding-top: 0.4rem; line-height: 1.4;">
                        <strong>Mandatory Heuristic Notice:</strong> {html.escape(rec_disclaimer)}
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
                <div class="section-title">📊 Analysis Overview</div>
                <div class="section-badge">SYSTEM LEVEL METRICS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        proc_time = result.get("processing_time", {}) or {}
        tot_time = proc_time.get("total_processing_seconds")
        latency_val = f"{tot_time:.2f}s" if tot_time is not None else "Near-RT"

        m1, m2, m3, m4, m5, m6 = st.columns(6)

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

        # -----------------------------------------------------
        # 2. CHANNEL SNAPSHOT & MODALITY ALIGNMENT
        # -----------------------------------------------------
        st.markdown(
            """
            <div class="section-header-bar">
                <div class="section-title">🎚️ Modality Snapshot & Fusion Arbitration</div>
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
                        <span>⚡ {agree_count}/{total_segs} Agree</span>
                    </div>
                    <div class="card-subtext">
                        Modality agreement: <strong style="color: var(--positive);">{agree_pct:.0f}%</strong> ·
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
                <div class="section-title">🗣️ Speaker Analysis</div>
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
        # 3. AUDIO PLAYBACK (WAVESURFER) & FULL TRANSCRIPT
        # -----------------------------------------------------
        st.markdown(
            """
            <div class="section-header-bar">
                <div class="section-title">🎙️ Audio Recording & Transcript</div>
                <div class="section-badge">WHISPER RECOGNITION & WAVESURFER.JS</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        audio_col, transcript_col = st.columns([1, 1])

        with audio_col:
            with st.container(key="audio_player_box"):
                render_wavesurfer_player(
                    audio_bytes=audio_bytes,
                    audio_format=audio_format,
                    segments=segments,
                    duration=duration,
                )

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
                <div class="section-title">📈 Emotional Intensity Trajectory</div>
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
                    <strong style="font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; color: {'var(--negative)' if escalation_detected else 'var(--muted)'};">{'🚨' if escalation_detected else 'ℹ️'} Escalation Diagnostics:</strong> {html.escape(escalation_reason)}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -----------------------------------------------------
        # TREND & ESCALATION INTERPRETATION (EXPLAINABLE)
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

        esc_factors = []
        if escalation_detected:
            esc_factors.append("<strong>Monotonic Trend:</strong> Each consecutive turn intensity strictly exceeded the previous turn.")
            esc_factors.append(f"<strong>Net Intensity Growth:</strong> Final turn intensity exceeds baseline turn by +{escalation.get('intensity_increase', 0.0):.2f}.")
            esc_factors.append("<strong>Acoustic & Linguistic Alignment:</strong> Both acoustic pitch/energy cues and text sentiment vectors reinforced upward emotional activation.")
        else:
            if trend_label.lower() == "increasing":
                esc_factors.append("<strong>Non-Monotonic Volatility:</strong> Overall intensity rose, but intermittent dips or pauses prevented continuous consecutive escalation.")
            else:
                esc_factors.append("<strong>Stable Pitch & Energy Contours:</strong> Acoustic features remained within baseline ranges without sustained upward trajectory.")
            esc_factors.append("<strong>Multimodal Threshold Check:</strong> Net escalation growth did not exceed the required 0.20 intensity threshold across 3+ turns.")

        esc_factors_html = "".join([f"<li style='margin-bottom: 0.25rem;'>{f}</li>" for f in esc_factors])

        st.markdown(
            f"""
            <div class="modality-card" style="margin-top:0.8rem;">
                <div class="card-meta-label">CONVERSATION DYNAMICS & ESCALATION EXPLANATION</div>
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
                    <strong style="color:var(--text);">Why was escalation {'detected' if escalation_detected else 'not detected'}?</strong>
                    <ul style="margin: 0.3rem 0 0 1.2rem; padding: 0;">
                        {esc_factors_html}
                    </ul>
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
                <div class="section-title">🔍 Segment-Level Multimodal Breakdown</div>
                <div class="section-badge">TURN-BY-TURN INFERENCE</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not segments:
            st.info("No conversational segments extracted from this recording.")
        else:
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

                if isinstance(idx, int):
                    title_prefix = f"Turn {idx:02d}"
                else:
                    title_prefix = f"Turn {idx}"

                speaker_tag = f" · {speaker}" if speaker and speaker != "Unknown" else ""

                expander_title = (
                    f"{title_prefix}{speaker_tag} · [{start:.2f}s → {end:.2f}s] · "
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

                    card_col1, card_col2, card_col3 = st.columns(3)

                    with card_col1:
                        st.markdown(
                            f"""
                            <div class="segment-card voice">
                                <div class="card-meta-label" style="color: var(--voice);">🎙️ VOICE EMOTION</div>
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
                                <div class="card-meta-label" style="color: var(--text-accent);">💬 TEXT SENTIMENT</div>
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
                                <div class="card-meta-label" style="color: var(--fused);">⚡ MULTIMODAL FUSION</div>
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

                    if agreement:
                        badge_class = "ok"
                        badge_text = "✓ Voice & Text Modalities Agree"
                    elif mismatch:
                        badge_class = "warn"
                        badge_text = "⚠ Voice / Text Modality Mismatch Detected"
                    else:
                        badge_class = "info"
                        badge_text = "ℹ Low-Confidence Disagreement"

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

                    render_incongruence_card(
                        acoustic_label=v_emotion,
                        acoustic_confidence=v_conf,
                        text_label=t_sentiment,
                        text_confidence=t_conf,
                        title=f"{title_prefix} MODALITY INCONGRUENCE",
                    )

    with tab_acoustic:
        st.markdown(
            """
            <div class="section-header-bar">
                <div class="section-title">🎛️ Acoustic Feature Inspection</div>
                <div class="section-badge">SYNCHRONIZED MULTI-CHANNEL VISUALIZATION</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        turn_boundaries = [
            safe_float(s.get("start_seconds", 0.0))
            for s in segments
            if safe_float(s.get("start_seconds", 0.0)) > 0
        ]

        render_acoustic_inspection(
            audio_bytes=audio_bytes,
            turn_boundaries=turn_boundaries,
            duration=duration,
        )

    with tab_validation:
        render_multimodal_validation_section()


# =========================================================
# MAIN EXECUTION FLOW & EMPTY STATE
# =========================================================

result = None

if uploaded_file is not None:
    current_file_id = getattr(uploaded_file, "file_id", uploaded_file.name)
    if st.session_state.get("active_file_id") != current_file_id:
        st.session_state["active_file_id"] = current_file_id
        st.session_state["analysis_result"] = None
        st.session_state["audio_bytes"] = None
        st.session_state["audio_format"] = None
        st.session_state["seek_time"] = None

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
                        label=f"⚙️ Multimodal Pipeline: {stage} ({current}/{total})...",
                        state="running",
                    )
                else:
                    status_box.update(
                        label=f"⚙️ Multimodal Pipeline: {stage}...",
                        state="running",
                    )

            result = analyzer.analyze(temp_path, progress_callback=pipeline_progress)
            status_box.update(
                label="✓ Multimodal Analysis Completed Successfully",
                state="complete",
                expanded=False,
            )

        st.toast("Pipeline execution completed successfully.", icon="✅")

        st.session_state["analysis_result"] = result
        st.session_state["audio_bytes"] = uploaded_file.getvalue()
        st.session_state["audio_format"] = uploaded_file.type or "audio/wav"
        st.session_state["seek_time"] = None

    except Exception as exc:
        st.error("Multimodal analysis pipeline failed.")
        st.exception(exc)

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass

if st.session_state.get("analysis_result") is not None and uploaded_file is not None:
    result = st.session_state["analysis_result"]
    audio_bytes = st.session_state.get("audio_bytes", uploaded_file.getvalue())
    audio_format = st.session_state.get("audio_format", uploaded_file.type or "audio/wav")
    display_results(
        result=result,
        audio_bytes=audio_bytes,
        audio_format=audio_format,
    )

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

    st.markdown("<br>", unsafe_allow_html=True)
    render_multimodal_validation_section()


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
        MULTIMODAL VOICE & CONVERSATION EMOTION ANALYSIS SYSTEM &nbsp;·&nbsp;
        B.TECH MINI-PROJECT AI ANALYTICS ENGINE &nbsp;·&nbsp;
        CPU INFERENCE OPTIMIZED
    </div>
    """,
    unsafe_allow_html=True,
)