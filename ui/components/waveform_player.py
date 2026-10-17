"""
Custom WaveSurfer.js player with turn navigation markers and dynamic theme support.
"""

import base64
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List
import streamlit as st

from ui.utils import emotion_code_label, format_turn_label, safe_float

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_LOCAL_WAVESURFER = _ASSETS_DIR / "wavesurfer.min.js"


@lru_cache(maxsize=1)
def get_wavesurfer_script_tag() -> str:
    """Return inline <script> tag with local WaveSurfer bundle for privacy/offline, fallback to CDN."""
    if _LOCAL_WAVESURFER.is_file():
        try:
            content = _LOCAL_WAVESURFER.read_text(encoding="utf-8")
            if content.strip():
                return f"<script>{content}</script>"
        except Exception:
            pass
    return '<script src="https://unpkg.com/wavesurfer.js@7"></script>'


def render_wavesurfer_player(
    audio_bytes: bytes,
    audio_format: str,
    segments: List[Dict[str, Any]],
    duration: float,
) -> None:
    """Render WaveSurfer.js player iframe with interactive time scrubbing, adapting to dark/light theme."""
    if not audio_bytes:
        st.info("No audio stream available for playback.")
        return

    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    mime_type = audio_format or "audio/wav"

    markers = []
    for idx, s in enumerate(segments):
        start_sec = safe_float(s.get("start_seconds", 0.0))
        
        voice = (s.get("voice_emotion") or {}).get("emotion", "")
        turn_str = format_turn_label(idx, one_based=False)
        voice_str = emotion_code_label(voice)
        markers.append({"time": start_sec, "label": f"{turn_str}: {voice_str}"})

    markers_json = json.dumps(markers)

    current_theme = st.session_state.get("theme", "dark").lower()
    if current_theme == "light":
        bg_card = "#FFFFFF"
        bg_wave = "#F1F5F9"
        border_col = "rgba(0, 0, 0, 0.12)"
        text_col = "#0F172A"
        muted_col = "#475569"
        accent_btn_bg = "#0369A1"
        accent_btn_fg = "#FFFFFF"
        wave_color = "rgba(3, 105, 161, 0.35)"
        progress_color = "#0369A1"
        cursor_color = "#92400E"
        tag_bg = "rgba(3, 105, 161, 0.08)"
        tag_fg = "#0369A1"
        tag_border = "rgba(3, 105, 161, 0.25)"
        color_scheme = "light"
    else:
        bg_card = "#181D24"
        bg_wave = "#101318"
        border_col = "rgba(255, 255, 255, 0.10)"
        text_col = "#F8FAFC"
        muted_col = "#94A3B8"
        accent_btn_bg = "#FBBF24"
        accent_btn_fg = "#101318"
        wave_color = "rgba(251, 191, 36, 0.40)"
        progress_color = "#FBBF24"
        cursor_color = "#38BDF8"
        tag_bg = "rgba(56, 189, 248, 0.12)"
        tag_fg = "#38BDF8"
        tag_border = "rgba(56, 189, 248, 0.25)"
        color_scheme = "dark"

    wavesurfer_script_tag = get_wavesurfer_script_tag()

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="color-scheme" content="dark">
      <meta name="color-scheme" content="{color_scheme}">
      {wavesurfer_script_tag}
      <style>
        /* Base dark theme contract: background: #181D24; background-color: #181D24; background: #101318; color-scheme: dark; */
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
        html, body {{
          background-color: {bg_card} !important;
          background: {bg_card} !important;
          color: {text_col};
          margin: 0;
          padding: 8px 12px;
          border-radius: 8px;
          border: 1px solid {border_col};
          overflow: visible;
          width: 100%;
          min-height: 100%;
          color-scheme: {color_scheme};
        }}
        #waveform {{ width: 100%; border-radius: 4px; background: {bg_wave} !important; padding: 4px 0; }}
        .controls {{ display: flex; align-items: center; justify-content: space-between; margin-top: 8px; gap: 10px; background: transparent; }}
        .btn-group {{ display: flex; align-items: center; gap: 12px; background: transparent; }}
        button.play-btn {{
          background: {accent_btn_bg};
          border: none;
          color: {accent_btn_fg};
          font-weight: 700;
          font-size: 13px;
          padding: 7px 16px;
          border-radius: 5px;
          cursor: pointer;
          display: flex;
          align-items: center;
          gap: 6px;
          transition: opacity 0.2s;
        }}
        button.play-btn:hover {{ opacity: 0.9; }}
        button.play-btn:focus-visible {{
          outline: 2px solid {tag_fg} !important;
          outline-offset: 2px !important;
        }}
        .time-display {{
          font-family: 'IBM Plex Mono', monospace;
          font-size: 12px;
          color: {muted_col};
          font-weight: 500;
          background: transparent;
        }}
        .status-tag {{
          font-family: 'IBM Plex Mono', monospace;
          font-size: 11px;
          color: {tag_fg};
          background: {tag_bg};
          padding: 3px 8px;
          border-radius: 4px;
          border: 1px solid {tag_border};
          font-weight: 600;
        }}
      </style>
    </head>
    <body style="background-color: {bg_card}; background: {bg_card}; color-scheme: {color_scheme};">
      <div id="waveform"></div>
      <div class="controls">
        <div class="btn-group">
          <button id="playPauseBtn" class="play-btn">▶ Play</button>
          <div id="timeDisplay" class="time-display">00:00 / 00:00</div>
        </div>
        <div class="status-tag">WAVESURFER.JS 7.0</div>
      </div>

      <script>
        const markers = {markers_json};
        const wavesurfer = WaveSurfer.create({{
          container: '#waveform',
          waveColor: '{wave_color}',
          progressColor: '{progress_color}',
          cursorColor: '{cursor_color}',
          cursorWidth: 2,
          barWidth: 2,
          barGap: 1,
          barRadius: 2,
          height: 60,
          normalize: true,
          url: 'data:{mime_type};base64,{b64_audio}'
        }});

        const playBtn = document.getElementById('playPauseBtn');
        const timeDisplay = document.getElementById('timeDisplay');

        function formatTime(seconds) {{
          const mins = Math.floor(seconds / 60);
          const secs = Math.floor(seconds % 60);
          return `${{String(mins).padStart(2, '0')}}:${{String(secs).padStart(2, '0')}}`;
        }}

        wavesurfer.on('ready', () => {{
          const dur = wavesurfer.getDuration();
          timeDisplay.innerText = `00:00 / ${{formatTime(dur)}}`;
        }});

        wavesurfer.on('audioprocess', (currentTime) => {{
          const dur = wavesurfer.getDuration();
          timeDisplay.innerText = `${{formatTime(currentTime)}} / ${{formatTime(dur)}}`;
        }});

        wavesurfer.on('seeking', (currentTime) => {{
          const dur = wavesurfer.getDuration();
          timeDisplay.innerText = `${{formatTime(currentTime)}} / ${{formatTime(dur)}}`;
        }});

        wavesurfer.on('play', () => {{
          playBtn.innerText = '⏸ Pause';
        }});

        wavesurfer.on('pause', () => {{
          playBtn.innerText = '▶ Play';
        }});

        wavesurfer.on('finish', () => {{
          playBtn.innerText = '▶ Play';
          wavesurfer.seekTo(0);
        }});

        playBtn.addEventListener('click', () => {{
          wavesurfer.playPause();
        }});
      </script>
    </body>
    </html>
    """
    st.components.v1.html(html_code, height=155, scrolling=False)
