"""
Custom WaveSurfer.js dark theme player with turn navigation markers.
"""

import base64
import json
from typing import Any, Dict, List
import streamlit as st

from ui.utils import safe_float


def render_wavesurfer_player(
    audio_bytes: bytes,
    audio_format: str,
    segments: List[Dict[str, Any]],
    duration: float,
) -> None:
    """Render dark mode WaveSurfer.js player iframe with interactive time scrubbing."""
    if not audio_bytes:
        st.info("No audio stream available for playback.")
        return

    b64_audio = base64.b64encode(audio_bytes).decode("utf-8")
    mime_type = audio_format or "audio/wav"

    markers = []
    for s in segments:
        start_sec = safe_float(s.get("start_seconds", 0.0))
        idx = s.get("segment_index", 0)
        voice = (s.get("voice_emotion") or {}).get("emotion", "")
        markers.append({"time": start_sec, "label": f"T{idx}: {voice}"})

    markers_json = json.dumps(markers)

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <meta name="color-scheme" content="dark">
      <script src="https://unpkg.com/wavesurfer.js@7"></script>
      <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
        html, body {{
          background-color: #181D24 !important;
          background: #181D24 !important;
          color: #F1F3F5;
          margin: 0;
          padding: 8px;
          border-radius: 8px;
          border: 1px solid rgba(255, 255, 255, 0.08);
          overflow: hidden;
          width: 100%;
          height: 100%;
          color-scheme: dark;
        }}
        #waveform {{ width: 100%; border-radius: 4px; background: #101318 !important; padding: 6px 0; }}
        .controls {{ display: flex; align-items: center; justify-content: space-between; margin-top: 10px; gap: 10px; background: transparent; }}
        .btn-group {{ display: flex; align-items: center; gap: 10px; background: transparent; }}
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
          background: transparent;
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
    <body style="background-color: #181D24; background: #181D24; color-scheme: dark;">
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
          waveColor: 'rgba(232, 163, 61, 0.45)',
          progressColor: '#E8A33D',
          cursorColor: '#4FB3A9',
          cursorWidth: 2,
          barWidth: 2,
          barGap: 1,
          barRadius: 2,
          height: 64,
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
    st.components.v1.html(html_code, height=135, scrolling=False)
