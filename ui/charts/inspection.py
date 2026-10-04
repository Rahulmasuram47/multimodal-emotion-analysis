"""
Synchronized 3-tier acoustic feature inspection (Spectrogram, F0 Pitch, RMS Energy).
"""

import hashlib
import io
import textwrap
from typing import Any, Dict, List

import librosa
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import soundfile as sf
import streamlit as st

from ui.utils import safe_float  # noqa: F401
from ui.charts.theme import apply_chart_theme, get_chart_theme_tokens, show_chart

# html_block returns a cleaned HTML string (used as st.markdown(html_block(...), unsafe_allow_html=True)).
# Use the shared one from ui.utils if it exists, otherwise fall back to a local version.
try:
    from ui.utils import html_block
except ImportError:
    def html_block(html: str) -> str:
        """Dedent and strip so Markdown doesn't treat indented HTML as a code block."""
        return textwrap.dedent(html).strip()


@st.cache_data(show_spinner=False)
def extract_acoustic_features(audio_bytes: bytes, audio_hash: str) -> Dict[str, Any]:
    """Extract spectrogram, F0 pitch, and RMS energy contour curves."""
    try:
        y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != 16000:
            y = librosa.resample(y, orig_sr=sr, target_sr=16000)
            sr = 16000
    except Exception:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)

    hop_length = 512
    n_fft = 2048

    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    spectrogram_db = librosa.amplitude_to_db(stft, ref=np.max)
    times_spec = librosa.times_like(spectrogram_db, sr=sr, hop_length=hop_length)

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times_rms = librosa.times_like(rms, sr=sr, hop_length=hop_length)

    f0, voiced_flag, voiced_probs = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=hop_length,
    )
    times_f0 = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # MFCC extraction (13 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    mfcc_mean = [round(float(v), 3) for v in np.mean(mfcc, axis=1)]
    mfcc_std = [round(float(v), 3) for v in np.std(mfcc, axis=1)]

    return {
        "spectrogram_db": spectrogram_db.tolist(),
        "times_spec": times_spec.tolist(),
        "freqs": freqs.tolist(),
        "rms": rms.tolist(),
        "times_rms": times_rms.tolist(),
        "f0": [None if np.isnan(v) else float(v) for v in f0],
        "times_f0": times_f0.tolist(),
        "mfcc_mean": mfcc_mean,
        "mfcc_std": mfcc_std,
    }


def render_acoustic_inspection(
    audio_bytes: bytes,
    turn_boundaries: List[float] = None,
    duration: float = 0.0,
) -> None:
    """Render synchronized Plotly spectrogram, pitch, and RMS contours with turn boundary markers."""
    if not audio_bytes:
        st.info("No audio data available for acoustic inspection.")
        return

    audio_hash = hashlib.sha256(audio_bytes).hexdigest()[:16]

    with st.spinner("Extracting synchronized acoustic feature contours..."):
        features = extract_acoustic_features(audio_bytes, audio_hash)

    tokens = get_chart_theme_tokens()

    times_spec = np.array(features["times_spec"])
    freqs = np.array(features["freqs"])
    spectrogram_db = np.array(features["spectrogram_db"])
    times_f0 = np.array(features["times_f0"])
    f0 = features["f0"]
    times_rms = np.array(features["times_rms"])
    rms = np.array(features["rms"])

    freq_mask = freqs <= 4000
    sub_freqs = freqs[freq_mask]
    sub_spec = spectrogram_db[freq_mask, :]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Mel-Frequency Spectrogram (0 - 4 kHz)",
            "Fundamental Frequency F0 / Pitch (Hz)",
            "Root-Mean-Square (RMS) Energy Contour",
        ),
        row_heights=[0.5, 0.25, 0.25],
    )

    fig.add_trace(
        go.Heatmap(
            z=sub_spec,
            x=times_spec,
            y=sub_freqs,
            colorscale="Viridis",
            colorbar=dict(
                title=dict(text="dB", font=dict(family="IBM Plex Mono", size=10, color=tokens["font_color"])),
                tickfont=dict(family="IBM Plex Mono", size=9, color=tokens["font_color"]),
                len=0.45,
                y=0.78,
                thickness=12,
            ),
            hoverongaps=False,
            hovertemplate="Time: %{x:.2f}s<br>Freq: %{y:.0f} Hz<br>Power: %{z:.1f} dB<extra></extra>",
        ),
        row=1,
        col=1,
    )

    clean_times_f0 = [t for t, p in zip(times_f0, f0) if p is not None]
    clean_f0 = [p for p in f0 if p is not None]

    fig.add_trace(
        go.Scatter(
            x=clean_times_f0,
            y=clean_f0,
            mode="lines+markers",
            line=dict(color="#4FB3A9", width=2),
            marker=dict(size=3, color="#4FB3A9"),
            name="Pitch (F0)",
            hovertemplate="Time: %{x:.2f}s<br>Pitch: %{y:.1f} Hz<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=times_rms,
            y=rms,
            mode="lines",
            line=dict(color="#E8A33D", width=2),
            fill="tozeroy",
            fillcolor="rgba(232, 163, 61, 0.15)",
            name="RMS Energy",
            hovertemplate="Time: %{x:.2f}s<br>RMS: %{y:.4f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    if turn_boundaries:
        for t_sec in turn_boundaries:
            if t_sec > 0:
                for row_idx in [1, 2, 3]:
                    fig.add_vline(
                        x=t_sec,
                        line_width=1,
                        line_dash="dash",
                        line_color=tokens["vline_color"],  # visible in both themes
                        row=row_idx,
                        col=1,
                    )

    apply_chart_theme(fig, height=580, margin=dict(l=65, r=25, t=40, b=35), show_legend=False)

    fig.update_xaxes(title_text="Time (seconds)", row=3, col=1)
    fig.update_yaxes(title_text="Freq (Hz)", title_standoff=12, row=1, col=1)
    fig.update_yaxes(title_text="F0 (Hz)", title_standoff=12, row=2, col=1)
    fig.update_yaxes(title_text="RMS", title_standoff=12, row=3, col=1)
    show_chart(fig)

    # MFCC Summary (Prompt G / Item 27)
    mfcc_mean = features.get("mfcc_mean", [])
    mfcc_std = features.get("mfcc_std", [])
    if mfcc_mean and len(mfcc_mean) > 1:
        st.markdown(
            html_block("""
<div class="section-header-bar" style="margin-top: 1.2rem;">
<div class="section-title">📊 Mel-Frequency Cepstral Coefficients (MFCC 2–13)</div>
<div class="section-badge">TIMBRAL ACOUSTIC SUMMARY</div>
</div>
"""),
            unsafe_allow_html=True,
        )
        st.caption(
            "ℹ️ **About MFCCs**: Mel-Frequency Cepstral Coefficients (MFCCs 2–13) capture the spectral envelope and "
            "timbral shape of the speaker's vocal tract across frames. MFCC 1 (overall signal power) is excluded here "
            "to prevent scale dominance. These coefficients represent acoustic timbre and spectral distribution, "
            "not discrete emotional categories directly."
        )

        # Item 27: Plot MFCC 2–13 to avoid MFCC 1 energy dominance
        mfcc_labels = [f"MFCC {i+1}" for i in range(1, len(mfcc_mean))]
        mfcc_y = mfcc_mean[1:]
        mfcc_err = mfcc_std[1:]

        mfcc_fig = go.Figure()
        mfcc_fig.add_trace(
            go.Bar(
                x=mfcc_labels,
                y=mfcc_y,
                error_y=dict(type="data", array=mfcc_err, visible=True),
                marker_color=tokens["accent"],
                opacity=0.85,
                name="Mean ± Std across frames",
            )
        )
        apply_chart_theme(mfcc_fig, height=250, margin=dict(l=40, r=20, t=30, b=30), show_legend=False)
        mfcc_fig.update_layout(
            title=dict(
                text="Timbral Cepstral Envelope (MFCC 2–13, mean ± std across frames)",
                font=dict(color=tokens["title_color"], size=12),
                x=0,
            ),
        )
        mfcc_fig.update_yaxes(title="Coefficient Value (mean ± std across frames)")
        show_chart(mfcc_fig)