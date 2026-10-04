"""
Stylesheet loader for the Multimodal Emotion Analysis dashboard.
Loads tokens.css, layout.css, and components.css from ui/styles/ and injects into Streamlit.
Applies active theme token overrides (dark/light) dynamically based on st.session_state.
"""

from pathlib import Path
import streamlit as st

_STYLES_DIR = Path(__file__).parent / "styles"

_DARK = {
    "--bg": "#101318",
    "--surface": "#181D24",
    "--surface-raised": "#202630",
    "--border": "rgba(255, 255, 255, 0.10)",
    "--border-strong": "rgba(255, 255, 255, 0.18)",
    "--text": "#F8FAFC",
    "--text-muted": "#94A3B8",
    "--accent": "#38BDF8",
    "--voice": "#FBBF24",
    "--text-modality": "#38BDF8",
    "--fused": "#E2E8F0",
    "--positive": "#4ADE80",
    "--warning": "#FBBF24",
    "--critical": "#F87171",
    "--panel": "#181D24",
    "--panel-2": "#202630",
    "--panel-hover": "#262E3B",
    "--muted": "#94A3B8",
    "--muted-dark": "#64748B",
    "--negative": "#F87171",
    "--text-accent": "#38BDF8",
}

_LIGHT = {
    "--bg": "#F8FAFC",
    "--surface": "#FFFFFF",
    "--surface-raised": "#F1F5F9",
    "--border": "rgba(0, 0, 0, 0.12)",
    "--border-strong": "rgba(0, 0, 0, 0.22)",
    "--text": "#0F172A",
    "--text-muted": "#475569",
    "--accent": "#0369A1",
    "--voice": "#92400E",
    "--text-modality": "#0369A1",
    "--fused": "#334155",
    "--positive": "#166534",
    "--warning": "#92400E",
    "--critical": "#B91C1C",
    "--panel": "#FFFFFF",
    "--panel-2": "#F1F5F9",
    "--panel-hover": "#E2E8F0",
    "--muted": "#475569",
    "--muted-dark": "#64748B",
    "--negative": "#B91C1C",
    "--text-accent": "#0369A1",
}


def _theme_override(theme: str) -> str:
    """Build the CSS override block for the active theme."""
    is_light = theme == "light"
    palette = _LIGHT if is_light else _DARK
    variables = "\n".join(f"    {k}: {v} !important;" for k, v in palette.items())
    scheme = "light" if is_light else "dark"
    bg = palette["--bg"]
    text = palette["--text"]
    surface = palette["--surface"]
    border = palette["--border"]

    css = f"""
:root {{
{variables}
    color-scheme: {scheme};
}}
.stApp, .main, [data-testid="stAppViewContainer"] {{
    background-color: {bg} !important;
    color: {text} !important;
}}
[data-testid="stHeader"] {{
    background-color: {bg} !important;
}}
[data-testid="stSidebar"] {{
    background-color: {surface} !important;
    border-right: 1px solid {border} !important;
}}
"""
    if is_light:
        # components.css hardcodes dark audio styling; undo it in light mode
        css += """
div[data-testid="stAudio"],
.stAudio,
audio,
div[data-testid="stCustomComponentV1"],
div[data-testid="stCustomComponentV1"] iframe,
iframe {
    color-scheme: light !important;
}
audio {
    filter: none !important;
}
"""
    return css


def load_css() -> None:
    """Load all external stylesheets and inject them as a single <style> block."""
    css_parts = []
    for name in ("tokens.css", "layout.css", "components.css"):
        path = _STYLES_DIR / name
        if path.exists():
            css_parts.append(path.read_text(encoding="utf-8"))

    current_theme = st.session_state.get("theme", "dark")
    css_parts.append(_theme_override(current_theme))

    combined_css = "\n\n".join(css_parts)
    st.markdown(f"<style>\n{combined_css}\n</style>", unsafe_allow_html=True)