"""
Stylesheet loader for the Multimodal Emotion Analysis dashboard.
Loads tokens.css, layout.css, and components.css from ui/styles/ and injects into Streamlit.
Applies active theme token overrides (dark/light) dynamically based on st.session_state.
"""

from pathlib import Path
import streamlit as st

_STYLES_DIR = Path(__file__).parent / "styles"


def load_css() -> None:
    """Load all external stylesheets and inject them as a single <style> block."""
    tokens_path = _STYLES_DIR / "tokens.css"
    layout_path = _STYLES_DIR / "layout.css"
    components_path = _STYLES_DIR / "components.css"

    css_parts = []
    for path in (tokens_path, layout_path, components_path):
        if path.exists():
            css_parts.append(path.read_text(encoding="utf-8"))

    current_theme = st.session_state.get("theme", "dark")
    if current_theme == "light":
        light_override = """
        :root {
            --bg: #F8FAFC !important;
            --surface: #FFFFFF !important;
            --surface-raised: #F1F5F9 !important;
            --border: rgba(0, 0, 0, 0.12) !important;
            --border-strong: rgba(0, 0, 0, 0.22) !important;
            --text: #0F172A !important;
            --text-muted: #475569 !important;
            --accent: #0369A1 !important;
            --voice: #92400E !important;
            --text-modality: #0369A1 !important;
            --fused: #334155 !important;
            --positive: #166534 !important;
            --warning: #92400E !important;
            --critical: #B91C1C !important;
            --panel: #FFFFFF !important;
            --panel-2: #F1F5F9 !important;
            --panel-hover: #E2E8F0 !important;
            --muted: #475569 !important;
            --muted-dark: #64748B !important;
            --negative: #B91C1C !important;
            --text-accent: #0369A1 !important;
        }
        .stApp, .main, [data-testid="stAppViewContainer"] {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
        }
        [data-testid="stSidebar"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid rgba(0, 0, 0, 0.12) !important;
        }
        """
        css_parts.append(light_override)
    else:
        dark_override = """
        :root {
            --bg: #101318 !important;
            --surface: #181D24 !important;
            --surface-raised: #202630 !important;
            --border: rgba(255, 255, 255, 0.10) !important;
            --border-strong: rgba(255, 255, 255, 0.18) !important;
            --text: #F8FAFC !important;
            --text-muted: #94A3B8 !important;
            --accent: #38BDF8 !important;
            --voice: #FBBF24 !important;
            --text-modality: #38BDF8 !important;
            --fused: #E2E8F0 !important;
            --positive: #4ADE80 !important;
            --warning: #FBBF24 !important;
            --critical: #F87171 !important;
            --panel: #181D24 !important;
            --panel-2: #202630 !important;
            --panel-hover: #262E3B !important;
            --muted: #94A3B8 !important;
            --muted-dark: #64748B !important;
            --negative: #F87171 !important;
            --text-accent: #38BDF8 !important;
        }
        .stApp, .main, [data-testid="stAppViewContainer"] {
            background-color: #101318 !important;
            color: #F8FAFC !important;
        }
        [data-testid="stSidebar"] {
            background-color: #181D24 !important;
            border-right: 1px solid rgba(255, 255, 255, 0.10) !important;
        }
        """
        css_parts.append(dark_override)

    combined_css = "\n\n".join(css_parts)
    st.markdown(f"<style>\n{combined_css}\n</style>", unsafe_allow_html=True)
