"""
Stylesheet loader for the Multimodal Emotion Analysis dashboard.
Loads tokens.css, layout.css, and components.css from ui/styles/ and injects into Streamlit.
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

    combined_css = "\n\n".join(css_parts)
    st.markdown(f"<style>\n{combined_css}\n</style>", unsafe_allow_html=True)
