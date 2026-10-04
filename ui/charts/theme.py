"""
Shared Plotly Chart Theming Helper.

Provides a unified theme configuration that dynamically adapts to the active
interface theme (dark or light) stored in st.session_state.
"""

from typing import Any, Dict, Optional
import streamlit as st
import plotly.graph_objects as go


def get_current_theme_mode() -> str:
    """Detect active UI theme mode ('dark' or 'light')."""
    try:
        return str(st.session_state.get("theme", "dark")).lower()
    except Exception:
        return "dark"


def get_chart_theme_tokens() -> Dict[str, Any]:
    """Return color and typography tokens tailored for Plotly figures."""
    mode = get_current_theme_mode()
    if mode == "light":
        return {
            "paper_bgcolor": "#F8FAFC",
            "plot_bgcolor": "#FFFFFF",
            "font_color": "#334155",      # darker than before for readability
            "title_color": "#0F172A",
            "gridcolor": "rgba(0, 0, 0, 0.08)",
            "zerolinecolor": "rgba(0, 0, 0, 0.22)",
            "border_color": "rgba(0, 0, 0, 0.12)",
            "vline_color": "rgba(15, 23, 42, 0.35)",
            "hover_bg": "#FFFFFF",
            "accent": "#0369A1",
            "voice": "#92400E",
            "positive": "#166534",
            "critical": "#B91C1C",
        }
    return {
        "paper_bgcolor": "#101318",
        "plot_bgcolor": "#181D24",
        "font_color": "#94A3B8",
        "title_color": "#F8FAFC",
        "gridcolor": "rgba(255, 255, 255, 0.08)",
        "zerolinecolor": "rgba(255, 255, 255, 0.18)",
        "border_color": "rgba(255, 255, 255, 0.10)",
        "vline_color": "rgba(255, 255, 255, 0.25)",
        "hover_bg": "#202630",
        "accent": "#38BDF8",
        "voice": "#FBBF24",
        "positive": "#4ADE80",
        "critical": "#F87171",
    }


def apply_chart_theme(
    fig: go.Figure,
    height: Optional[int] = None,
    margin: Optional[Dict[str, int]] = None,
    show_legend: bool = False,
) -> go.Figure:
    """Apply the active theme to a Plotly figure."""
    tokens = get_chart_theme_tokens()

    layout_kwargs: Dict[str, Any] = {
        "paper_bgcolor": tokens["paper_bgcolor"],
        "plot_bgcolor": tokens["plot_bgcolor"],
        "font": dict(family="IBM Plex Mono, monospace", color=tokens["font_color"], size=11),
        "showlegend": show_legend,
        "legend": dict(font=dict(color=tokens["font_color"])),
        "hoverlabel": dict(
            bgcolor=tokens["hover_bg"],
            font=dict(color=tokens["title_color"], family="IBM Plex Mono, monospace"),
            bordercolor=tokens["border_color"],
        ),
    }
    if height is not None:
        layout_kwargs["height"] = height
    if margin is not None:
        layout_kwargs["margin"] = margin

    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(
        gridcolor=tokens["gridcolor"],
        zerolinecolor=tokens["zerolinecolor"],
        tickfont=dict(color=tokens["font_color"]),
        title_font=dict(color=tokens["font_color"]),
    )
    fig.update_yaxes(
        gridcolor=tokens["gridcolor"],
        zerolinecolor=tokens["zerolinecolor"],
        tickfont=dict(color=tokens["font_color"]),
        title_font=dict(color=tokens["font_color"]),
    )
    # Subplot titles are annotations; make them follow the theme too
    fig.update_annotations(font=dict(color=tokens["title_color"], size=12))
    return fig


def show_chart(fig: go.Figure, **kwargs) -> None:
    """Render a Plotly figure WITHOUT Streamlit's own theme overriding our colors."""
    st.plotly_chart(fig, use_container_width=True, theme=None, **kwargs)