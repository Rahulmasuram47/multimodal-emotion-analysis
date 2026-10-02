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
        return st.session_state.get("theme", "dark").lower()
    except Exception:
        return "dark"


def get_chart_theme_tokens() -> Dict[str, Any]:
    """Return color and typography tokens tailored for Plotly figures."""
    mode = get_current_theme_mode()
    if mode == "light":
        return {
            "paper_bgcolor": "#F8FAFC",
            "plot_bgcolor": "#FFFFFF",
            "font_color": "#475569",
            "title_color": "#0F172A",
            "gridcolor": "rgba(0, 0, 0, 0.08)",
            "zerolinecolor": "rgba(0, 0, 0, 0.18)",
            "border_color": "rgba(0, 0, 0, 0.12)",
            "accent": "#0369A1",
            "voice": "#92400E",
            "positive": "#166534",
            "critical": "#B91C1C",
        }
    else:
        return {
            "paper_bgcolor": "#101318",
            "plot_bgcolor": "#181D24",
            "font_color": "#94A3B8",
            "title_color": "#F8FAFC",
            "gridcolor": "rgba(255, 255, 255, 0.08)",
            "zerolinecolor": "rgba(255, 255, 255, 0.18)",
            "border_color": "rgba(255, 255, 255, 0.10)",
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
    return fig
