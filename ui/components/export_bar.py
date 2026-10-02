"""
Report generation and export actions (JSON / CSV download buttons).
"""

import json
from typing import Any, Dict
import pandas as pd
import streamlit as st

from ui.components.transparency_panels import SYSTEM_NOTICES_AND_LIMITATIONS


def build_json_export_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return JSON export dict additively enriched with system notices and limitations."""
    payload = dict(result)
    payload["notices_and_limitations"] = dict(SYSTEM_NOTICES_AND_LIMITATIONS)
    return payload


def render_export_bar(result: Dict[str, Any]) -> None:
    """Render JSON and CSV report export download buttons."""
    if not result:
        return

    export_col1, export_col2, export_col3 = st.columns([1, 1, 2])
    with export_col1:
        export_payload = build_json_export_payload(result)
        st.download_button(
            "DOWNLOAD JSON REPORT",
            data=json.dumps(export_payload, indent=2, default=str),
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
