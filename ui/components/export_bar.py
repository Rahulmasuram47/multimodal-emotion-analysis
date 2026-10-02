"""
Report generation and export actions (JSON / CSV download buttons).
"""

import json
import re
from typing import Any, Dict
import pandas as pd
import streamlit as st

from ui.components.transparency_panels import SYSTEM_NOTICES_AND_LIMITATIONS
from ui.utils import emotion_code_label


def _clean_transcript_text(text: str, speaker: str = "") -> str:
    """Strip speaker prefix like 'Speaker A:' from transcript text for exports and analytics."""
    clean = str(text or "").strip()
    if speaker and speaker != "Unknown":
        clean = re.sub(rf"^{re.escape(speaker)}\s*:\s*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"^Speaker\s+[A-Za-z0-9]+\s*:\s*", "", clean, flags=re.IGNORECASE)
    return clean.strip()


def build_json_export_payload(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return JSON export dict additively enriched with system notices, limitations,
    and canonical emotion labels (item 9, item 13).
    """
    payload = dict(result)
    payload["notices_and_limitations"] = dict(SYSTEM_NOTICES_AND_LIMITATIONS)

    # Item 13: Clean transcripts; Item 9: canonical emotion labels
    cleaned_segments = []
    for s in payload.get("segments", []) or []:
        seg_copy = dict(s)
        spk = str(seg_copy.get("speaker", ""))
        seg_copy["transcript"] = _clean_transcript_text(seg_copy.get("transcript", ""), spk)

        # Map emotion codes
        if "voice_emotion" in seg_copy and isinstance(seg_copy["voice_emotion"], dict):
            ve = dict(seg_copy["voice_emotion"])
            raw_emo = ve.get("emotion", "")
            ve["emotion_raw"] = raw_emo
            ve["emotion"] = emotion_code_label(raw_emo)
            seg_copy["voice_emotion"] = ve

        if "text_sentiment" in seg_copy and isinstance(seg_copy["text_sentiment"], dict):
            ts = dict(seg_copy["text_sentiment"])
            raw_sent = ts.get("sentiment", "")
            ts["sentiment_raw"] = raw_sent
            ts["sentiment"] = emotion_code_label(raw_sent)
            seg_copy["text_sentiment"] = ts

        cleaned_segments.append(seg_copy)

    payload["segments"] = cleaned_segments
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
            spk = s.get("speaker", "")
            clean_text = _clean_transcript_text(s.get("transcript", ""), spk)
            rows.append(
                {
                    "segment": s.get("segment_index", ""),
                    "start_seconds": s.get("start_seconds", ""),
                    "end_seconds": s.get("end_seconds", ""),
                    "speaker": spk,
                    "transcript": clean_text,
                    "voice_emotion": emotion_code_label(v.get("emotion", "")),
                    "voice_confidence": v.get("confidence", ""),
                    "text_sentiment": emotion_code_label(t.get("sentiment", "")),
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
