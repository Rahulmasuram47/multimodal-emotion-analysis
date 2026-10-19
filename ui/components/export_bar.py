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

    # Phase 12: Voice backend provenance
    from ui.utils import get_voice_backend_info
    vb_info = get_voice_backend_info(result)
    payload["voice_backend_used"] = vb_info["backend_used"]
    payload["model_id"] = vb_info["model_id"]
    payload["fallback_reason"] = vb_info["fallback_reason"]
    payload["voice_confidence_informative"] = vb_info.get("confidence_informative", True)
    payload["voice_confidence_capped"] = any(
        bool((s.get("voice_emotion") or {}).get("confidence_capped", False))
        for s in (payload.get("segments") or [])
    )

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
            ve["raw_confidence"] = ve.get("raw_confidence")
            ve["voice_confidence_capped"] = bool(ve.get("confidence_capped", False))
            ve_leaning = ve.get("leaning") or ve.get("raw_label")
            ve["emotion"] = emotion_code_label(raw_emo, raw_label=ve_leaning)
            seg_copy["voice_emotion"] = ve

        if "text_sentiment" in seg_copy and isinstance(seg_copy["text_sentiment"], dict):
            ts = dict(seg_copy["text_sentiment"])
            raw_sent = ts.get("sentiment", "")
            ts["sentiment_raw"] = raw_sent
            ts["sentiment"] = emotion_code_label(raw_sent)
            seg_copy["text_sentiment"] = ts

        # None-safe ASR fields on segment
        seg_copy["asr_flags"] = seg_copy.get("asr_flags") or payload.get("asr_flags") or []
        seg_copy["language"] = payload.get("language")
        seg_copy["language_probability"] = payload.get("language_probability")
        seg_copy["acoustic_descriptors"] = s.get("acoustic_descriptors", {})

        cleaned_segments.append(seg_copy)
    payload["segments"] = cleaned_segments
    payload["language"] = payload.get("language")
    payload["language_probability"] = payload.get("language_probability")
    payload["asr_flags"] = payload.get("asr_flags") or []
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
        from ui.utils import get_voice_backend_info
        vb_info = get_voice_backend_info(result)
        rows = []
        for s in result.get("segments", []) or []:
            v = s.get("voice_emotion", {}) or {}
            t = s.get("text_sentiment", {}) or {}
            f = s.get("fusion", {}) or {}
            ad = s.get("acoustic_descriptors", {}) or {}
            spk = s.get("speaker", "")
            clean_text = _clean_transcript_text(s.get("transcript", ""), spk)
            lang = result.get("language")
            lang_prob = result.get("language_probability")
            asr_flags_list = s.get("asr_flags") or result.get("asr_flags") or []
            asr_flags_str = ", ".join(asr_flags_list) if asr_flags_list else ""
            v_leaning = v.get("leaning") or v.get("raw_label")
            rows.append(
                {
                    "segment": s.get("segment_index", ""),
                    "start_seconds": s.get("start_seconds", ""),
                    "end_seconds": s.get("end_seconds", ""),
                    "speaker": spk,
                    "transcript": clean_text,
                    "language": lang,
                    "language_probability": lang_prob,
                    "asr_flags": asr_flags_str,
                    "voice_backend_used": vb_info["backend_used"],
                    "model_id": vb_info["model_id"],
                    "fallback_reason": vb_info["fallback_reason"] or "",
                    "voice_confidence_informative": vb_info.get("confidence_informative", True),
                    "voice_emotion": emotion_code_label(v.get("emotion", ""), raw_label=v_leaning),
                    "voice_confidence": v.get("confidence", ""),
                    "raw_confidence": v.get("raw_confidence", ""),
                    "text_sentiment": emotion_code_label(t.get("sentiment", "")) if t and t.get("sentiment") else "N/A",
                    "text_confidence": t.get("confidence", "") if t and t.get("confidence") is not None else "",
                    "fused_confidence": f.get("combined_confidence", ""),
                    "agreement": f.get("agreement", False),
                    "mismatch": f.get("mismatch", False),
                    "intensity": s.get("intensity", ""),
                    "pitch_trend": ad.get("pitch_trend", "n/a"),
                    "pitch_range_st": ad.get("pitch_range_st", ""),
                    "energy_level": ad.get("energy_level", ""),
                    "energy_dbfs": ad.get("energy_dbfs", ""),
                    "speaking_rate_wps": ad.get("speaking_rate_wps", ""),
                    "rate_band": ad.get("rate_band", "n/a"),
                    "pause_ratio": ad.get("pause_ratio", ""),
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
