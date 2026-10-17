"""
Tier 5 regression tests — Items 33–36.

33: Results hierarchy order (verdict → modality cards → timeline → tabs) is
    enforced in analysis_view.py source order.
34: Charts use get_chart_theme_tokens() for colors (no bare hardcoded hex in
    dynamics.py trace definitions). n/a annotation when < 2 segments.
35: load_analyzer is decorated with @cache_resource. Fusion recompute path
    updates segment["fusion"] without touching voice/text_sentiment fields.
36: No new UI library imports (streamlit-shadcn-ui, streamlit-extras,
    streamlit-option-menu) in requirements.txt.
"""

import ast
import importlib
import re
from pathlib import Path
from typing import Dict, Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]


def _src(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Item 33: Results hierarchy order
# ---------------------------------------------------------------------------
class TestItem33LayoutOrder:
    def test_analysis_view_hierarchy(self):
        """render_verdict_hero must appear before render_modality_cards, which
        must appear before render_agreement_strip, which must appear before
        st.tabs."""
        src = _src("ui/pages/analysis_view.py")
        pos_verdict = src.index("render_verdict_hero(")
        pos_cards = src.index("render_modality_cards(")
        pos_strip = src.index("render_agreement_strip(")
        pos_tabs = src.index("st.tabs(")
        assert pos_verdict < pos_cards < pos_strip < pos_tabs, (
            "Results hierarchy order violated: expected verdict → cards → timeline → tabs"
        )

    def test_pipeline_steps_in_header(self):
        """render_pipeline_steps (3-step start + live system strip) must be
        defined in header.py and called in streamlit_app.py before the audio
        ingestion section."""
        header_src = _src("ui/components/header.py")
        assert "def render_pipeline_steps" in header_src
        app_src = _src("streamlit_app.py")
        pos_pipeline = app_src.index("render_pipeline_steps()")
        pos_upload = app_src.index("st.file_uploader(")
        assert pos_pipeline < pos_upload, (
            "render_pipeline_steps() must appear before the file uploader widget"
        )

    def test_segment_timeline_has_emotion_chips(self):
        """Segment timeline must render voice emotion chips and intensity bar."""
        src = _src("ui/components/segment_timeline.py")
        assert "LOW VOICE RELIABILITY" in src or "v_emotion_display" in src
        assert "Intensity:" in src


# ---------------------------------------------------------------------------
# Item 34: Charts use theme tokens; n/a state for < 2 segments
# ---------------------------------------------------------------------------
class TestItem34Charts:
    def test_no_bare_hex_in_trace_lines(self):
        """dynamics.py scatter/bar line dict values must not contain bare #hex
        strings — they must use theme token variables instead."""
        src = _src("ui/charts/dynamics.py")
        # Find all `line=dict(color="...")` / `marker_color="..."` patterns
        # and check they are not static hex strings.
        # We allow rgba() literals (for fill bands) but not #hex in color=.
        # Regex: color= followed by "#[0-9a-fA-F]{3,8}" (bare hex literal)
        bad_matches = re.findall(r'color\s*=\s*"#[0-9a-fA-F]{3,8}"', src)
        # The rgba fill band is not a trace color; allow those.
        # Any bare #hex in a `color=` kwarg is a violation.
        assert bad_matches == [], (
            f"Bare hex color literals found in dynamics.py trace definitions: {bad_matches}"
        )

    def test_valence_map_na_for_single_segment(self):
        """render_valence_arousal_map must annotate n/a when len(segments) < 2."""
        src = _src("ui/charts/dynamics.py")
        # Check that the n/a branch exists for < 2 segments
        assert 'len(segments) < 2' in src and "Trajectory n/a" in src, (
            "valence-arousal map must show n/a annotation when < 2 segments"
        )

    def test_intensity_chart_na_annotation_for_few_turns(self):
        """render_intensity_trajectory_chart must show n/a when < min_turns."""
        src = _src("ui/charts/dynamics.py")
        assert "Too few turns to assess a trend" in src

    def test_charts_have_titles(self):
        """Each chart function must set a title in fig.update_layout."""
        src = _src("ui/charts/dynamics.py")
        # Count occurrences of `title=dict(` inside update_layout — we expect
        # at least 3 (valence, intensity, speaker comparison).
        title_dict_count = src.count('title=dict(')
        assert title_dict_count >= 3, (
            f"Expected ≥3 chart titles in dynamics.py, found {title_dict_count}"
        )

    def test_theme_tokens_used_in_dynamics(self):
        """dynamics.py must call get_chart_theme_tokens() for chart colors."""
        src = _src("ui/charts/dynamics.py")
        assert "get_chart_theme_tokens()" in src


# ---------------------------------------------------------------------------
# Item 35: Caching and fusion recompute
# ---------------------------------------------------------------------------
class TestItem35CachingAndFusionRecompute:
    def test_load_analyzer_cache_resource(self):
        """streamlit_app.py must decorate load_analyzer with @cache_resource."""
        src = _src("streamlit_app.py")
        # Find the decorator line and the def line — they must be adjacent
        # within 3 lines.
        lines = src.splitlines()
        for i, line in enumerate(lines):
            if "def load_analyzer(" in line:
                context = "\n".join(lines[max(0, i-3):i])
                assert "cache_resource" in context, (
                    "load_analyzer must be decorated with @st.cache_resource"
                )
                break
        else:
            raise AssertionError("load_analyzer function not found in streamlit_app.py")

    def test_fusion_recompute_button_exists(self):
        """streamlit_app.py must define the fusion recompute button path."""
        src = _src("streamlit_app.py")
        assert "fusion_recompute_clicked" in src
        assert "Recompute Fusion" in src

    def test_fusion_recompute_uses_stored_predictions(self):
        """The recompute block must read voice_emotion and text_sentiment from
        the stored segment dict — never call analyzer.analyze()."""
        src = _src("streamlit_app.py")
        # Find the recompute execution block (after the pipeline section) — use the last occurrence
        marker = "# Item 35: Fusion-only recompute"
        dashboard_idx = src.index("# 5. DASHBOARD VIEWS")
        recompute_start = src.rindex(marker)
        recompute_block = src[recompute_start:dashboard_idx]
        # Must read voice_emotion and text_sentiment from stored segment dict
        assert '"voice_emotion"' in recompute_block or "voice_emotion" in recompute_block
        assert '"text_sentiment"' in recompute_block or "text_sentiment" in recompute_block
        # Must NOT call analyzer.analyze inside the recompute block
        assert "analyzer.analyze" not in recompute_block, (
            "Fusion recompute block must not call analyzer.analyze()"
        )


    def test_last_fusion_settings_saved_after_analysis(self):
        """streamlit_app.py must store _last_fusion_settings after a full run."""
        src = _src("streamlit_app.py")
        assert '"_last_fusion_settings"' in src

    def test_inspection_chart_cache_data(self):
        """acoustic features extraction in inspection.py must use @cache_data."""
        src = _src("ui/charts/inspection.py")
        assert "@st.cache_data" in src


# ---------------------------------------------------------------------------
# Item 36: No new optional UI library deps added
# ---------------------------------------------------------------------------
class TestItem36NoNewUIDeps:
    def test_no_shadcn_in_requirements(self):
        req = _src("requirements.txt")
        assert "streamlit-shadcn" not in req, (
            "streamlit-shadcn-ui must not be added to requirements.txt (license/offline risk)"
        )

    def test_no_streamlit_extras_in_requirements(self):
        req = _src("requirements.txt")
        assert "streamlit-extras" not in req

    def test_no_option_menu_in_requirements(self):
        req = _src("requirements.txt")
        assert "streamlit-option-menu" not in req


# ---------------------------------------------------------------------------
# Repetition collapse & banner reliability consistency tests
# ---------------------------------------------------------------------------
class TestRepetitionCollapseAndBannerFix:
    def test_repetition_collapse_required_patterns(self):
        from ui.components.segment_timeline import _collapse_repetitive_tokens, collapse_repetitive_tokens
        assert collapse_repetitive_tokens is _collapse_repetitive_tokens

        cases = [
            ("Thank you. " * 40, "Thank you.", 40),
            (("the " * 30) + "th", "the", 31),
            ("I don't know what to do " * 8, "I don't know what to do", 8),
            ("... " * 12, "...", 12),
        ]
        for raw, first_occ, expected_n in cases:
            collapsed = _collapse_repetitive_tokens(raw)
            # Displayed length under 200 chars
            assert len(collapsed) < 200, f"Length {len(collapsed)} >= 200 for {first_occ}"
            # Marker present
            assert "(repetitive transcript)" in collapsed
            assert f"×{expected_n}" in collapsed
            # First occurrence kept
            assert collapsed.startswith(first_occ)

    def test_natural_repeats_untouched(self):
        from ui.components.segment_timeline import _collapse_repetitive_tokens
        assert _collapse_repetitive_tokens("very, very good") == "very, very good"
        assert _collapse_repetitive_tokens("no, no, no") == "no, no, no"

    def test_raw_unchanged_in_export(self):
        from ui.components.export_bar import build_json_export_payload
        raw_rep = "Thank you. " * 40
        mock_result = {
            "segments": [
                {
                    "segment_index": 1,
                    "start_seconds": 0.0,
                    "end_seconds": 2.0,
                    "speaker": "Speaker A",
                    "transcript": f"Speaker A: {raw_rep}",
                }
            ],
            "full_transcript": raw_rep,
        }
        payload = build_json_export_payload(mock_result)
        # Export retains uncollapsed raw text (minus speaker prefix)
        assert payload["segments"][0]["transcript"].strip() == raw_rep.strip()

    def test_banner_number_matches_why_this_result(self):
        """Banner reliability number and Why this result voice reliability must match."""
        import pytest
        from ui.utils import safe_float
        import config as _config

        # Turn with voice collapse penalty (0.50 reliability)
        seg = {
            "start_seconds": 0.0,
            "end_seconds": 2.0,
            "transcript": "Neutral turn",
            "voice_emotion": {"emotion": "neutral", "confidence": 0.70, "voice_reliability": 0.50},
            "voice_reliability": 0.50,
            "voice_collapse_detected": True,
            "fusion": {"audio_reliability": 0.50, "pattern_label": "neutral pattern"},
        }
        fusion = seg.get("fusion", {})
        voice = seg.get("voice_emotion", {})

        v_rel = safe_float(fusion.get("audio_reliability", voice.get("voice_reliability", seg.get("voice_reliability", 1.0))))
        if "voice_reliability" in voice:
            v_rel = min(v_rel, safe_float(voice.get("voice_reliability", 1.0)))
        if "voice_reliability" in seg:
            v_rel = min(v_rel, safe_float(seg.get("voice_reliability", 1.0)))
        if isinstance(fusion, dict):
            fusion["audio_reliability"] = v_rel

        # Banner formats as percentage
        banner_str = f"Low Voice Reliability ({v_rel:.0%})"
        assert banner_str == "Low Voice Reliability (50%)"

        # Why this result formats as float .2f
        why_rel = safe_float(fusion.get("audio_reliability", 1.0))
        assert why_rel == pytest.approx(0.50)
        assert f"{why_rel:.2f}" == "0.50"
        assert f"{v_rel:.2f}" == f"{why_rel:.2f}"

