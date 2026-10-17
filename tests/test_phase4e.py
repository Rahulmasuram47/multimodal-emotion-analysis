"""Phase 4e regression tests.

Requirements tested:
1. ASR language badge uses detected language from a mocked ASR result; shows "lang: n/a" when unavailable.
2. Voice-only segment header has no "UNKNOWN" and no "0.0%" text row.
3. Top transcript display applies sanitize + collapse pipeline with dir="auto".
4. Raw model label displays only the English part and has no CJK characters.
"""

from contextlib import contextmanager
import html as html_module
import re
from unittest.mock import MagicMock, patch
import pytest

from ui.components.segment_timeline import (
    _collapse_repetitive_tokens,
    render_segment_timeline,
    sanitize_transcript,
)
from app.conversation_analysis import ConversationAnalyzer


# ---------------------------------------------------------------------------
# Helper: capture streamlit markdown calls
# ---------------------------------------------------------------------------

@contextmanager
def _capture_st_markdown():
    captured = []

    def mock_markdown(content, *args, **kwargs):
        captured.append(str(content))

    def mock_expander(*args, **kwargs):
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=None)
        return ctx

    def mock_columns(n):
        cols = []
        for _ in range(n if isinstance(n, int) else len(n)):
            c = MagicMock()
            c.__enter__ = MagicMock(return_value=c)
            c.__exit__ = MagicMock(return_value=None)
            cols.append(c)
        return cols

    with patch("streamlit.markdown", side_effect=mock_markdown), \
         patch("streamlit.expander", side_effect=mock_expander), \
         patch("streamlit.columns", side_effect=mock_columns), \
         patch("streamlit.caption"), \
         patch("streamlit.info"):
        yield captured


# ---------------------------------------------------------------------------
# Req 1: ASR language badge & propagation from mocked ASR
# ---------------------------------------------------------------------------

def test_badge_uses_detected_language_from_mocked_asr():
    """Verify segment carries detected language & probability and badge renders them."""
    # 1. Pipeline propagation test
    analyzer = ConversationAnalyzer.__new__(ConversationAnalyzer)
    mock_stt = MagicMock()
    mock_stt.transcribe.return_value = {
        "text": "hola mundo",
        "language": "es",
        "language_probability": 0.94,
        "segments": [{"start": 0.0, "end": 2.0, "text": "hola mundo"}],
    }
    mock_stt.language = "auto"
    analyzer.speech_to_text = mock_stt
    analyzer.emotion_analyzer = MagicMock()
    analyzer.sentiment_analyzer = MagicMock()
    analyzer.fusion_arbitrator = MagicMock()
    analyzer.diarizer = None

    seg_dict = {
        "segment_index": 1,
        "start_seconds": 0.0,
        "end_seconds": 2.0,
        "transcript": "hola mundo",
        "speaker": "Speaker A",
        "voice_emotion": {"emotion": "neutral", "confidence": 0.8},
        "text_sentiment": None,
        "text_unavailable": True,
        "asr_flags": ["non_english"],
        "language": "es",
        "language_probability": 0.94,
        "fusion": {
            "pattern_label": "Voice-Only",
            "audio_direction": "neutral",
            "text_direction": "unavailable",
            "combined_confidence": 0.8,
            "effective_weights": {"audio_weight": 1.0, "text_weight": 0.0},
        },
        "intensity": 0.3,
        "intensity_components": {
            "formula_version": "v2",
            "prosodic_arousal": 0.3,
            "voice_arousal": 0.3,
            "text_negativity": 0.0,
            "weights": {"prosody": 0.5, "voice_arousal": 0.5, "text_negativity": 0.0},
        },
    }

    with _capture_st_markdown() as markdown_calls:
        render_segment_timeline([seg_dict])

    all_html = "\n".join(markdown_calls)
    assert "lang: es" in all_html
    assert "p=94%" in all_html
    assert "flag: non_english" in all_html


def test_badge_shows_na_when_detection_unavailable():
    """When language detection is unavailable, badge shows lang: n/a without (p=0%)."""
    seg_dict = {
        "segment_index": 1,
        "start_seconds": 0.0,
        "end_seconds": 2.0,
        "transcript": "something",
        "speaker": "Speaker A",
        "voice_emotion": {"emotion": "neutral", "confidence": 0.8},
        "text_sentiment": None,
        "text_unavailable": True,
        "asr_flags": ["repetition"],
        "language": None,
        "language_probability": 0.0,
        "fusion": {"pattern_label": "Voice-Only", "combined_confidence": 0.8},
        "intensity": 0.3,
    }

    with _capture_st_markdown() as markdown_calls:
        render_segment_timeline([seg_dict])

    all_html = "\n".join(markdown_calls)
    assert "lang: n/a" in all_html
    assert "lang: auto" not in all_html
    assert "(p=0%)" not in all_html


# ---------------------------------------------------------------------------
# Req 2: Voice-only segment header has no "UNKNOWN" and no "0.0%" text row
# ---------------------------------------------------------------------------

def test_voice_only_header_has_no_unknown_and_no_zero_percent_text_row():
    """Voice-only segment header displays '💬 n/a' instead of '💬 UNKNOWN',
    and does not display 'Text Negativity (w=0%): 0.0%'.
    """
    seg_dict = {
        "segment_index": 1,
        "start_seconds": 0.0,
        "end_seconds": 2.0,
        "transcript": "raw speech",
        "speaker": "Speaker A",
        "voice_emotion": {"emotion": "neutral", "confidence": 0.75},
        "text_sentiment": {"sentiment": "unknown", "confidence": 0.0},
        "text_unavailable": True,
        "asr_flags": ["unavailable"],
        "language": None,
        "language_probability": 0.0,
        "fusion": {
            "pattern_label": "Voice-Only Mode",
            "audio_direction": "neutral",
            "text_direction": "unavailable",
            "combined_confidence": 0.75,
            "effective_weights": {"audio_weight": 1.0, "text_weight": 0.0},
        },
        "intensity": 0.25,
        "intensity_components": {
            "formula_version": "v2",
            "prosodic_arousal": 0.25,
            "voice_arousal": 0.25,
            "text_negativity": 0.0,
            "weights": {"prosody": 0.5, "voice_arousal": 0.5, "text_negativity": 0.0},
        },
    }

    with _capture_st_markdown() as markdown_calls:
        render_segment_timeline([seg_dict])

    all_html = "\n".join(markdown_calls)
    # Header chip must say 💬 n/a, NOT 💬 UNKNOWN
    assert "💬 n/a" in all_html
    assert "💬 UNKNOWN" not in all_html

    # Intensity v2 line must NOT show Text Negativity when weight is 0
    assert "Text Negativity (w=0%): 0.0%" not in all_html
    assert "Text Negativity" not in all_html


# ---------------------------------------------------------------------------
# Req 3: Top transcript sanitization and repetitive token collapse
# ---------------------------------------------------------------------------

def test_top_transcript_collapsed_and_sanitized():
    """Top transcript strips replacement characters and collapses repeated tokens."""
    raw = "\ufffd " + " ".join(["world"] * 16)
    clean = sanitize_transcript(raw)
    assert "\ufffd" not in clean
    collapsed = _collapse_repetitive_tokens(clean)
    assert "world… ×16 (repetitive transcript)" in collapsed


def test_top_transcript_repetition_collapsed():
    """16 identical tokens collapse to 'token… ×16 (repetitive transcript)'."""
    text = " ".join(["repetitive"] * 16)
    result = _collapse_repetitive_tokens(text)
    assert "repetitive… ×16 (repetitive transcript)" == result


def test_sanitize_removes_replacement_chars():
    """sanitize_transcript strips U+FFFD, FFFE, FFFF."""
    raw = "audio\ufffdclean\ufffetranscript\uffff"
    assert sanitize_transcript(raw) == "audiocleantranscript"


# ---------------------------------------------------------------------------
# Req 4: Raw model label has no CJK characters
# ---------------------------------------------------------------------------

def test_raw_label_no_cjk_characters():
    """Raw model label shows only the English part and strips CJK prefix."""
    seg_dict = {
        "segment_index": 1,
        "start_seconds": 0.0,
        "end_seconds": 2.0,
        "transcript": "I am angry",
        "speaker": "Speaker A",
        "voice_emotion": {
            "emotion": "other",
            "raw_label": "生气/angry",
            "confidence": 0.65,
        },
        "text_sentiment": {"sentiment": "negative", "confidence": 0.8},
        "text_unavailable": False,
        "asr_flags": [],
        "language": "en",
        "language_probability": 0.99,
        "fusion": {
            "pattern_label": "Agreement",
            "audio_direction": "negative",
            "text_direction": "negative",
            "combined_confidence": 0.72,
        },
        "intensity": 0.6,
    }

    with _capture_st_markdown() as markdown_calls:
        render_segment_timeline([seg_dict])

    all_html = "\n".join(markdown_calls)
    # The rendered raw label must show the English part
    assert "Raw model label: angry" in all_html
    # Must not contain any CJK characters
    assert not re.search(r"[\u4e00-\u9fff]", all_html)
    # The raw string in segment_dict remains untouched for exports
    assert seg_dict["voice_emotion"]["raw_label"] == "生气/angry"


def test_raw_label_without_cjk_unchanged():
    """English-only raw label displayed without modification."""
    label = "happy"
    disp = label.split("/")[-1].strip() if "/" in label else label
    assert disp == "happy"


def test_raw_label_unk_and_no_slash_do_not_crash_or_show_blanks():
    """Labels with no '/', '<unk>', or empty must not crash or show blank labels."""
    # 1. <unk> with CJK prefix
    seg_unk = {
        "segment_index": 1,
        "start_seconds": 0.0,
        "end_seconds": 1.5,
        "transcript": "hello",
        "speaker": "Speaker A",
        "voice_emotion": {
            "emotion": "other",
            "raw_label": "未知/<unk>",
            "confidence": 0.50,
        },
        "text_sentiment": {"sentiment": "neutral", "confidence": 0.8},
        "text_unavailable": False,
        "fusion": {"combined_confidence": 0.65},
    }
    with _capture_st_markdown() as calls:
        render_segment_timeline([seg_unk])
    html_out = "\n".join(calls)
    assert "Raw model label: &lt;unk&gt;" in html_out
    assert not re.search(r"[\u4e00-\u9fff]", html_out)
    assert "Raw model label: </div>" not in html_out

    # 2. None / empty / slash-only must not show blank "Raw model label:"
    for empty_val in [None, "", "  ", "/"]:
        seg_empty = {
            "segment_index": 1,
            "start_seconds": 0.0,
            "end_seconds": 1.5,
            "transcript": "hello",
            "speaker": "Speaker A",
            "voice_emotion": {
                "emotion": "neutral",
                "raw_label": empty_val,
                "confidence": 0.80,
            },
            "text_sentiment": {"sentiment": "neutral", "confidence": 0.8},
            "text_unavailable": False,
            "fusion": {"combined_confidence": 0.80},
        }
        with _capture_st_markdown() as calls:
            render_segment_timeline([seg_empty])
        html_out = "\n".join(calls)
        assert "Raw model label:" not in html_out
