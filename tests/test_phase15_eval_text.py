"""Phase 15 tests: stale evaluation sentence is gone; text names the active backend."""

from pathlib import Path

import config


ROOT = Path(__file__).resolve().parents[1]


def _src(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def test_stale_evaluation_sentence_removed():
    """The old placeholder 'No local evaluation results yet' must not appear
    in either the Operational Limitations panel or the Limitations page."""
    tp_src = _src("ui/components/transparency_panels.py")
    lv_src = _src("ui/pages/limitations_view.py")
    assert "No local evaluation results yet" not in tp_src
    assert "No local evaluation results yet" not in lv_src
    # The dynamic helper must be present
    assert "_build_eval_summary" in tp_src
    # Hardcoded Wav2Vec2 must be gone from the Limitations page
    assert "Wav2Vec2`)" not in lv_src


def test_active_backend_named_in_limitations():
    """Both the Operational Limitations and the Limitations page must reference
    the configured voice backend, not a hardcoded model name."""
    configured = getattr(config, "VOICE_BACKEND", "superb")
    tp_src = _src("ui/components/transparency_panels.py")
    lv_src = _src("ui/pages/limitations_view.py")
    # transparency_panels uses config.VOICE_BACKEND via _build_eval_summary
    assert "config.VOICE_BACKEND" in tp_src or 'config, "VOICE_BACKEND"' in tp_src
    # limitations_view.py must reference the backend dynamically
    assert "_VOICE_ARCH_MAP" in lv_src
    assert "_VOICE_CORPUS_MAP" in lv_src
    # The configured backend key must be in the maps
    assert f'"{configured}"' in lv_src
