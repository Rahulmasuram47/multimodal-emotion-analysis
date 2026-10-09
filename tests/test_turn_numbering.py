from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DISPLAY_FILES = [
    "ui/components/segment_timeline.py",
    "ui/components/waveform_player.py",
    "ui/charts/dynamics.py",
]


def test_turn_labels_do_not_depend_on_segment_index():
    """segment_index is 1-based (set in conversation_analysis.py), so display
    code must number turns by list position instead of re-reading it."""
    for rel in DISPLAY_FILES:
        src = (ROOT / rel).read_text(encoding="utf-8")
        assert 'get("segment_index"' not in src, f"{rel} still reads segment_index"