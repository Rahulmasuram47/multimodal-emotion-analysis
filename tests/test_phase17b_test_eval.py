"""
Unit tests for Phase 17b: CREMA-D Test Evaluator active backend alignment.
Mocked tests; no models, network, or datasets loaded.
"""
from unittest.mock import MagicMock, patch
import pytest

from evaluation.crema_d import compute_cache_key
from evaluation.run_crema_d_test_evaluation import main


def test_dry_run_prints_backend_without_loading_model(monkeypatch, capsys):
    """Verify --dry-run prints active backend, model id, and counts without loading weights."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_crema_d_test_evaluation.py",
            "--dry-run",
            "--decision-file",
            "evaluation/results/head_decision.json",
            "--split-file",
            "evaluation/results/crema_d_split.json",
        ],
    )
    mock_backend_info = {
        "active_backend": "emotion2vec",
        "active_model_id": "emotion2vec/emotion2vec_plus_base",
        "active_reason": "Mocked emotion2vec test",
        "emotion2vec": {"available": True},
        "superb": {"available": True, "model_id": "superb/wav2vec2-base-superb-er"},
    }
    with patch("evaluation.run_crema_d_baseline.inspect_voice_backends", return_value=mock_backend_info), \
         patch("app.voice_backends.emotion2vec.Emotion2VecBackend") as mock_e2v:
        main()
        mock_e2v.assert_not_called()

    captured = capsys.readouterr().out
    assert "Active Backend: emotion2vec" in captured
    assert "Active Model ID: emotion2vec/emotion2vec_plus_base" in captured
    assert "Head Decision Status: not pursued" in captured


def test_non_emotion2vec_backend_raises(monkeypatch):
    """Verify that a non-emotion2vec active backend raises RuntimeError rather than falling back silently."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_crema_d_test_evaluation.py",
            "--dry-run",
            "--decision-file",
            "evaluation/results/head_decision.json",
            "--split-file",
            "evaluation/results/crema_d_split.json",
        ],
    )
    mock_backend_info = {
        "active_backend": "superb",
        "active_model_id": "superb/wav2vec2-base-superb-er",
        "active_reason": "Mocked fallback",
        "emotion2vec": {"available": False, "error": "funasr missing"},
        "superb": {"available": True},
    }
    with patch("evaluation.run_crema_d_baseline.inspect_voice_backends", return_value=mock_backend_info):
        with pytest.raises(RuntimeError, match="emotion2vec"):
            main()


def test_cache_keys_differ_by_model_id():
    """Verify that cache keys include model_id and differ between emotion2vec and superb."""
    key_e2v = compute_cache_key(
        model_id="emotion2vec/emotion2vec_plus_base",
        revision="main",
        target_sr=16000,
        clip_filename="1001_DFA_ANG_XX.wav",
        labels=["angry", "happy", "neutral", "sad"],
    )
    key_superb = compute_cache_key(
        model_id="superb/wav2vec2-base-superb-er",
        revision="main",
        target_sr=16000,
        clip_filename="1001_DFA_ANG_XX.wav",
        labels=["angry", "happy", "neutral", "sad"],
    )
    assert key_e2v != key_superb
    assert len(key_e2v) == 64
    assert len(key_superb) == 64
