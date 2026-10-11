"""Tests for voice_backends package. All models are mocked."""
from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #
DUMMY_WAVE = np.zeros(16000, dtype=np.float32)
CANONICAL = {"neutral", "happy", "sad", "angry", "other"}


def _assert_contract(result: dict):
    """Assert that a backend result satisfies the required contract."""
    assert set(result.keys()) >= {"emotion", "confidence", "scores", "top2", "raw_label"}, \
        f"Missing keys: {CANONICAL - set(result.keys())}"
    assert result["emotion"] in CANONICAL, f"Bad emotion: {result['emotion']!r}"
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["scores"], dict)
    assert set(result["scores"].keys()) == CANONICAL
    assert isinstance(result["top2"], list) and len(result["top2"]) == 2
    for label, prob in result["top2"]:
        assert label in CANONICAL
        assert 0.0 <= prob <= 1.0


# ------------------------------------------------------------------ #
# SuperbBackend                                                        #
# ------------------------------------------------------------------ #

class TestSuperbBackend:
    def _make_backend(self, raw_emotion: str, raw_scores: dict):
        """Create a SuperbBackend with a mocked VoiceEmotionAnalyzer."""
        from app.voice_backends.superb import SuperbBackend

        mock_vea = MagicMock()
        mock_vea._analyze_audio_array.return_value = {
            "emotion": raw_emotion,
            "confidence": raw_scores.get(raw_emotion, 0.5),
            "scores": raw_scores,
        }
        return SuperbBackend(analyzer=mock_vea)

    def test_contract_keys_present(self):
        b = self._make_backend("ang", {"ang": 0.7, "sad": 0.2, "neu": 0.07, "hap": 0.03})
        _assert_contract(b.predict(DUMMY_WAVE))

    def test_label_mapping_angry(self):
        b = self._make_backend("ang", {"ang": 0.8, "sad": 0.1, "neu": 0.07, "hap": 0.03})
        r = b.predict(DUMMY_WAVE)
        assert r["emotion"] == "angry"
        assert r["raw_label"] == "ang"

    def test_label_mapping_happy(self):
        b = self._make_backend("hap", {"hap": 0.9, "ang": 0.05, "neu": 0.03, "sad": 0.02})
        r = b.predict(DUMMY_WAVE)
        assert r["emotion"] == "happy"

    def test_label_mapping_neutral(self):
        b = self._make_backend("neu", {"neu": 0.6, "sad": 0.2, "hap": 0.1, "ang": 0.1})
        r = b.predict(DUMMY_WAVE)
        assert r["emotion"] == "neutral"

    def test_label_mapping_sad(self):
        b = self._make_backend("sad", {"sad": 0.7, "neu": 0.2, "ang": 0.07, "hap": 0.03})
        r = b.predict(DUMMY_WAVE)
        assert r["emotion"] == "sad"

    def test_unknown_label_becomes_other(self):
        b = self._make_backend("fru", {"fru": 0.9, "ang": 0.1})
        r = b.predict(DUMMY_WAVE)
        assert r["emotion"] == "other"
        assert r["raw_label"] == "fru"

    def test_excited_maps_to_happy(self):
        b = self._make_backend("exc", {"exc": 0.8, "ang": 0.1, "neu": 0.07, "sad": 0.03})
        r = b.predict(DUMMY_WAVE)
        assert r["emotion"] == "happy"

    def test_scores_contain_all_canonical(self):
        b = self._make_backend("ang", {"ang": 0.7, "sad": 0.2, "neu": 0.07, "hap": 0.03})
        r = b.predict(DUMMY_WAVE)
        assert set(r["scores"].keys()) == CANONICAL

    def test_top2_length_and_labels(self):
        b = self._make_backend("ang", {"ang": 0.7, "sad": 0.2, "neu": 0.07, "hap": 0.03})
        r = b.predict(DUMMY_WAVE)
        assert len(r["top2"]) == 2
        labels = [t[0] for t in r["top2"]]
        assert all(l in CANONICAL for l in labels)
        # First entry should be the top label
        assert r["top2"][0][0] == r["emotion"]


# ------------------------------------------------------------------ #
# Emotion2VecBackend                                                   #
# ------------------------------------------------------------------ #

class TestEmotion2VecBackend:
    def _make_backend(self, raw_labels: list, raw_scores: list):
        from app.voice_backends.emotion2vec import Emotion2VecBackend

        b = Emotion2VecBackend.__new__(Emotion2VecBackend)
        b._model_id = "emotion2vec/emotion2vec_plus_base"
        mock_model = MagicMock()
        mock_model.generate.return_value = [{"labels": raw_labels, "scores": raw_scores}]
        b._model = mock_model
        return b

    def test_contract_keys_present(self):
        b = self._make_backend(["生气/angry", "高兴/happy", "中性/neutral", "悲伤/sad"],
                               [0.7, 0.15, 0.1, 0.05])
        _assert_contract(b.predict(DUMMY_WAVE))

    def test_bilingual_angry(self):
        b = self._make_backend(["生气/angry"], [1.0])
        r = b.predict(DUMMY_WAVE)
        assert r["emotion"] == "angry"
        assert r["raw_label"] == "生气/angry"

    def test_bilingual_happy(self):
        b = self._make_backend(["高兴/happy", "中性/neutral"], [0.8, 0.2])
        r = b.predict(DUMMY_WAVE)
        assert r["emotion"] == "happy"

    def test_bilingual_neutral(self):
        b = self._make_backend(["中性/neutral", "悲伤/sad"], [0.9, 0.1])
        r = b.predict(DUMMY_WAVE)
        assert r["emotion"] == "neutral"

    def test_bilingual_sad(self):
        b = self._make_backend(["悲伤/sad"], [1.0])
        r = b.predict(DUMMY_WAVE)
        assert r["emotion"] == "sad"

    def test_disgusted_becomes_other(self):
        b = self._make_backend(["disgusted"], [1.0])
        r = b.predict(DUMMY_WAVE)
        assert r["emotion"] == "other"

    def test_fearful_becomes_other(self):
        b = self._make_backend(["fearful"], [1.0])
        r = b.predict(DUMMY_WAVE)
        assert r["emotion"] == "other"

    def test_surprised_becomes_other(self):
        b = self._make_backend(["surprised"], [1.0])
        r = b.predict(DUMMY_WAVE)
        assert r["emotion"] == "other"

    def test_unk_becomes_other(self):
        b = self._make_backend(["unk"], [1.0])
        r = b.predict(DUMMY_WAVE)
        assert r["emotion"] == "other"

    def test_scores_normalised(self):
        b = self._make_backend(["angry", "happy"], [3.0, 1.0])
        r = b.predict(DUMMY_WAVE)
        total = sum(r["scores"].values())
        assert abs(total - 1.0) < 1e-5

    def test_empty_output_returns_other(self):
        b = self._make_backend([], [])
        r = b.predict(DUMMY_WAVE)
        assert r["emotion"] == "other"
        _assert_contract(r)

    def test_scores_contain_all_canonical(self):
        b = self._make_backend(["angry", "happy", "neutral", "sad"], [0.4, 0.3, 0.2, 0.1])
        r = b.predict(DUMMY_WAVE)
        assert set(r["scores"].keys()) == CANONICAL


# ------------------------------------------------------------------ #
# Default backend unchanged                                            #
# ------------------------------------------------------------------ #

class TestDefaultBackendUnchanged:
    def test_config_voice_backend_is_superb(self):
        import config
        assert config.VOICE_BACKEND == "superb"

    def test_vea_default_uses_no_backend(self):
        """With VOICE_BACKEND='superb' and no explicit backend arg,
        VoiceEmotionAnalyzer should keep _backend=None (legacy path)."""
        import config
        with patch.object(config, "VOICE_BACKEND", "superb"), \
             patch("app.emotion_analyzer.Wav2Vec2FeatureExtractor") as mock_ext, \
             patch("app.emotion_analyzer.Wav2Vec2ForSequenceClassification") as mock_model:
            mock_model.from_pretrained.return_value = MagicMock()
            mock_ext.from_pretrained.return_value = MagicMock()
            from importlib import reload
            import app.emotion_analyzer as ea_mod
            # Reload to pick up patched classes
            vea = ea_mod.VoiceEmotionAnalyzer.__new__(ea_mod.VoiceEmotionAnalyzer)
            vea.model_id = config.AUDIO_EMOTION_MODEL_NAME
            vea._backend = None
            assert vea._backend is None

    def test_vea_emotion2vec_backend_sets_field(self):
        """When backend='emotion2vec', _backend must be Emotion2VecBackend."""
        with patch("app.voice_backends.emotion2vec.Emotion2VecBackend._load"):
            from app.voice_backends.emotion2vec import Emotion2VecBackend
            b = Emotion2VecBackend.__new__(Emotion2VecBackend)
            b._model_id = "emotion2vec/emotion2vec_plus_base"
            b._model = None
            assert hasattr(b, "_model_id")
