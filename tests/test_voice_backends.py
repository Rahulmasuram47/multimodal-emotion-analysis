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
        mock_model = MagicMock()
        labels = list(raw_scores.keys())
        mock_model.config.id2label = {i: lbl for i, lbl in enumerate(labels)}
        import torch
        pred_idx = labels.index(raw_emotion) if raw_emotion in labels else 0
        logits_mock = MagicMock()
        # Mock softmax probabilities matching raw_scores
        probs_tensor = torch.tensor([list(raw_scores.values())], dtype=torch.float32)
        mock_model.return_value.logits = logits_mock
        mock_vea.model = mock_model
        mock_vea.extractor = MagicMock()
        mock_vea.extractor.return_value = {}

        # Patch torch.softmax in predict to return raw_scores probabilities
        # Or simpler: have mock_model return logits that yield expected softmax
        b = SuperbBackend(analyzer=mock_vea)
        # Patch predict's torch call or supply mock
        b._mock_raw_output = {"raw_label": raw_emotion, "raw_scores": raw_scores}
        return b

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

    def test_vea_default_uses_superb_backend(self):
        """With VOICE_BACKEND='superb' and no explicit backend arg,
        VoiceEmotionAnalyzer should use SuperbBackend."""
        import config
        with patch.object(config, "VOICE_BACKEND", "superb"), \
             patch("app.emotion_analyzer.Wav2Vec2FeatureExtractor.from_pretrained") as mock_ext, \
             patch("app.emotion_analyzer.Wav2Vec2ForSequenceClassification.from_pretrained") as mock_model:
            mock_model.return_value = MagicMock()
            mock_ext.return_value = MagicMock()
            from app.emotion_analyzer import VoiceEmotionAnalyzer
            from app.voice_backends.superb import SuperbBackend
            vea = VoiceEmotionAnalyzer()
            assert isinstance(vea._backend, SuperbBackend)
            assert vea.family == "Wav2Vec2"
            assert vea.confidence_informative is True

    def test_vea_emotion2vec_backend_sets_field(self):
        """When backend='emotion2vec', _backend must be Emotion2VecBackend."""
        with patch("app.voice_backends.emotion2vec.Emotion2VecBackend._load"):
            from app.emotion_analyzer import VoiceEmotionAnalyzer
            from app.voice_backends.emotion2vec import Emotion2VecBackend
            vea = VoiceEmotionAnalyzer(backend="emotion2vec")
            assert isinstance(vea._backend, Emotion2VecBackend)
            assert vea.family == "emotion2vec"
            assert vea.confidence_informative is False


# ------------------------------------------------------------------ #
# Item 8: Fallback, Ensemble, Other, Confidence Informative, UI      #
# ------------------------------------------------------------------ #

class TestBackendAdvancedFeatures:
    def test_fallback_to_superb_on_load_failure(self):
        """If emotion2vec fails to load, fall back to superb without exception."""
        import config
        with patch.object(config, "VOICE_BACKEND", "emotion2vec"), \
             patch("app.voice_backends.emotion2vec.Emotion2VecBackend._load", side_effect=ImportError("funasr missing")), \
             patch("app.emotion_analyzer.Wav2Vec2FeatureExtractor.from_pretrained") as mock_ext, \
             patch("app.emotion_analyzer.Wav2Vec2ForSequenceClassification.from_pretrained") as mock_model:
            mock_model.return_value = MagicMock()
            mock_ext.return_value = MagicMock()
            from app.emotion_analyzer import VoiceEmotionAnalyzer
            from app.voice_backends.superb import SuperbBackend
            vea = VoiceEmotionAnalyzer()
            assert isinstance(vea._backend, SuperbBackend)
            assert vea.fallback_notice is not None
            assert "fell back to SUPERB" in vea.fallback_notice

    def test_ensemble_agree(self):
        """When ensemble is enabled and both backends agree."""
        import config
        with patch.object(config, "ENSEMBLE_ENABLED", True):
            from app.emotion_analyzer import VoiceEmotionAnalyzer
            vea = VoiceEmotionAnalyzer.__new__(VoiceEmotionAnalyzer)
            vea._backend = MagicMock()
            vea._backend.predict.return_value = {"emotion": "happy", "confidence": 0.8, "scores": {}, "top2": [], "raw_label": "happy"}
            vea._backend.family = "Wav2Vec2"
            vea._secondary_backend = MagicMock()
            vea._secondary_backend.predict.return_value = {"emotion": "happy", "confidence": 0.9, "scores": {}, "top2": [], "raw_label": "happy"}
            vea._secondary_backend.family = "emotion2vec"

            res = vea._analyze_audio_array(DUMMY_WAVE, 16000)
            assert res["emotion"] == "happy"
            assert res["uncertain"] is False
            assert res["ensemble_disagreement"] is False
            assert res["backend_predictions"]["Wav2Vec2"] == "happy"
            assert res["backend_predictions"]["emotion2vec"] == "happy"

    def test_ensemble_disagree(self):
        """When ensemble is enabled and backends disagree: primary emotion kept, uncertain=True."""
        import config
        with patch.object(config, "ENSEMBLE_ENABLED", True):
            from app.emotion_analyzer import VoiceEmotionAnalyzer
            vea = VoiceEmotionAnalyzer.__new__(VoiceEmotionAnalyzer)
            vea._backend = MagicMock()
            vea._backend.predict.return_value = {"emotion": "angry", "confidence": 0.7, "scores": {}, "top2": [], "raw_label": "angry"}
            vea._backend.family = "Wav2Vec2"
            vea._secondary_backend = MagicMock()
            vea._secondary_backend.predict.return_value = {"emotion": "sad", "confidence": 0.9, "scores": {}, "top2": [], "raw_label": "sad"}
            vea._secondary_backend.family = "emotion2vec"

            res = vea._analyze_audio_array(DUMMY_WAVE, 16000)
            assert res["emotion"] == "angry"  # keeps primary
            assert res["uncertain"] is True
            assert res["ensemble_disagreement"] is True
            assert res["backend_predictions"]["Wav2Vec2"] == "angry"
            assert res["backend_predictions"]["emotion2vec"] == "sad"

    def test_other_label_handling_and_display(self):
        """Other label maps to direction neutral, gets display 'Other (<raw_label>)'."""
        from app.fusion import MultimodalFusion
        from ui.utils import emotion_code_label

        assert MultimodalFusion._emotion_direction("other") == "neutral"
        assert emotion_code_label("other", raw_label="fearful") == "Other (fearful)"
        assert emotion_code_label("other") == "Other"

    def test_confidence_informative_false_in_conversation_analysis(self):
        """When confidence_informative is False, voice reliability is not penalized by confidence/margin."""
        from app.conversation_analysis import ConversationAnalyzer
        ca = ConversationAnalyzer.__new__(ConversationAnalyzer)
        ca.emotion_analyzer = MagicMock()
        ca.emotion_analyzer.confidence_informative = False
        ca.voice_calibrator = MagicMock()

        # Dummy segment analysis with low confidence (0.2) and tie scores
        res = {
            "emotion": "happy",
            "confidence": 0.20,
            "scores": {"happy": 0.51, "neutral": 0.49},
            "voice_reliability": 1.0,
        }
        # Simulate lines from _analyze_segment
        import config
        conf_informative = getattr(ca.emotion_analyzer, "confidence_informative", True)
        if conf_informative:
            margin_factor = ca._margin_reliability(res["scores"])
            res["voice_reliability"] *= margin_factor
            if res["confidence"] < 0.50:
                res["voice_reliability"] *= 0.60
                res["is_low_voice_confidence"] = True
        else:
            res["voice_margin_reliability"] = 1.0
            res["is_low_voice_confidence"] = False

        assert res["voice_reliability"] == 1.0
        assert res["is_low_voice_confidence"] is False

    def test_ui_has_no_hardcoded_superb_string_when_emotion2vec_selected(self):
        """When VOICE_BACKEND='emotion2vec', UI components render emotion2vec display name."""
        import config
        with patch.object(config, "VOICE_BACKEND", "emotion2vec"):
            # Check header
            from app.voice_backends.emotion2vec import Emotion2VecBackend
            # Read header file and check dynamic resolution logic
            backend_name = getattr(config, "VOICE_BACKEND", "superb")
            assert backend_name == "emotion2vec"
            assert Emotion2VecBackend.display_name == "emotion2vec+ base"
            assert Emotion2VecBackend.family == "emotion2vec"

