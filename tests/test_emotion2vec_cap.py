import pytest
import numpy as np

import config
from app.voice_backends.emotion2vec import Emotion2VecBackend
from app.voice_backends.superb import SuperbBackend
from app.fusion import MultimodalFusion
from ui.utils import emotion_code_label, emotion_icon
from ui.components.export_bar import build_json_export_payload
from ui.components.modality_cards import render_channel_meters_html


class DummyModel:
    def __init__(self, raw_labels, raw_scores):
        self.raw_labels = raw_labels
        self.raw_scores = raw_scores

    def generate(self, waveform, granularity="utterance", extract_embedding=False):
        return [{"labels": self.raw_labels, "scores": self.raw_scores}]


def test_emotion2vec_conf_cap_applied():
    backend = Emotion2VecBackend()
    backend._model = DummyModel(
        raw_labels=["生气/angry", "开心/happy"],
        raw_scores=[0.998, 0.002],
    )
    res = backend.predict(np.zeros(16000, dtype=np.float32))

    assert res["emotion"] == "angry"
    assert res["raw_confidence"] == pytest.approx(0.998, rel=1e-3)
    assert res["confidence"] <= config.VOICE_CONF_CAP
    assert res["confidence"] == pytest.approx(0.65, rel=1e-3)


def test_superb_unchanged_no_cap():
    class DummyAnalyzer:
        pass

    backend = SuperbBackend(analyzer=DummyAnalyzer())
    backend._mock_raw_output = {
        "raw_label": "ang",
        "raw_scores": {"ang": 0.95, "neu": 0.05},
    }
    res = backend.predict(np.zeros(16000, dtype=np.float32))
    assert res["emotion"] == "angry"
    assert res["confidence"] == pytest.approx(0.95, rel=1e-3)
    assert res["raw_confidence"] == pytest.approx(0.95, rel=1e-3)


def test_emotion2vec_other_maps_to_uncertain_with_leaning():
    # Test 'disgusted' from other-set
    backend = Emotion2VecBackend()
    backend._model = DummyModel(
        raw_labels=["厌恶/disgusted", "生气/angry"],
        raw_scores=[0.90, 0.10],
    )
    res = backend.predict(np.zeros(16000, dtype=np.float32))

    assert res["emotion"] == "uncertain"
    assert res["leaning"] == "disgusted"
    assert res["confidence"] <= 0.30
    assert res["raw_confidence"] == pytest.approx(0.90, rel=1e-3)

    # Test 'surprised'
    backend._model = DummyModel(
        raw_labels=["震惊/surprised", "开心/happy"],
        raw_scores=[0.85, 0.15],
    )
    res_sur = backend.predict(np.zeros(16000, dtype=np.float32))
    assert res_sur["emotion"] == "uncertain"
    assert res_sur["leaning"] == "surprised"
    assert res_sur["confidence"] <= 0.30


def test_uncertain_fusion_no_mismatch_and_neutral_direction():
    fusion = MultimodalFusion()
    audio_res = {
        "emotion": "uncertain",
        "confidence": 0.30,
        "voice_reliability": 0.20,
    }
    # Even if text has very high positive or negative confidence, no mismatch should be raised
    text_res = {
        "sentiment": "positive",
        "confidence": 0.95,
        "text_reliability": 1.0,
    }
    fused = fusion.fuse(audio_res, text_res)

    assert fused["audio_direction"] == "neutral"
    assert fused["mismatch"] is False
    assert fused["audio_reliability"] == 0.20


def test_ui_label_and_icon_handling():
    # Label formatting
    assert emotion_code_label("uncertain", raw_label="sad") == "Uncertain (leaning Sad)"
    assert emotion_code_label("uncertain", raw_label="难过/sad") == "Uncertain (leaning Sad)"
    assert emotion_code_label("uncertain") == "Uncertain"
    assert emotion_code_label("other", raw_label="disgusted") == "Other (disgusted)"
    assert emotion_code_label("other") == "Other"

    # Neutral icon returned for uncertain and other
    assert emotion_icon("uncertain") == "😐"
    assert emotion_icon("other") == "😐"


def test_export_payload_includes_raw_confidence():
    res = {
        "segments": [
            {
                "segment_index": 1,
                "start_seconds": 0.0,
                "end_seconds": 2.0,
                "transcript": "Test segment",
                "voice_emotion": {
                    "emotion": "uncertain",
                    "leaning": "fearful",
                    "confidence": 0.25,
                    "raw_confidence": 0.92,
                },
                "text_sentiment": {
                    "sentiment": "neutral",
                    "confidence": 0.80,
                },
                "fusion": {
                    "combined_confidence": 0.50,
                    "agreement": False,
                    "mismatch": False,
                },
            }
        ]
    }
    payload = build_json_export_payload(res)
    seg = payload["segments"][0]
    assert seg["voice_emotion"]["emotion"] == "Uncertain (leaning Fearful)"
    assert seg["voice_emotion"]["raw_confidence"] == 0.92
