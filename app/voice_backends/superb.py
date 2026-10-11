"""SUPERB Wav2Vec2 voice-emotion backend.

Delegates all heavy lifting to the existing VoiceEmotionAnalyzer so model
weights are loaded only once regardless of which call-site created the
analyzer.
"""
from typing import Any, Dict

import numpy as np

from app.voice_backends.base import VoiceBackend


# SUPERB ER id2label: {0: 'neu', 1: 'hap', 2: 'ang', 3: 'sad'}  (abbrev.)
# Some checkpoint variants spell them out.  Map both to canonical names.
_SUPERB_MAP: Dict[str, str] = {
    "neu": "neutral",
    "neutral": "neutral",
    "hap": "happy",
    "happy": "happy",
    "ang": "angry",
    "angry": "angry",
    "sad": "sad",
    "exc": "happy",   # excited → happy
    "fru": "other",   # frustrated
    "dis": "other",
    "fea": "other",
    "sur": "other",
}


def _canonical(raw: str) -> str:
    return _SUPERB_MAP.get(raw.lower(), "other")


class SuperbBackend(VoiceBackend):
    """Thin wrapper around the existing Wav2Vec2FeatureExtractor/Model pair."""

    def __init__(self, analyzer=None, model_id: str | None = None):
        """Accept an already-initialised VoiceEmotionAnalyzer to avoid
        double-loading the model.  If *analyzer* is None one is created."""
        if analyzer is not None:
            self._vea = analyzer
        else:
            from app.emotion_analyzer import VoiceEmotionAnalyzer
            self._vea = VoiceEmotionAnalyzer(model_id=model_id)

    def predict(self, waveform: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        raw = self._vea._analyze_audio_array(waveform.astype(np.float32), sr)
        raw_label: str = raw["emotion"]
        emotion = _canonical(raw_label)

        # Re-key scores to canonical labels; accumulate when many raw labels
        # collapse to the same canonical bucket.
        canonical_scores: Dict[str, float] = {k: 0.0 for k in self.CANONICAL}
        for raw_key, prob in raw["scores"].items():
            canonical_scores[_canonical(raw_key)] += prob

        confidence = canonical_scores[emotion]
        top2 = self._top2(canonical_scores)

        return {
            "emotion": emotion,
            "confidence": confidence,
            "scores": canonical_scores,
            "top2": top2,
            "raw_label": raw_label,
        }
