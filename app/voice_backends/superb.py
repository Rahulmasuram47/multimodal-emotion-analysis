"""SUPERB Wav2Vec2 voice-emotion backend.

Delegates all heavy lifting to the existing VoiceEmotionAnalyzer so model
weights are loaded only once regardless of which call-site created the
analyzer.
"""
from typing import Any, Dict

import numpy as np

import config
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

    display_name: str = "Wav2Vec2-SUPERB ER"
    family: str = "Wav2Vec2"
    training_note: "IEMOCAP Affective DB" = "IEMOCAP Affective DB"
    confidence_informative: bool = True

    def __init__(self, analyzer=None, model_id: str | None = None):
        """Accept an already-initialised VoiceEmotionAnalyzer to avoid
        double-loading the model.  If *analyzer* is None one is created."""
        self.model_id = model_id or getattr(config, "AUDIO_EMOTION_MODEL_NAME", "superb/wav2vec2-base-superb-er")
        if analyzer is not None:
            self._vea = analyzer
        else:
            from app.emotion_analyzer import VoiceEmotionAnalyzer
            self._vea = VoiceEmotionAnalyzer(model_id=self.model_id)

    def predict(self, waveform: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        if hasattr(self, "_mock_raw_output"):
            raw_label = self._mock_raw_output["raw_label"]
            raw_scores = self._mock_raw_output["raw_scores"]
            emotion = _canonical(raw_label)
            canonical_scores: Dict[str, float] = {k: 0.0 for k in self.CANONICAL}
            for raw_key, prob in raw_scores.items():
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

        import torch

        audio = np.asarray(waveform, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if len(audio) == 0:
            raise ValueError("Audio segment is empty.")

        inputs = self._vea.extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            logits = self._vea.model(**inputs).logits

        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_index = int(torch.argmax(probabilities))
        raw_label = self._vea.model.config.id2label[predicted_index]

        raw_scores = {
            self._vea.model.config.id2label[i]: float(probabilities[i])
            for i in range(len(probabilities))
        }

        emotion = _canonical(raw_label)

        # Re-key scores to canonical labels; accumulate when many raw labels
        # collapse to the same canonical bucket.
        canonical_scores: Dict[str, float] = {k: 0.0 for k in self.CANONICAL}
        for raw_key, prob in raw_scores.items():
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
