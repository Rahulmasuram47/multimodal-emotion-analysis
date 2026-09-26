"""
Voice Emotion Analysis Module.

Uses Wav2Vec2 to predict emotion from speech audio.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

from app.audio_processor import load_and_preprocess_audio
import config

logger = logging.getLogger(__name__)


class VoiceEmotionAnalyzer:
    """Analyze speech emotion using a pretrained Wav2Vec2 model."""

    def __init__(self, model_id: Optional[str] = None):
        # Falls back to the locked-in production model from config.py when
        # no model_id is given, so existing callers (ConversationAnalyzer,
        # etc.) keep using the same model as before with zero code changes.
        # A caller (e.g. the Phase 3 benchmark script) can still pass a
        # different model_id to compare candidates without touching
        # production config.
        self.model_id = model_id or config.AUDIO_EMOTION_MODEL_NAME

        logger.info("Loading Wav2Vec2 emotion model: %s", self.model_id)

        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_id
        )

        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self.model_id
        )

        self.model.eval()

        logger.info("Wav2Vec2 emotion model loaded successfully.")

    def _analyze_audio_array(
        self,
        audio: np.ndarray,
        sr: int,
    ) -> Dict[str, Any]:
        """Run emotion inference on an already-loaded audio segment."""

        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if len(audio) == 0:
            raise ValueError("Audio segment is empty.")

        inputs = self.extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probabilities = torch.softmax(logits, dim=-1)[0]

        predicted_index = int(torch.argmax(probabilities))

        emotion = self.model.config.id2label[predicted_index]

        confidence = float(probabilities[predicted_index])

        scores = {
            self.model.config.id2label[i]: float(probabilities[i])
            for i in range(len(probabilities))
        }

        return {
            "emotion": emotion,
            "confidence": confidence,
            "scores": scores,
        }

    def analyze(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze emotion from a complete audio file.
        """

        audio, sr, duration = load_and_preprocess_audio(audio_path)

        result = self._analyze_audio_array(audio, sr)

        result["duration_seconds"] = duration
        result["sample_rate"] = sr

        logger.info(
            "Voice emotion prediction: %s "
            "(confidence=%.4f)",
            result["emotion"],
            result["confidence"],
        )

        return result

    def analyze_segment(
        self,
        audio: np.ndarray,
        sr: int,
        start_seconds: float,
        end_seconds: float,
    ) -> Dict[str, Any]:
        """
        Analyze emotion for a specific time segment.

        Args:
            audio: Preprocessed full audio array.
            sr: Sample rate.
            start_seconds: Segment start time.
            end_seconds: Segment end time.

        Returns:
            Emotion prediction for that segment.
        """

        if start_seconds < 0:
            raise ValueError("Segment start time cannot be negative.")

        if end_seconds <= start_seconds:
            raise ValueError(
                "Segment end time must be greater than start time."
            )

        start_sample = int(start_seconds * sr)
        end_sample = int(end_seconds * sr)

        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)

        segment_audio = audio[start_sample:end_sample]

        if len(segment_audio) == 0:
            raise ValueError(
                "Requested segment contains no audio samples."
            )

        result = self._analyze_audio_array(
            segment_audio,
            sr,
        )

        result["start_seconds"] = start_seconds
        result["end_seconds"] = end_seconds
        result["duration_seconds"] = (
            end_seconds - start_seconds
        )

        logger.info(
            "Segment emotion: %.2f-%.2fs â†’ %s "
            "(confidence=%.4f)",
            start_seconds,
            end_seconds,
            result["emotion"],
            result["confidence"],
        )

        return result