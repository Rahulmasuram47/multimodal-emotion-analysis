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
    """Analyze speech emotion using a pretrained Wav2Vec2 or pluggable backend model."""

    def __init__(self, model_id: Optional[str] = None, backend: Optional[str] = None):
        self.model_id = model_id or config.AUDIO_EMOTION_MODEL_NAME
        self.fallback_notice: Optional[str] = None
        self._secondary_backend = None

        self.configured_backend = backend or getattr(config, "VOICE_BACKEND", "superb")
        self.fallback_active: bool = False
        self.fallback_reason: Optional[str] = None
        # Initialize backend_used early so cached pre-Phase-12 instances are safe
        self.backend_used: str = "superb"
        backend_name = self.configured_backend

        if backend_name == "emotion2vec":
            try:
                from app.voice_backends.emotion2vec import Emotion2VecBackend
                b_instance = Emotion2VecBackend(model_id=model_id)
                # Verify loadability to catch missing funasr / network errors early
                b_instance._load()
                self._backend = b_instance
                self.model = None
                self.extractor = None
                self.backend_used = "emotion2vec"
            except Exception as e:
                logger.warning("Failed to load emotion2vec backend (%s); falling back to superb.", e)
                self.fallback_notice = f"Failed to load emotion2vec backend ({e}); fell back to SUPERB."
                self.fallback_active = True
                self.fallback_reason = str(e)
                backend_name = "superb"
                self._backend = None

        if backend_name != "emotion2vec":
            logger.info("Loading Wav2Vec2 emotion model: %s", self.model_id)
            self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_id)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_id)
            self.model.eval()
            logger.info("Wav2Vec2 emotion model loaded successfully.")
            from app.voice_backends.superb import SuperbBackend
            self._backend = SuperbBackend(analyzer=self)
            self.backend_used = "superb"

        # Ensemble initialization if enabled
        if getattr(config, "ENSEMBLE_ENABLED", False):
            try:
                if self.backend_name == "superb":
                    from app.voice_backends.emotion2vec import Emotion2VecBackend
                    sec = Emotion2VecBackend()
                    sec._load()
                    self._secondary_backend = sec
                else:
                    from app.voice_backends.superb import SuperbBackend
                    self._secondary_backend = SuperbBackend()
            except Exception as e:
                logger.warning("Failed to initialize secondary ensemble backend: %s", e)
                self._secondary_backend = None

    @property
    def backend_name(self) -> str:
        return getattr(self._backend, "family", "superb") if self._backend else "superb"

    @property
    def display_name(self) -> str:
        return getattr(self._backend, "display_name", "Wav2Vec2-SUPERB ER")

    @property
    def family(self) -> str:
        return getattr(self._backend, "family", "Wav2Vec2")

    @property
    def training_note(self) -> str:
        return getattr(self._backend, "training_note", "IEMOCAP Affective DB")

    @property
    def confidence_informative(self) -> bool:
        return getattr(self._backend, "confidence_informative", True)

    def backend_info(self) -> Dict[str, Any]:
        """Return provenance and execution metadata for the active voice backend.

        Uses getattr with defaults throughout so that @st.cache_resource instances
        created before Phase 12 (which lack the new attributes) still work safely.
        """
        backend_used = getattr(self, "backend_used", "superb")
        configured = getattr(self, "configured_backend", getattr(config, "VOICE_BACKEND", "superb"))
        active_model_id = getattr(self._backend, "model_id", self.model_id) if self._backend else self.model_id
        if not active_model_id or active_model_id == "unknown":
            active_model_id = (
                getattr(config, "EMOTION2VEC_MODEL_NAME", "emotion2vec/emotion2vec_plus_base")
                if backend_used == "emotion2vec"
                else getattr(config, "AUDIO_EMOTION_MODEL_NAME", "superb/wav2vec2-base-superb-er")
            )
        return {
            "configured_backend": configured,
            "backend_used": backend_used,
            "model_id": active_model_id,
            "display_name": self.display_name,
            "fallback_active": getattr(self, "fallback_active", False),
            "fallback_reason": getattr(self, "fallback_reason", None),
            "confidence_informative": self.confidence_informative,
        }

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

        # Delegate to pluggable backend
        result = self._backend.predict(audio, sr)

        # Ensemble check if enabled
        if getattr(config, "ENSEMBLE_ENABLED", False) and self._secondary_backend is not None:
            try:
                sec_res = self._secondary_backend.predict(audio, sr)
                primary_emo = result.get("emotion")
                sec_emo = sec_res.get("emotion")
                b_preds = {
                    getattr(self._backend, "family", "primary"): primary_emo,
                    getattr(self._secondary_backend, "family", "secondary"): sec_emo,
                }
                result["backend_predictions"] = b_preds
                if primary_emo != sec_emo:
                    result["uncertain"] = True
                    result["ensemble_disagreement"] = True
                else:
                    result["uncertain"] = False
                    result["ensemble_disagreement"] = False
            except Exception as e:
                logger.warning("Ensemble secondary prediction failed: %s", e)

        return result

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