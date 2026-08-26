"""
Voice Emotion Analysis Module.

Uses Wav2Vec2 to predict emotion from preprocessed speech audio.
"""

import logging

import torch
import numpy as np
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

from app.audio_processor import load_and_preprocess_audio

logger = logging.getLogger(__name__)


class VoiceEmotionAnalyzer:
    """Analyze speech emotion using a pretrained Wav2Vec2 model."""

    def __init__(self):
        self.model_id = "superb/wav2vec2-base-superb-er"

        logger.info("Loading Wav2Vec2 emotion model...")

        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_id
        )

        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self.model_id
        )

        self.model.eval()

        logger.info("Wav2Vec2 emotion model loaded successfully.")

    def analyze(self, audio_path: str):
        """
        Analyze emotion from an audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Dictionary containing predicted emotion,
            confidence, and all emotion scores.
        """

        # Use the project's audio processing pipeline
        audio, sr, duration = load_and_preprocess_audio(audio_path)

        # Convert to numpy float32 if necessary
        audio = np.asarray(audio, dtype=np.float32)

        # Prepare Wav2Vec2 input
        inputs = self.extractor(
            audio,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )

        # Run inference
        with torch.no_grad():
            logits = self.model(**inputs).logits

        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=-1)[0]

        # Get predicted emotion
        predicted_index = int(torch.argmax(probabilities))

        emotion = self.model.config.id2label[predicted_index]
        confidence = float(probabilities[predicted_index])

        # Get all emotion scores
        scores = {
            self.model.config.id2label[i]: float(probabilities[i])
            for i in range(len(probabilities))
        }

        result = {
            "emotion": emotion,
            "confidence": confidence,
            "scores": scores,
            "duration_seconds": duration,
            "sample_rate": sr,
        }

        logger.info(
            f"Voice emotion prediction: {emotion} "
            f"(confidence={confidence:.4f})"
        )

        return result