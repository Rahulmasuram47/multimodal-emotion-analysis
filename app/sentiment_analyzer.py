"""
Text Sentiment Analysis Module.

Uses CardiffNLP RoBERTa to classify transcript sentiment.
"""

import logging
from typing import Any, Dict

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyze sentiment from text using a pretrained RoBERTa model."""

    def __init__(self):
        self.model_id = (
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )

        logger.info("Loading sentiment model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id
        )

        self.model.eval()

        logger.info("Sentiment model loaded successfully.")

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment from text.

        Args:
            text: Text to analyze.

        Returns:
            Dictionary containing sentiment, confidence,
            and all sentiment scores.
        """

        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        text = text.strip()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
        )

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probabilities = torch.softmax(logits, dim=-1)[0]

        predicted_index = int(torch.argmax(probabilities))

        sentiment = self.model.config.id2label[predicted_index]

        confidence = float(probabilities[predicted_index])

        scores = {
            self.model.config.id2label[i]: float(probabilities[i])
            for i in range(len(probabilities))
        }

        result = {
            "sentiment": sentiment,
            "confidence": confidence,
            "scores": scores,
            "text": text,
        }

        logger.info(
            "Text sentiment prediction: %s "
            "(confidence=%.4f)",
            sentiment,
            confidence,
        )

        return result