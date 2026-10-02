"""
Tests for Speech-to-Text resilience, VAD silent audio handling,
word timestamps extraction, and text reliability computation.
"""

from unittest.mock import MagicMock, patch
import pytest
import numpy as np

import config
from app.speech_to_text import SpeechToText
from app.sentiment_analyzer import SentimentAnalyzer



class TestSpeechVADResilience:
    """Verify STT and VAD edge cases, silent audio, and text reliability."""

    @pytest.fixture
    def mock_whisper(self):
        with patch("app.speech_to_text.WhisperModel") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            yield mock_inst

    def test_silent_audio_produces_no_speech_detected_flag(self, mock_whisper):
        """When VAD filters out all frames (pure silence), no_speech_detected is True."""
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99
        mock_whisper.transcribe.return_value = ([], mock_info)

        stt = SpeechToText(model_size="base")
        res = stt.transcribe("dummy_silent.wav")

        assert res["no_speech_detected"] is True
        assert res["text"] == ""
        assert len(res["segments"]) == 0
        assert res["language"] == "en"

    def test_text_reliability_high_confidence(self, mock_whisper):
        """High logprob and low no_speech_prob yield high text_reliability."""
        mock_seg = MagicMock()
        mock_seg.text = "Hello world, this is a clear recording."
        mock_seg.start = 0.0
        mock_seg.end = 2.5
        mock_seg.avg_logprob = -0.10
        mock_seg.no_speech_prob = 0.02
        mock_seg.words = []

        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.98
        mock_whisper.transcribe.return_value = ([mock_seg], mock_info)

        stt = SpeechToText()
        res = stt.transcribe("audio.wav")

        assert res["no_speech_detected"] is False
        seg = res["segments"][0]
        assert seg["text_reliability"] > 0.85
        assert seg["is_low_confidence"] is False

    def test_text_reliability_low_confidence_due_to_negative_logprob(self, mock_whisper):
        """Very negative avg_logprob triggers low confidence flag."""
        mock_seg = MagicMock()
        mock_seg.text = "Muffled whispering here."
        mock_seg.start = 1.0
        mock_seg.end = 3.0
        mock_seg.avg_logprob = -2.20
        mock_seg.no_speech_prob = 0.05
        mock_seg.words = []

        mock_whisper.transcribe.return_value = ([mock_seg], MagicMock())

        stt = SpeechToText()
        res = stt.transcribe("audio.wav")

        seg = res["segments"][0]
        assert seg["is_low_confidence"] is True
        assert seg["text_reliability"] < 0.65

    def test_text_reliability_low_confidence_due_to_high_no_speech(self, mock_whisper):
        """High no_speech_prob triggers low confidence flag."""
        mock_seg = MagicMock()
        mock_seg.text = "Background noise artifacts."
        mock_seg.start = 0.0
        mock_seg.end = 1.5
        mock_seg.avg_logprob = -0.30
        mock_seg.no_speech_prob = 0.75
        mock_seg.words = []

        mock_whisper.transcribe.return_value = ([mock_seg], MagicMock())

        stt = SpeechToText()
        res = stt.transcribe("audio.wav")

        seg = res["segments"][0]
        assert seg["is_low_confidence"] is True
        assert seg["no_speech_prob"] == 0.75

    def test_text_reliability_clipping_bounds(self, mock_whisper):
        """Extreme negative logprob is clipped at minimum 0.20."""
        mock_seg = MagicMock()
        mock_seg.text = "Completely garbled."
        mock_seg.start = 0.0
        mock_seg.end = 1.0
        mock_seg.avg_logprob = -100.0
        mock_seg.no_speech_prob = 1.0
        mock_seg.words = []

        mock_whisper.transcribe.return_value = ([mock_seg], MagicMock())

        stt = SpeechToText()
        res = stt.transcribe("audio.wav")

        seg = res["segments"][0]
        assert seg["text_reliability"] == 0.20

    def test_word_timestamps_parsing(self, mock_whisper):
        """Word-level timestamps are correctly parsed when present."""
        w1 = MagicMock(word="Hello", start=0.1, end=0.5, probability=0.95)
        w2 = MagicMock(word="there", start=0.6, end=1.0, probability=0.92)

        mock_seg = MagicMock()
        mock_seg.text = "Hello there"
        mock_seg.start = 0.1
        mock_seg.end = 1.0
        mock_seg.avg_logprob = -0.15
        mock_seg.no_speech_prob = 0.01
        mock_seg.words = [w1, w2]

        mock_whisper.transcribe.return_value = ([mock_seg], MagicMock())

        stt = SpeechToText()
        res = stt.transcribe("audio.wav")

        words = res["segments"][0]["words"]
        assert len(words) == 2
        assert words[0]["word"] == "Hello"
        assert words[0]["start"] == 0.1
        assert words[1]["word"] == "there"
        assert words[1]["end"] == 1.0

    def test_empty_words_attribute_handled_gracefully(self, mock_whisper):
        """If words attribute is None, empty list is stored without error."""
        mock_seg = MagicMock()
        mock_seg.text = "Just words"
        mock_seg.start = 0.0
        mock_seg.end = 1.0
        mock_seg.avg_logprob = -0.1
        mock_seg.no_speech_prob = 0.0
        mock_seg.words = None

        mock_whisper.transcribe.return_value = ([mock_seg], MagicMock())

        stt = SpeechToText()
        res = stt.transcribe("audio.wav")

        assert res["segments"][0]["words"] == []

    def test_whitespace_only_segments_filtered_out(self, mock_whisper):
        """Segments with only whitespace or empty text are skipped."""
        seg1 = MagicMock(text="   ", start=0.0, end=1.0, words=[])
        seg2 = MagicMock(text="Actual text", start=1.0, end=2.0, avg_logprob=-0.2, no_speech_prob=0.01, words=[])
        seg3 = MagicMock(text="", start=2.0, end=3.0, words=[])

        mock_whisper.transcribe.return_value = ([seg1, seg2, seg3], MagicMock())

        stt = SpeechToText()
        res = stt.transcribe("audio.wav")

        assert len(res["segments"]) == 1
        assert res["segments"][0]["text"] == "Actual text"
        assert res["text"] == "Actual text"

    def test_language_forced_sets_whisper_param(self, mock_whisper):
        """When language is explicitly specified, whisper receives language param."""
        mock_whisper.transcribe.return_value = ([], MagicMock())

        stt = SpeechToText(language="en")
        stt.transcribe("audio.wav")

        call_kwargs = mock_whisper.transcribe.call_args[1]
        assert call_kwargs["language"] == "en"

    def test_language_auto_sets_whisper_param_none(self, mock_whisper):
        """When language is 'auto', whisper receives None so it can auto-detect."""
        mock_whisper.transcribe.return_value = ([], MagicMock())

        stt = SpeechToText(language="auto")
        stt.transcribe("audio.wav")

        call_kwargs = mock_whisper.transcribe.call_args[1]
        assert call_kwargs["language"] is None

    def test_sentiment_analyzer_validates_empty_text(self):
        """SentimentAnalyzer validates non-empty input by raising ValueError."""
        with patch("app.sentiment_analyzer.AutoTokenizer"), patch("app.sentiment_analyzer.AutoModelForSequenceClassification"):
            analyzer = SentimentAnalyzer()
            with pytest.raises(ValueError, match="Text cannot be empty"):
                analyzer.analyze("")
            with pytest.raises(ValueError, match="Text cannot be empty"):
                analyzer.analyze("    ")

