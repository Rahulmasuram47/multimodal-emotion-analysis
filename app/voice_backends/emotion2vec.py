"""emotion2vec+ voice-emotion backend (FunASR, HuggingFace hub).

Label mapping
-------------
emotion2vec_plus_base produces bilingual labels such as "生气/angry".
We parse the English portion (after '/') and map:
  angry/happy/neutral/sad  -> same canonical label
  disgusted/fearful/surprised/other/unk/<unknown>  -> "other"
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

from app.voice_backends.base import VoiceBackend
import config

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Label mapping                                                        #
# ------------------------------------------------------------------ #
_DIRECT = {"angry", "happy", "neutral", "sad"}
_OTHER_SET = {"disgusted", "fearful", "surprised", "other", "unk"}


def _parse_english(raw: str) -> str:
    """Extract the English part from a bilingual label like '生气/angry'."""
    if "/" in raw:
        return raw.split("/")[-1].strip().lower()
    return raw.strip().lower()


def _map_label(raw: str) -> str:
    eng = _parse_english(raw)
    if eng in _DIRECT:
        return eng
    return "other"


def _normalise(scores: Dict[str, float]) -> Dict[str, float]:
    total = sum(scores.values())
    if total <= 0:
        return {k: 0.0 for k in scores}
    return {k: v / total for k, v in scores.items()}


class Emotion2VecBackend(VoiceBackend):
    """Lazy-loaded emotion2vec+ backend via FunASR."""

    display_name: str = "emotion2vec+ base"
    family: str = "emotion2vec"
    training_note: str = "Bilingual (ZH/EN) Affective Speech Corpus"
    confidence_informative: bool = False

    def __init__(self, model_id: Optional[str] = None):
        self._model_id = model_id or getattr(
            config, "EMOTION2VEC_MODEL_NAME", "emotion2vec/emotion2vec_plus_base"
        )
        self.model_id = self._model_id
        self._model = None  # lazy

    def _load(self):
        if self._model is not None:
            return
        try:
            from funasr import AutoModel  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "funasr is required for Emotion2VecBackend. "
                "Install with: pip install funasr"
            ) from exc
        logger.info("Loading emotion2vec model: %s", self._model_id)
        self._model = AutoModel(model=self._model_id, hub="hf")
        logger.info("emotion2vec model loaded.")

    def predict(self, waveform: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        self._load()
        # FunASR expects float32 numpy array input; returns list of dicts
        result_list: List[Dict] = self._model.generate(
            waveform.astype(np.float32),
            granularity="utterance",
            extract_embedding=False,
        )
        # First (and only) result for utterance-level
        res = result_list[0] if result_list else {}

        # Extract raw labels and scores
        # shape: res["labels"] = ["生气/angry", ...], res["scores"] = [0.8, ...]
        raw_labels: List[str] = res.get("labels", [])
        raw_scores: List[float] = res.get("scores", [])

        if not raw_labels:
            # Fallback: no output
            empty: Dict[str, float] = {k: 0.0 for k in self.CANONICAL}
            return {
                "emotion": "other",
                "confidence": 0.0,
                "scores": empty,
                "top2": [("other", 0.0), ("neutral", 0.0)],
                "raw_label": "unk",
            }

        # Build canonical score dict
        canonical_scores: Dict[str, float] = {k: 0.0 for k in self.CANONICAL}
        raw_score_map: Dict[str, float] = dict(zip(raw_labels, raw_scores))
        for raw_lbl, prob in raw_score_map.items():
            canonical_scores[_map_label(raw_lbl)] += prob
        canonical_scores = _normalise(canonical_scores)

        # Top predicted
        top_raw = max(raw_score_map, key=lambda k: raw_score_map[k])
        emotion = _map_label(top_raw)
        confidence = canonical_scores[emotion]
        top2 = self._top2(canonical_scores)

        return {
            "emotion": emotion,
            "confidence": confidence,
            "scores": canonical_scores,
            "top2": top2,
            "raw_label": top_raw,
        }
