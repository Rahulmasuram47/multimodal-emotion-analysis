"""Abstract base class for voice emotion backends."""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

import numpy as np


class VoiceBackend(ABC):
    """Contract all voice-emotion backends must satisfy.

    predict() must return a dict with exactly these keys:
      - emotion    : str  one of {neutral, happy, sad, angry, other}
      - confidence : float  probability of the top label [0, 1]
      - scores     : dict  {label: probability}  all canonical labels
      - top2       : list  [(label, prob), (label, prob)]  top-2 by prob
      - raw_label  : str  the original model label before mapping
    """

    CANONICAL = {"neutral", "happy", "sad", "angry", "other"}

    display_name: str = "Unknown Voice Model"
    model_id: str = "unknown"
    family: str = "unknown"
    training_note: str = "unknown"
    confidence_informative: bool = True

    @abstractmethod
    def predict(self, waveform: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        """Run inference on a 1-D float32 waveform at *sr* Hz."""
        ...

    # ------------------------------------------------------------------ #
    # Shared helper                                                        #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _top2(scores: Dict[str, float]) -> List[Tuple[str, float]]:
        sorted_pairs = sorted(scores.items(), key=lambda kv: -kv[1])
        return sorted_pairs[:2]
