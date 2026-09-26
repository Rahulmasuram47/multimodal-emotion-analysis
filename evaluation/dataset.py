"""
Evaluation Dataset Loader and Schema for Multimodal Emotion Analysis.

Two supported CSV schemas:
  Legacy (evaluation_dataset_template.csv):
    filename, true_voice_emotion, true_text_sentiment, true_speaker_count,
    true_transcript, notes

  Extended (evaluation_dataset.csv — Phase 2 onward):
    file, source, true_voice_emotion, true_text_sentiment, true_transcript,
    true_speaker_count, true_escalation, true_incongruent, notes, annotator

Ground truth values may be N/A, None, or empty when unavailable.
Blank cells are ALWAYS excluded from metric computation — never treated as a class.
"""

from dataclasses import dataclass
import csv
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from config import AUDIO_EMOTION_CLASSES, TEXT_SENTIMENT_CLASSES

# Legacy schema (template)
LEGACY_REQUIRED_COLUMNS: Sequence[str] = (
    "filename",
    "true_voice_emotion",
    "true_text_sentiment",
    "true_speaker_count",
    "true_transcript",
    "notes",
)

# Extended schema (Phase 2 annotation CSV)
EXTENDED_REQUIRED_COLUMNS: Sequence[str] = (
    "file",
    "true_voice_emotion",
    "true_text_sentiment",
    "true_transcript",
    "true_speaker_count",
    "true_escalation",
    "true_incongruent",
    "notes",
)

# For backwards compat, keep the old name as an alias.
REQUIRED_COLUMNS = LEGACY_REQUIRED_COLUMNS

NA_VALUES = {"n/a", "na", "none", "null", "", "nan"}
BINARY_YES_VALUES = {"yes", "y", "1", "true"}
BINARY_NO_VALUES = {"no", "n", "0", "false"}


def _parse_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    val_str = str(value).strip()
    if val_str.lower() in NA_VALUES:
        return None
    return val_str


def _parse_optional_int(value: Any) -> Optional[int]:
    parsed_str = _parse_optional_str(value)
    if parsed_str is None:
        return None
    try:
        val_float = float(parsed_str)
        return int(val_float)
    except (ValueError, TypeError):
        return None


def _parse_binary_label(value: Any) -> Optional[str]:
    """Parse a yes/no/na binary field.  Returns 'yes', 'no', or None."""
    parsed = _parse_optional_str(value)
    if parsed is None:
        return None
    lv = parsed.lower()
    if lv in BINARY_YES_VALUES:
        return "yes"
    if lv in BINARY_NO_VALUES:
        return "no"
    return None  # unrecognised value → treat as unannotated


def normalize_voice_emotion_label(label: Optional[str]) -> Optional[str]:
    """Normalize raw voice emotion label to standard class (neutral, happy, angry, sad)."""
    parsed = _parse_optional_str(label)
    if parsed is None:
        return None
    norm = parsed.lower().strip()
    mapping = {
        "neu": "neutral",
        "neutral": "neutral",
        "hap": "happy",
        "happy": "happy",
        "ang": "angry",
        "angry": "angry",
        "sad": "sad",
    }
    return mapping.get(norm, norm)


def normalize_text_sentiment_label(label: Optional[str]) -> Optional[str]:
    """Normalize raw text sentiment label to standard class (positive, negative, neutral)."""
    parsed = _parse_optional_str(label)
    if parsed is None:
        return None
    norm = parsed.lower().strip()
    mapping = {
        "pos": "positive",
        "positive": "positive",
        "neg": "negative",
        "negative": "negative",
        "neu": "neutral",
        "neutral": "neutral",
    }
    return mapping.get(norm, norm)


@dataclass
class EvaluationSample:
    """A single evaluation record with optional ground truth labels.

    Blank/NA cells in any field are stored as None and EXCLUDED from metrics.
    """

    filename: str
    true_voice_emotion: Optional[str] = None
    true_text_sentiment: Optional[str] = None
    true_speaker_count: Optional[int] = None
    true_transcript: Optional[str] = None
    # Extended fields (Phase 2)
    true_escalation: Optional[str] = None      # 'yes' | 'no' | None
    true_incongruent: Optional[str] = None     # 'yes' | 'no' | None
    source: str = ""
    annotator: str = ""
    notes: str = ""

    def __post_init__(self) -> None:
        self.filename = str(self.filename).strip()
        self.true_voice_emotion = normalize_voice_emotion_label(self.true_voice_emotion)
        self.true_text_sentiment = normalize_text_sentiment_label(self.true_text_sentiment)
        self.true_speaker_count = _parse_optional_int(self.true_speaker_count)
        self.true_transcript = _parse_optional_str(self.true_transcript)
        self.true_escalation = _parse_binary_label(self.true_escalation)
        self.true_incongruent = _parse_binary_label(self.true_incongruent)
        self.source = str(self.source or "").strip()
        self.annotator = str(self.annotator or "").strip()
        self.notes = str(self.notes or "").strip()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.filename,
            "source": self.source,
            "true_voice_emotion": self.true_voice_emotion or "",
            "true_text_sentiment": self.true_text_sentiment or "",
            "true_transcript": self.true_transcript or "",
            "true_speaker_count": str(self.true_speaker_count) if self.true_speaker_count is not None else "",
            "true_escalation": self.true_escalation or "",
            "true_incongruent": self.true_incongruent or "",
            "notes": self.notes or "",
            "annotator": self.annotator or "",
        }

    def to_legacy_dict(self) -> Dict[str, Any]:
        """Serialise to legacy template column order."""
        return {
            "filename": self.filename,
            "true_voice_emotion": self.true_voice_emotion or "N/A",
            "true_text_sentiment": self.true_text_sentiment or "N/A",
            "true_speaker_count": str(self.true_speaker_count) if self.true_speaker_count is not None else "N/A",
            "true_transcript": self.true_transcript or "N/A",
            "notes": self.notes or "",
        }


# Extended column order used when writing Phase-2 CSVs.
EXTENDED_COLUMNS: Sequence[str] = (
    "file",
    "source",
    "true_voice_emotion",
    "true_text_sentiment",
    "true_transcript",
    "true_speaker_count",
    "true_escalation",
    "true_incongruent",
    "notes",
    "annotator",
)


class EvaluationDataset:
    """Collection of evaluation samples with validation and filtering utilities."""

    def __init__(self, samples: Optional[List[EvaluationSample]] = None) -> None:
        self.samples: List[EvaluationSample] = samples or []

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def __getitem__(self, index: int) -> EvaluationSample:
        return self.samples[index]

    def add_sample(self, sample: EvaluationSample) -> None:
        self.samples.append(sample)

    # ── Presence checks ──────────────────────────────────────────────────────

    @property
    def has_voice_emotion_labels(self) -> bool:
        return any(s.true_voice_emotion is not None for s in self.samples)

    @property
    def has_text_sentiment_labels(self) -> bool:
        return any(s.true_text_sentiment is not None for s in self.samples)

    @property
    def has_transcript_labels(self) -> bool:
        return any(s.true_transcript is not None for s in self.samples)

    @property
    def has_speaker_count_labels(self) -> bool:
        return any(s.true_speaker_count is not None for s in self.samples)

    @property
    def has_escalation_labels(self) -> bool:
        return any(s.true_escalation is not None for s in self.samples)

    @property
    def has_incongruence_labels(self) -> bool:
        return any(s.true_incongruent is not None for s in self.samples)

    # ── Filters ──────────────────────────────────────────────────────────────

    def filter_with_voice_emotion(self) -> List[EvaluationSample]:
        return [s for s in self.samples if s.true_voice_emotion is not None]

    def filter_with_text_sentiment(self) -> List[EvaluationSample]:
        return [s for s in self.samples if s.true_text_sentiment is not None]

    def filter_with_transcript(self) -> List[EvaluationSample]:
        return [s for s in self.samples if s.true_transcript is not None]

    def filter_with_speaker_count(self) -> List[EvaluationSample]:
        return [s for s in self.samples if s.true_speaker_count is not None]

    def filter_with_escalation(self) -> List[EvaluationSample]:
        return [s for s in self.samples if s.true_escalation is not None]

    def filter_with_incongruence(self) -> List[EvaluationSample]:
        return [s for s in self.samples if s.true_incongruent is not None]

    # ── I/O ──────────────────────────────────────────────────────────────────

    def to_csv(self, file_path_or_buffer: Union[str, Path, io.StringIO]) -> None:
        """Write dataset to extended CSV format."""
        is_buffer = hasattr(file_path_or_buffer, "write")
        f = file_path_or_buffer if is_buffer else open(file_path_or_buffer, "w", newline="", encoding="utf-8")
        try:
            writer = csv.DictWriter(f, fieldnames=list(EXTENDED_COLUMNS))
            writer.writeheader()
            for sample in self.samples:
                writer.writerow(sample.to_dict())
        finally:
            if not is_buffer:
                f.close()

    def to_legacy_csv(self, file_path_or_buffer: Union[str, Path, io.StringIO]) -> None:
        """Write dataset using the legacy template column schema."""
        is_buffer = hasattr(file_path_or_buffer, "write")
        f = file_path_or_buffer if is_buffer else open(file_path_or_buffer, "w", newline="", encoding="utf-8")
        try:
            writer = csv.DictWriter(f, fieldnames=list(LEGACY_REQUIRED_COLUMNS))
            writer.writeheader()
            for sample in self.samples:
                writer.writerow(sample.to_legacy_dict())
        finally:
            if not is_buffer:
                f.close()

    @classmethod
    def from_csv_string(cls, csv_content: str) -> "EvaluationDataset":
        """Parse dataset from a CSV string — supports both legacy and extended schemas."""
        f = io.StringIO(csv_content.strip())
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV header is empty or missing.")

        fieldnames_lower = [fn.lower().strip() for fn in reader.fieldnames]
        is_extended = "file" in fieldnames_lower

        if is_extended:
            missing = [col for col in EXTENDED_REQUIRED_COLUMNS if col not in reader.fieldnames]
            if missing:
                raise ValueError(f"CSV missing required columns: {missing}. Expected: {list(EXTENDED_REQUIRED_COLUMNS)}")
        else:
            missing = [col for col in LEGACY_REQUIRED_COLUMNS if col not in reader.fieldnames]
            if missing:
                raise ValueError(f"CSV missing required columns: {missing}. Expected: {list(LEGACY_REQUIRED_COLUMNS)}")

        samples: List[EvaluationSample] = []
        for row in reader:
            if is_extended:
                fname = row.get("file", "").strip()
            else:
                fname = row.get("filename", "").strip()

            if not fname:
                continue

            samples.append(
                EvaluationSample(
                    filename=fname,
                    true_voice_emotion=row.get("true_voice_emotion"),
                    true_text_sentiment=row.get("true_text_sentiment"),
                    true_speaker_count=row.get("true_speaker_count"),
                    true_transcript=row.get("true_transcript"),
                    true_escalation=row.get("true_escalation") if is_extended else None,
                    true_incongruent=row.get("true_incongruent") if is_extended else None,
                    source=row.get("source", "") if is_extended else "",
                    annotator=row.get("annotator", "") if is_extended else "",
                    notes=row.get("notes", ""),
                )
            )
        return cls(samples)

    @classmethod
    def from_csv(cls, file_path: Union[str, Path]) -> "EvaluationDataset":
        """Load dataset from a CSV file (auto-detects schema)."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Evaluation dataset file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8-sig") as f:
            return cls.from_csv_string(f.read())


def load_evaluation_dataset(path: Union[str, Path, None] = None) -> EvaluationDataset:
    """Load the evaluation dataset.

    If *path* is None, falls back to config.EVAL_DATASET_PATH then
    data/evaluation_dataset_template.csv (legacy).
    """
    if path is None:
        from config import EVAL_DATASET_PATH
        candidate = Path(EVAL_DATASET_PATH)
        if not candidate.exists():
            # Fall back to legacy template
            candidate = Path("data/evaluation_dataset_template.csv")
        path = candidate
    return EvaluationDataset.from_csv(path)
