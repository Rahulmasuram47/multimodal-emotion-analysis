"""
Speaker diarization for the Multimodal Voice & Conversation Emotion Analysis System.

Uses:
    pyannote/speaker-diarization-community-1

The Hugging Face access token is read from the HF_TOKEN environment variable.

This module:
1. Loads the Community-1 diarization pipeline.
2. Runs speaker diarization on an audio file.
3. Prefers exclusive speaker diarization for easier alignment with
   Whisper transcription timestamps.
4. Converts pyannote speaker IDs into Speaker A, Speaker B, etc.
5. Assigns each Whisper segment to the speaker with the greatest
   timestamp overlap.
6. Uses a small nearest-turn fallback when timestamps have tiny gaps.
"""

import os
from typing import Any, Dict, List, Optional


DEFAULT_MODEL_NAME = "pyannote/speaker-diarization-community-1"


class SpeakerDiarizer:
    """Speaker diarization using pyannote Community-1."""

    def __init__(
        self,
        hf_token: Optional[str] = None,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.model_name = model_name
        self._pipeline = None

    def _load_pipeline(self):
        """Load and cache the pyannote diarization pipeline."""

        if self._pipeline is not None:
            return self._pipeline

        if not self.hf_token:
            raise RuntimeError(
                "No Hugging Face token found. Set the HF_TOKEN environment "
                "variable before running speaker diarization."
            )

        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError(
                "pyannote.audio is not installed. Install it with: "
                "pip install pyannote.audio"
            ) from exc

        self._pipeline = Pipeline.from_pretrained(
            self.model_name,
            token=self.hf_token,
        )

        if self._pipeline is None:
            raise RuntimeError(
                f"Could not load diarization pipeline: {self.model_name}"
            )

        return self._pipeline

    @staticmethod
    def _get_annotation(output):
        """
        Get the best available diarization annotation.

        Community-1 provides:
            output.speaker_diarization
            output.exclusive_speaker_diarization

        Exclusive diarization is preferred because it is easier to
        reconcile with Whisper transcription segments.
        """

        exclusive = getattr(
            output,
            "exclusive_speaker_diarization",
            None,
        )

        if exclusive is not None:
            return exclusive

        diarization = getattr(
            output,
            "speaker_diarization",
            None,
        )

        if diarization is not None:
            return diarization

        # Compatibility fallback in case a pipeline directly returns
        # an Annotation-like object.
        if hasattr(output, "itertracks"):
            return output

        raise RuntimeError(
            "The pyannote diarization pipeline returned an unexpected "
            "output format. Could not find speaker_diarization."
        )

    def diarize(self, audio_path: str) -> List[Dict[str, Any]]:
        """
        Run speaker diarization.

        Returns:
            List of dictionaries:
            [
                {
                    "start": 0.0,
                    "end": 3.25,
                    "speaker": "SPEAKER_00"
                },
                ...
            ]
        """

        if not audio_path:
            raise ValueError("audio_path cannot be empty.")

        if not os.path.exists(audio_path):
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        pipeline = self._load_pipeline()

        # Community-1 automatically handles mono conversion and
        # resampling to 16 kHz when required.
        output = pipeline(audio_path)

        annotation = self._get_annotation(output)

        raw_turns: List[Dict[str, Any]] = []

        for turn, _, speaker in annotation.itertracks(
            yield_label=True
        ):
            start = float(turn.start)
            end = float(turn.end)

            if end <= start:
                continue

            raw_turns.append(
                {
                    "start": start,
                    "end": end,
                    "speaker": str(speaker),
                }
            )

        if not raw_turns:
            raise RuntimeError(
                "Speaker diarization completed but no speaker turns "
                "were detected."
            )

        raw_turns.sort(key=lambda item: item["start"])

        # Merge consecutive turns belonging to the same speaker when
        # the gap between them is very small.
        merged_turns: List[Dict[str, Any]] = []

        merge_gap = 0.25

        for turn in raw_turns:
            if (
                merged_turns
                and merged_turns[-1]["speaker"] == turn["speaker"]
                and turn["start"] - merged_turns[-1]["end"]
                <= merge_gap
            ):
                merged_turns[-1]["end"] = max(
                    merged_turns[-1]["end"],
                    turn["end"],
                )
            else:
                merged_turns.append(dict(turn))

        return merged_turns

    @staticmethod
    def _overlap(
        a_start: float,
        a_end: float,
        b_start: float,
        b_end: float,
    ) -> float:
        """Return overlap duration between two time intervals."""

        return max(
            0.0,
            min(a_end, b_end) - max(a_start, b_start),
        )

    @staticmethod
    def _distance_to_turn(
        seg_start: float,
        seg_end: float,
        turn_start: float,
        turn_end: float,
    ) -> float:
        """
        Return the time distance between a segment and a diarization turn.

        Returns 0 when the intervals overlap.
        """

        if seg_end < turn_start:
            return turn_start - seg_end

        if turn_end < seg_start:
            return seg_start - turn_end

        return 0.0

    @staticmethod
    def _build_label_map(
        turns: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """
        Convert pyannote speaker IDs into human-readable labels.

        Example:
            SPEAKER_00 -> Speaker A
            SPEAKER_01 -> Speaker B
        """

        label_map: Dict[str, str] = {}

        for turn in turns:
            raw_speaker = str(turn["speaker"])

            if raw_speaker in label_map:
                continue

            speaker_index = len(label_map)

            # Supports A-Z normally.
            if speaker_index < 26:
                label = f"Speaker {chr(ord('A') + speaker_index)}"
            else:
                label = f"Speaker {speaker_index + 1}"

            label_map[raw_speaker] = label

        return label_map

    def assign_speakers(
        self,
        segments: List[Dict[str, Any]],
        turns: List[Dict[str, Any]],
        label_map: Optional[Dict[str, str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Assign a speaker to each transcription/emotion segment.

        The primary method is maximum timestamp overlap.

        If there is no overlap because of a small timestamp mismatch,
        the closest diarization turn is used when it is within the
        configured fallback tolerance.
        """

        if not segments:
            return []

        if not turns:
            return [
                {
                    **dict(segment),
                    "speaker": "Unknown",
                }
                for segment in segments
            ]

        if label_map is None:
            label_map = self._build_label_map(turns)

        labeled_segments: List[Dict[str, Any]] = []

        # Maximum allowed timestamp gap for nearest-turn fallback.
        nearest_turn_tolerance = 1.0

        for segment in segments:
            seg = dict(segment)

            seg_start = float(
                seg.get("start_seconds", seg.get("start", 0.0))
            )

            seg_end = float(
                seg.get(
                    "end_seconds",
                    seg.get("end", seg_start),
                )
            )

            if seg_end < seg_start:
                seg_start, seg_end = seg_end, seg_start

            best_turn: Optional[Dict[str, Any]] = None
            best_overlap = 0.0

            # ---------------------------------------------------------
            # 1. Primary assignment: maximum timestamp overlap
            # ---------------------------------------------------------
            for turn in turns:
                overlap = self._overlap(
                    seg_start,
                    seg_end,
                    float(turn["start"]),
                    float(turn["end"]),
                )

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_turn = turn

            # ---------------------------------------------------------
            # 2. Fallback: nearest turn for tiny timestamp differences
            # ---------------------------------------------------------
            if best_turn is None:
                nearest_turn: Optional[Dict[str, Any]] = None
                nearest_distance = float("inf")

                for turn in turns:
                    distance = self._distance_to_turn(
                        seg_start,
                        seg_end,
                        float(turn["start"]),
                        float(turn["end"]),
                    )

                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_turn = turn

                if (
                    nearest_turn is not None
                    and nearest_distance <= nearest_turn_tolerance
                ):
                    best_turn = nearest_turn

            # ---------------------------------------------------------
            # 3. Store speaker label
            # ---------------------------------------------------------
            if best_turn is not None:
                raw_label = str(best_turn["speaker"])

                seg["speaker"] = label_map.get(
                    raw_label,
                    raw_label,
                )
            else:
                seg["speaker"] = "Unknown"

            labeled_segments.append(seg)

        return labeled_segments