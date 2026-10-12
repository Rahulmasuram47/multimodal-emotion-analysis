"""
Central configuration file for Multimodal Emotion Analysis System.
Holds all model choices, audio processing parameters, fusion weights, and thresholds.
"""
import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
SAMPLE_AUDIO_DIR = os.path.join(PROJECT_ROOT, 'sample_audio')

# Audio Processing Configuration
TARGET_SAMPLE_RATE = 16000  # Standard sample rate for speech models (Hz)
SUPPORTED_AUDIO_EXTENSIONS = ('.wav', '.flac', '.mp3', '.ogg', '.m4a')
MAX_AUDIO_DURATION_SECONDS = 600  # Maximum 10 minutes per conversation
MIN_AUDIO_DURATION_SECONDS = 0.5  # Minimum 0.5 seconds for analysis
N_MFCC = 13  # Number of MFCC coefficients to extract
N_FFT = 2048
HOP_LENGTH = 512

# Locked-In Model Identifiers (Section 2 of PROJECT_SPEC.md)
AUDIO_EMOTION_MODEL_NAME = 'superb/wav2vec2-base-superb-er'
STT_MODEL_SIZE = 'base'  # faster-whisper model size ('base' or 'small')
TEXT_SENTIMENT_MODEL_NAME = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
TEXT_SENTIMENT_FALLBACK_MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'

# Voice backend selector: "superb" (production baseline) or "emotion2vec" (default)
VOICE_BACKEND: str = "emotion2vec"
# emotion2vec model id
EMOTION2VEC_MODEL_NAME: str = "emotion2vec/emotion2vec_plus_base"
# Ensemble configuration: if True, runs both backends and detects disagreement
ENSEMBLE_ENABLED: bool = False
# Floor for voice reliability when an unmapped/other label or ensemble disagreement occurs
VOICE_RELIABILITY_FLOOR_OTHER: float = 0.35
# emotion2vec softmax is saturated (mean conf 0.998 correct / 0.999 wrong on the 8-clip dev
# benchmark, accuracy 62.5%, preliminary); cap confidence near measured accuracy; revisit
# with more labelled data.
VOICE_CONF_CAP: float = 0.65

# Emotion & Sentiment Categories
AUDIO_EMOTION_CLASSES = ['neutral', 'happy', 'angry', 'sad']
TEXT_SENTIMENT_CLASSES = ['negative', 'neutral', 'positive']

# Fusion Configuration
DEFAULT_AUDIO_WEIGHT = 0.5
DEFAULT_TEXT_WEIGHT = 0.5
CONFIDENCE_THRESHOLD = 0.4

# Evaluation & Annotation Configuration
# ─────────────────────────────────────────────────────────────────────────
# Minimum number of annotated samples required before a metric is reported.
# Rationale: with N < 5, a single misclassification swings accuracy by ≥ 20 pp,
# making the metric statistically uninformative. 5 is the practical floor.
EVAL_MIN_N: int = 5

# Number of bootstrap resampling iterations for 95% CI estimation.
# 1000 gives stable CI width; 2000 is better if run time permits.
EVAL_BOOTSTRAP_ITERATIONS: int = 1000

# Random seed for bootstrap and learned-fusion CV (reproducibility).
EVAL_RANDOM_SEED: int = 42

# Path (relative to project root) for the annotation-ready evaluation dataset.
# The original template is kept untouched at data/evaluation_dataset_template.csv.
EVAL_DATASET_PATH: str = 'data/evaluation_dataset.csv'

# Folder where team recordings should be saved (created by Annotation Helper).
RECORDINGS_DIR: str = 'recordings'

# Escalation classes used for binary escalation metrics.
ESCALATION_CLASSES = ['yes', 'no']

# Incongruence classes used for binary incongruence metrics.
INCONGRUENCE_CLASSES = ['yes', 'no']

# Voice emotion label options shown in Annotation Helper dropdowns.
VOICE_EMOTION_OPTIONS = ['neutral', 'happy', 'angry', 'sad']

# Text sentiment label options shown in Annotation Helper dropdowns.
TEXT_SENTIMENT_OPTIONS = ['positive', 'neutral', 'negative']

# ─────────────────────────────────────────────────────────────────────────
# Phase 3: Voice Emotion Calibration
# ─────────────────────────────────────────────────────────────────────────
# Master switch. When False, calibration is fully bypassed and raw model
# output is used everywhere (identical to pre-Phase-3 behavior).
VOICE_CALIBRATION_ENABLED: bool = True

# Prosody sanity rule (data-free heuristic, usable even at very small N).
VOICE_CALIBRATION_PROSODY_RULE_ENABLED: bool = True
VOICE_CALIBRATION_RMS_THRESHOLD: float = 0.05          # tune once labeled data exists
VOICE_CALIBRATION_ZCR_STD_THRESHOLD: float = 0.02      # tune once labeled data exists
VOICE_CALIBRATION_RELATIVE_LOUDNESS_THRESHOLD: float = 0.40  # Normalized relative loudness [0, 1]
VOICE_CALIBRATION_F0_RANGE_SEMITONES_THRESHOLD: float = 6.0   # Semitones range over voiced frames
VOICE_CALIBRATION_SAD_DAMPENING_FACTOR: float = 0.5    # 0.5 = halve 'sad' mass when rule fires

# Temperature scaling & prior correction (data-fitted; refuses to fit below this many labeled
# voice-emotion samples, to avoid overfitting a correction to a handful
# of clips). Re-run fitting once more team recordings are annotated.
VOICE_CALIBRATION_MIN_FIT_SAMPLES: int = 15

# Voice-model collapse detector: if one class gets above this share of predictions
# across a recording/batch, flag collapse and down-weight voice reliability in fusion.
VOICE_COLLAPSE_THRESHOLD: float = 0.70
VOICE_COLLAPSE_RELIABILITY_PENALTY: float = 0.50

# Tier 4 (Item 32): Voice confidence & reliability thresholds
# When segment voice confidence falls below this threshold, apply a reliability penalty
# to lower the voice weight in multimodal fusion and show 'Low voice reliability' chip.
LOW_VOICE_CONFIDENCE_THRESHOLD: float = 0.50
LOW_VOICE_CONFIDENCE_RELIABILITY_PENALTY: float = 0.60
LOW_VOICE_RELIABILITY_DISPLAY_THRESHOLD: float = 0.65


# ─────────────────────────────────────────────────────────────────────────
# Phase 3: Short-Segment Handling
# ─────────────────────────────────────────────────────────────────────────
# Segments shorter than this are considered too short for reliable acoustic
# emotion inference on their own. Their audio window is widened using
# neighboring context before running the acoustic model.
SHORT_SEGMENT_THRESHOLD_SECONDS: float = 1.0

# When widening a short segment's audio window, expand by this many seconds
# on each side (clamped to the full audio's bounds and to not overlap past
# the midpoint of an adjacent segment).
SHORT_SEGMENT_CONTEXT_PADDING_SECONDS: float = 0.5

# Reliability score assigned to a segment whose acoustic window had to be
# widened due to being under SHORT_SEGMENT_THRESHOLD_SECONDS. A normal-
# length segment gets reliability 1.0.
SHORT_SEGMENT_RELIABILITY_SCORE: float = 0.6

# ─────────────────────────────────────────────────────────────────────────
# Phase 3: Speech-to-Text Configuration
# ─────────────────────────────────────────────────────────────────────────
# faster-whisper model size actually used at runtime. STT_MODEL_SIZE above
# was already defined but not wired into SpeechToText's default — this is
# now the single source of truth for production Whisper size.
# Options: 'tiny', 'base', 'small', 'medium', 'large-v3' (CPU-feasible: base/small).

# Language forced on transcription. 'auto' lets Whisper detect it per file
# (matches current/original behavior exactly). Set to an ISO 639-1 code
# (e.g. 'en') to force a language when you know all input will be English —
# this skips language-detection uncertainty entirely.
STT_LANGUAGE: str = 'auto'

# Per-segment ASR confidence threshold used by the UI to flag a transcript
# segment as "uncertain" (low avg_logprob or high no_speech_prob). These
# are faster-whisper's own native segment-level signals.
STT_LOW_CONFIDENCE_AVG_LOGPROB: float = -1.0   # more negative = less confident
STT_HIGH_NO_SPEECH_PROB_THRESHOLD: float = 0.6  # closer to 1.0 = likely non-speech

# ASR validation guards
REPETITION_RATIO_MAX: float = 0.8  # with at least 8 tokens, repetition ratio >= threshold flags segment
NON_ENGLISH_MIN_PROB: float = 0.6  # detected language != 'en' with prob >= threshold flags non_english
LOW_LANG_CONF: float = 0.5         # warning only when language probability < threshold


# ─────────────────────────────────────────────────────────────────────────
# Phase 4: Prosody-Based Intensity
# ─────────────────────────────────────────────────────────────────────────
# 'v2' = new prosody-based intensity (this phase). 'v1' = legacy pure-
# confidence formula (pre-Phase-4), kept available for the ablation
# comparison so the old and new formulas can be reported side by side.
INTENSITY_FORMULA_VERSION: str = 'v2'

# v2 intensity = weighted sum of three components (weights should sum to
# ~1.0, not strictly enforced so you can experiment):
#   prosodic_arousal      — loudness + pitch movement + speech rate, from
#                            this segment's own audio (see app/prosody.py)
#   voice_arousal         — expected arousal from the voice-emotion class
#                            probabilities (angry/happy = high arousal,
#                            sad/neutral = low), NOT just top-class confidence
#   text_negativity       — probability mass on the 'negative' text class
INTENSITY_V2_WEIGHT_PROSODY: float = 0.35
INTENSITY_V2_WEIGHT_VOICE_AROUSAL: float = 0.35
INTENSITY_V2_WEIGHT_TEXT_NEGATIVITY: float = 0.30

# Per-class arousal weights used to compute the expected-value "voice
# arousal" component above. Heuristic starting points (angry/happy =
# high-arousal emotions, sad/neutral = low-arousal) — tune once labeled
# escalation data exists to check whether they help or hurt agreement
# with human-labeled true_escalation.
VOICE_EMOTION_AROUSAL_MAP: dict = {
    'angry': 1.0,
    'happy': 0.75,
    'neutral': 0.25,
    'sad': 0.15,
}

# Reference maximums and percentiles used to normalize raw prosodic features into [0, 1]
# before combining.
PROSODY_RMS_REFERENCE_MAX: float = 0.15          # Fallback RMS amplitude considered "loud"
PROSODY_ZCR_STD_REFERENCE_MAX: float = 0.05      # Fallback ZCR variability
PROSODY_SPEECH_RATE_REFERENCE_MAX: float = 4.0   # words/sec considered a "fast/agitated" rate

# Percentile normalization against recording's own energy distribution
PROSODY_LOUDNESS_PERCENTILE_LOW: float = 10.0    # 10th percentile in dB
PROSODY_LOUDNESS_PERCENTILE_HIGH: float = 90.0   # 90th percentile in dB

# Semitone F0 pitch range normalization
PROSODY_F0_SEMITONE_RANGE_MAX: float = 24.0      # 24 semitones (2 octaves) = 1.0
PROSODY_F0_MIN_HZ: float = 65.0                  # ~C2
PROSODY_F0_MAX_HZ: float = 500.0                 # ~B4

# Weights for combining the normalized prosodic sub-features into
# one prosodic_arousal score (sums to 1.0).
PROSODY_WEIGHT_LOUDNESS: float = 0.35
PROSODY_WEIGHT_PITCH_VARIATION: float = 0.30
PROSODY_WEIGHT_VOICED_RATIO: float = 0.15
PROSODY_WEIGHT_SPEECH_RATE: float = 0.20

# Backward compatibility aliases
PROSODY_WEIGHT_RMS: float = PROSODY_WEIGHT_LOUDNESS

# Segment Acoustic Descriptors (independent of Intensity v2)
PITCH_TREND_ST: float = 2.0                 # Semitones threshold for rising (> +2) or falling (< -2)
PITCH_F_REF_HZ: float = 440.0               # Reference frequency for semitone conversion (A4)
RATE_SLOW_WPS: float = 1.8                  # Words per active second threshold for slow speaking rate
RATE_FAST_WPS: float = 3.3                  # Words per active second threshold for fast speaking rate
PAUSE_DB_BELOW_PEAK: float = 20.0           # Frames below peak RMS dB by this delta are treated as pause
ENERGY_LEVEL_PERCENTILE_LOW: float = 33.3   # Recording frame RMS-dB percentile for low energy
ENERGY_LEVEL_PERCENTILE_HIGH: float = 66.7  # Recording frame RMS-dB percentile for high energy

# ─────────────────────────────────────────────────────────────────────────
# Phase 4: Escalation Detection
# ─────────────────────────────────────────────────────────────────────────
# 'strict' = original rule (every consecutive segment must strictly
#            increase, total rise >= ESCALATION_MIN_NET_RISE). Kept
#            available for the ablation comparison.
# 'tolerant' = new robust-slope rule: fits a linear slope across segment
#            intensities, allows individual dips up to
#            ESCALATION_DIP_TOLERANCE without breaking the trend, and
#            requires both a minimum slope AND a minimum net rise.
ESCALATION_MODE: str = 'tolerant'

ESCALATION_MIN_SEGMENTS: int = 3
ESCALATION_MIN_SLOPE: float = 0.03          # minimum intensity rise per turn (OLS slope)
ESCALATION_MIN_NET_RISE: float = 0.20       # minimum total rise from first to last segment
ESCALATION_DIP_TOLERANCE: float = 0.08      # max single-step drop still tolerated as "still escalating"


# ─────────────────────────────────────────────────────────────────────────
# Phase 5: Fusion Internals
# ─────────────────────────────────────────────────────────────────────────
# When both audio_reliability and text_reliability equal 1.0 (the default
# when neither Phase 3 short-segment widening nor low ASR confidence
# applied to this segment), the formulas below are mathematically
# IDENTICAL to the pre-Phase-5 fusion — this is intentional, so nothing
# that currently passes changes behavior unless a real reliability signal
# is present.

# Reliability assigned to a text-sentiment result when its segment's
# speech-to-text confidence was flagged low (see Phase 3:
# is_low_confidence in app/speech_to_text.py). 1.0 = full reliability,
# used whenever this signal is absent (identical to pre-Phase-5).
FUSION_TEXT_LOW_ASR_RELIABILITY: float = 0.6

# Arousal threshold (from VOICE_EMOTION_AROUSAL_MAP, Phase 4) above which
# a negative-direction voice emotion is described as "high-arousal"
# (anger-like) rather than "low-arousal" (sadness-like) in the fusion
# pattern label. Purely descriptive — never a diagnostic claim about the
# speaker.
FUSION_HIGH_AROUSAL_THRESHOLD: float = 0.6

# Threshold below which combined modality reliability triggers abstention
# ("Insufficient reliability. Review audio") rather than forcing a label.
FUSION_ABSTENTION_THRESHOLD: float = 0.35

# Default fusion weights exposed to the Phase 5 sidebar settings panel
# (session-state overrides only — config.py itself is never mutated at
# runtime). These mirror DEFAULT_AUDIO_WEIGHT/DEFAULT_TEXT_WEIGHT/
# CONFIDENCE_THRESHOLD above and exist so the UI has an explicit
# "reset to defaults" target.
FUSION_SETTINGS_DEFAULTS: dict = {
    "audio_weight": DEFAULT_AUDIO_WEIGHT,
    "text_weight": DEFAULT_TEXT_WEIGHT,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "abstention_threshold": FUSION_ABSTENTION_THRESHOLD,
}


# ─────────────────────────────────────────────────────────────────────────
# Phase 6: Speaker-Aware Analytics
# ─────────────────────────────────────────────────────────────────────────
# Minimum per-segment intensity rise (compared to that speaker's own
# previous segment) to be considered a candidate "escalation initiator"
# moment. Heuristic only — descriptive, not a blame attribution.
SPEAKER_ESCALATION_INITIATOR_MIN_RISE: float = 0.15

# Minimum number of distinct speakers required before speaker-aware
# analytics (per-speaker trends, talk-time share, initiator) are shown.
# Below this, the existing single-speaker Speaker Analysis section is
# sufficient and per-speaker breakdowns would be redundant.
SPEAKER_ANALYTICS_MIN_SPEAKERS: int = 2

# Diarization model identifier (Pyannote community-1)
DIARIZATION_MODEL_NAME: str = "pyannote/speaker-diarization-community-1"

# Minimum ratio of overlap between a speech segment and a diarization turn
# required to assign a speaker label. Below this ratio, the segment is labeled
# "Uncertain Speaker" to avoid forcing uncertain or noisy speaker assignments.
DIARIZATION_MIN_OVERLAP_RATIO: float = 0.35

# ─────────────────────────────────────────────────────────────────────────
# UI Display Constants (Tier 1 fixes)
# ─────────────────────────────────────────────────────────────────────────
# Application version shown in the header badge.
APP_VERSION: str = "2.0"

# Single canonical name for the production fusion strategy — used in the
# header chip AND the sidebar spec row so they always match.
FUSION_STRATEGY_NAME: str = "Multi-Evidence Arbitration"

# Fraction of turns that must share the same dominant emotion/sentiment
# before the key-finding says "predominantly X". Below this the key finding
# uses a "mixed" description listing the per-turn breakdown instead.
PREDOMINANTLY_THRESHOLD: float = 0.60

# Duration (seconds) below which a segment is considered "short" FOR
# DISPLAY PURPOSES (the "(widened short-segment window)" badge). Must be
# kept in sync with SHORT_SEGMENT_THRESHOLD_SECONDS; this alias makes the
# UI threshold explicit and independently configurable.
SHORT_SEG_SECONDS: float = SHORT_SEGMENT_THRESHOLD_SECONDS

# Per-segment text reliability value below which the "(low ASR confidence)"
# badge is shown. Derived from exp(avg_logprob) * (1 - no_speech_prob)
# in app/speech_to_text.py; 0.65 matches the existing hard-coded check.
LOW_ASR_CONFIDENCE_DISPLAY_THRESHOLD: float = 0.65
LOW_ASR_CONFIDENCE_THRESHOLD: float = LOW_ASR_CONFIDENCE_DISPLAY_THRESHOLD