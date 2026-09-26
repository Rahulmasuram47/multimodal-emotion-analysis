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

# Emotion & Sentiment Categories
AUDIO_EMOTION_CLASSES = ['neutral', 'happy', 'angry', 'sad']
TEXT_SENTIMENT_CLASSES = ['negative', 'neutral', 'positive']

# Fusion Configuration
DEFAULT_AUDIO_WEIGHT = 0.5
DEFAULT_TEXT_WEIGHT = 0.5
CONFIDENCE_THRESHOLD = 0.4

# Evaluation & Annotation Configuration
# ─────────────────────────────────────────────────────────────────────────────
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
