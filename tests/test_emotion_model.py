import os
import pytest

if not os.path.exists("sample_audio/real_voice.wav") or os.environ.get("HF_HUB_OFFLINE") == "1":
    pytest.skip("sample_audio missing or offline mode active", allow_module_level=True)

import soundfile as sf
import torch
import librosa
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)

AUDIO_PATH = "sample_audio/real_voice.wav"
MODEL_ID = "superb/wav2vec2-base-superb-er"
TARGET_SR = 16000

# Load original audio
audio, original_sr = sf.read(AUDIO_PATH)

print(f"Original sample rate: {original_sr} Hz")
print(f"Original samples: {len(audio)}")

# Convert stereo to mono if necessary
if audio.ndim > 1:
    audio = audio.mean(axis=1)

# Resample to 16 kHz if necessary
if original_sr != TARGET_SR:
    audio = librosa.resample(
        audio.astype("float32"),
        orig_sr=original_sr,
        target_sr=TARGET_SR,
    )

print(f"Processed sample rate: {TARGET_SR} Hz")
print(f"Processed samples: {len(audio)}")

# Load processor and model
processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)

# Prepare audio
inputs = processor(
    audio,
    sampling_rate=TARGET_SR,
    return_tensors="pt",
    padding=True,
)

# Predict emotion
with torch.no_grad():
    logits = model(**inputs).logits

probabilities = torch.softmax(logits, dim=-1)
predicted_class = torch.argmax(logits, dim=-1).item()
predicted_emotion = model.config.id2label[predicted_class]

print("\nEmotion Prediction")
print("------------------")
print(f"Predicted class: {predicted_class}")
print(f"Emotion: {predicted_emotion}")

print("\nEmotion Scores")
print("--------------")

for class_id, score in enumerate(probabilities.squeeze().tolist()):
    label = model.config.id2label[class_id]
    print(f"{label}: {score:.4f}")