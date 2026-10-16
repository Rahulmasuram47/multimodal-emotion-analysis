import os
import pytest

if not os.path.exists("sample_audio/real_voice.wav") or os.environ.get("HF_HUB_OFFLINE") == "1":
    pytest.skip("sample_audio missing or offline mode active", allow_module_level=True)

from app.emotion_analyzer import VoiceEmotionAnalyzer


analyzer = VoiceEmotionAnalyzer()

result = analyzer.analyze(
    "sample_audio/real_voice.wav"
)

print("\nVoice Emotion Result")
print("--------------------")

print("Emotion:", result["emotion"])
print("Confidence:", round(result["confidence"], 4))

print("\nAll Scores:")

for emotion, score in result["scores"].items():
    print(f"{emotion}: {score:.4f}")