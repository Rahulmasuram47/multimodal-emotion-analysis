import os
import pytest

if not os.path.exists("sample_audio/real_voice.wav") or os.environ.get("HF_HUB_OFFLINE") == "1":
    pytest.skip("sample_audio missing or offline mode active", allow_module_level=True)

from app.speech_to_text import SpeechToText


AUDIO_PATH = "sample_audio/real_voice.wav"


print("Loading Speech-to-Text model...")

stt = SpeechToText(
    model_size="base",
    device="cpu",
    compute_type="int8",
)

print("\nTranscribing audio...")

result = stt.transcribe(AUDIO_PATH)

print("\nSpeech-to-Text Result")
print("--------------------")

print("Language:", result["language"])
print(
    "Language probability:",
    round(result["language_probability"], 4),
)

print("\nTranscript:")
print(result["text"])

print("\nSegments:")

for segment in result["segments"]:
    print(
        f"[{segment['start']:.2f}s - "
        f"{segment['end']:.2f}s] "
        f"{segment['text']}"
    )