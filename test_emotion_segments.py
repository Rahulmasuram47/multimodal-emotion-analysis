from app.audio_processor import load_and_preprocess_audio
from app.emotion_analyzer import VoiceEmotionAnalyzer


AUDIO_PATH = "sample_audio/real_voice.wav"


print("Loading audio...")

audio, sr, duration = load_and_preprocess_audio(AUDIO_PATH)

print(f"Duration: {duration:.2f}s")
print(f"Sample rate: {sr} Hz")


print("\nLoading emotion model...")

analyzer = VoiceEmotionAnalyzer()


print("\nAnalyzing first segment...")

result = analyzer.analyze_segment(
    audio,
    sr,
    start_seconds=0.0,
    end_seconds=9.0,
)

print("\nSegment Emotion Result")
print("----------------------")

print(
    "Time:",
    result["start_seconds"],
    "to",
    result["end_seconds"],
    "seconds",
)

print("Emotion:", result["emotion"])
print("Confidence:", round(result["confidence"], 4))

print("\nAll Scores:")

for emotion, score in result["scores"].items():
    print(f"{emotion}: {score:.4f}")