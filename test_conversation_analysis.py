from app.conversation_analysis import ConversationAnalyzer


AUDIO_PATH = "sample_audio/real_voice.wav"


print("Starting conversation analysis...")

analyzer = ConversationAnalyzer()

result = analyzer.analyze(AUDIO_PATH)


print("\nConversation Analysis Result")
print("============================")

print("Language:", result["language"])
print(
    "Language probability:",
    round(result["language_probability"], 4),
)

print(
    "Duration:",
    round(result["duration_seconds"], 2),
    "seconds",
)

print("\nFull Transcript:")
print(result["full_transcript"])


print("\nSegment Analysis")
print("================")

for segment in result["segments"]:

    print(
        f"\nSegment {segment['segment_index']}"
    )

    print(
        "Time:",
        f"{segment['start_seconds']:.2f}s",
        "→",
        f"{segment['end_seconds']:.2f}s",
    )

    print(
        "Transcript:",
        segment["transcript"],
    )

    print(
        "Voice emotion:",
        segment["voice_emotion"]["emotion"],
        "(",
        round(
            segment["voice_emotion"]["confidence"],
            4,
        ),
        ")",
    )

    print(
        "Text sentiment:",
        segment["text_sentiment"]["sentiment"],
        "(",
        round(
            segment["text_sentiment"]["confidence"],
            4,
        ),
        ")",
    )

    print(
        "Agreement:",
        segment["fusion"]["agreement"],
    )

    print(
        "Mismatch:",
        segment["fusion"]["mismatch"],
    )

    print(
        "Combined confidence:",
        round(
            segment["fusion"]["combined_confidence"],
            4,
        ),
    )

    print(
        "Interpretation:",
        segment["fusion"]["interpretation"],
    )
    print("\nEmotional Trend")
print("================")

print(
    "Trend:",
    result["emotional_trend"]["trend"],
)

print(
    "Intensity scores:",
    result["emotional_trend"]["intensity_scores"],
)

print(
    "Intensity change:",
    result["emotional_trend"]["change"],
)


print("\nEscalation")
print("================")

print(
    "Escalation detected:",
    result["escalation"]["escalation_detected"],
)

print(
    "Intensity change:",
    result["escalation"]["intensity_change"],
)

print(
    "Reason:",
    result["escalation"]["reason"],
)