from app.fusion import MultimodalFusion


fusion = MultimodalFusion()


print("\n=== Test 1: Agreement ===")

audio_result = {
    "emotion": "angry",
    "confidence": 0.90,
}

text_result = {
    "sentiment": "negative",
    "confidence": 0.85,
}

result = fusion.fuse(audio_result, text_result)

print(result)


print("\n=== Test 2: Deliberate Mismatch ===")

audio_result = {
    "emotion": "angry",
    "confidence": 0.90,
}

text_result = {
    "sentiment": "positive",
    "confidence": 0.85,
}

result = fusion.fuse(audio_result, text_result)

print(result)


print("\n=== Test 3: Neutral Agreement ===")

audio_result = {
    "emotion": "neutral",
    "confidence": 0.80,
}

text_result = {
    "sentiment": "neutral",
    "confidence": 0.75,
}

result = fusion.fuse(audio_result, text_result)

print(result)