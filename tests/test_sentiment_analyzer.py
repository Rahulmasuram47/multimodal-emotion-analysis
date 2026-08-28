from app.sentiment_analyzer import SentimentAnalyzer


TEXT = (
    "Hello, this is a test recording for multi-modal emotion "
    "analysis project and speaking normally and testing the "
    "audio emotion system."
)


print("Loading sentiment model...")

analyzer = SentimentAnalyzer()

print("\nAnalyzing text...")

result = analyzer.analyze(TEXT)

print("\nText Sentiment Result")
print("---------------------")

print("Sentiment:", result["sentiment"])
print("Confidence:", round(result["confidence"], 4))

print("\nAll Scores:")

for sentiment, score in result["scores"].items():
    print(f"{sentiment}: {score:.4f}")