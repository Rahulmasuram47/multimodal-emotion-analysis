from app.audio_processor import (
    load_and_preprocess_audio,
    extract_acoustic_features,
)


audio_path = "sample_audio/real_voice.wav"

# Load and preprocess
audio, sr, duration = load_and_preprocess_audio(audio_path)

# Extract acoustic features
features = extract_acoustic_features(audio, sr)

print("\nAudio Feature Result")
print("--------------------")

print("Duration:", round(duration, 3), "seconds")
print("Sample rate:", sr)
print("Total samples:", features["total_samples"])

print("\nMFCC")
print("Number of coefficients:", features["n_mfcc"])
print("MFCC shape:", features["mfcc_shape"])

print("\nEnergy")
print("RMS mean:", round(features["rms_mean"], 4))
print("RMS max:", round(features["rms_max"], 4))

print("\nPitch")
print("Mean pitch:", round(features["pitch_mean_hz"], 2), "Hz")
print("Minimum pitch:", round(features["pitch_min_hz"], 2), "Hz")
print("Maximum pitch:", round(features["pitch_max_hz"], 2), "Hz")
print("Voiced fraction:", round(features["voiced_fraction"], 4))

print("\nOther Features")
print("ZCR mean:", round(features["zcr_mean"], 4))
print(
    "Spectral centroid:",
    round(features["spectral_centroid_mean"], 2),
    "Hz",
)