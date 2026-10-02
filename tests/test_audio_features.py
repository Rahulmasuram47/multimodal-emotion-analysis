from app.audio_processor import (
    load_and_preprocess_audio,
    extract_acoustic_features,
)


def test_extract_acoustic_features():
    audio_path = "sample_audio/real_voice.wav"
    audio, sr, duration = load_and_preprocess_audio(audio_path)
    features = extract_acoustic_features(audio, sr)

    assert duration > 0
    assert sr == 16000
    assert features["total_samples"] > 0
    assert features["n_mfcc"] == 13
    assert features["rms_mean"] > 0