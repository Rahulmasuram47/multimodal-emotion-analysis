import os
import sys
import shutil
import numpy as np
import soundfile as sf
import librosa

def run_milestone1_verification():
    print('==============================================================')
    print('       MILESTONE 1: AUDIO SETUP & VERIFICATION TEST       ')
    print('============================================================')
    
    # 1. Environment & Python details
    print(f'[INFO] Python Version: {sys.version}')
    print(f'[INFO] Python Executable: {sys.executable}')
    print(f'[INFO] NumPy Version: {np.__version__}')
    print(f'[INFO] Librosa Version: {librosa.__version__}')
    print(f'[INFO] SoundFile Version: {sf.__version__}')
    
    # 2. ffmpeg check
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        print(f'[INFO] ffmpeg binary found at: {ffmpeg_path}')
    else:
        print('[WARN] ffmpeg is not currently on PATH. (Standard WAV/FLAC audio files via soundfile are supported directly without ffmpeg).')
        
    # 3. Audio File Verification
    sample_file = os.path.join('sample_audio', 'sample_16k.wav')
    if not os.path.exists(sample_file):
        raise FileNotFoundError(f'Sample audio file not found at: {sample_file}')
        
    print(f'[INFO] Loading audio file: {sample_file}')
    
    # Read using soundfile
    data, sr = sf.read(sample_file)
    duration_sf = len(data) / sr
    print(f'[INFO] SoundFile read: sample_rate = {sr} Hz, total_samples = {len(data)}, duration = {duration_sf:.3f} s')
    
    # Read using librosa (standard resampled loading test)
    y, target_sr = librosa.load(sample_file, sr=16000)
    duration_librosa = librosa.get_duration(y=y, sr=target_sr)
    print(f'[INFO] Librosa load: target_sr = {target_sr} Hz, audio_shape = {y.shape}, duration = {duration_librosa:.3f} s')
    
    # 4. Extract Acoustic Features (MFCC)
    n_mfcc = 13
    mfccs = librosa.feature.mfcc(y=y, sr=target_sr,
                                 n_mfcc=n_mfcc)
    print(f'[INFO] MFCC Features extracted successfully!')
    print(f'[INFO] MFCC Matrix Shape: {mfccs.shape} (n_mfcc={mfccs.shape[0]}, time_frames={mfccs.shape[1]})')
    print(f'[INFO] MFCC Mean: {np.mean(mfccs):.4f}, Std: {np.std(mfccs):.4f}')
    
    # 5. Assertions / Validation
    assert len(y) > 0, 'Audio signal must not be empty'
    assert target_sr == 16000, f'Expected 16000 Hz, got {target_sr}'
    assert mfccs.shape[0] == n_mfcc, f'Expected {n_mfcc} MFCC coefficients, got {mfccs.shape[0]}'
    assert mfccs.shape[1] > 0, 'MFCC time frames must be > 0'
    
    print('=============================================================')
    print('   [SUCCESS] Milestone 1 verification tests passed!       ')
    print('=============================================================')

def test_milestone1_verification():
    run_milestone1_verification()


if __name__ == '__main__':
    run_milestone1_verification()
