import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

TRAIN_AUDIO_DIR = "/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final/train"
TEST_AUDIO_DIR  = "/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final/test"

TRAIN_TRANSCRIPTS = "/kaggle/working/train_transcripts.csv"
TEST_TRANSCRIPTS  = "/kaggle/working/test_transcripts.csv"
OUT_DIR           = "/kaggle/working"

def extract_features(audio_path, transcript):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
    except Exception as e:
        # Fallback for completely corrupted files
        return {
            "duration": 0.0, "rms_mean": 0.0, "rms_std": 0.0, 
            "zcr_mean": 0.0, "silence_ratio": 1.0, "wpm": 0.0, "word_count": 0
        }
    
    # 1. Duration
    duration = librosa.get_duration(y=y, sr=sr)
    if duration == 0:
        duration = 0.001
        
    # 2. RMS Energy (Loudness proxy)
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    
    # 3. Zero Crossing Rate (Voice vs unvoiced proxy)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_mean = np.mean(zcr)
    
    # 4. Silence Ratio
    # Split audio into non-silent intervals. top_db=30 is a standard threshold.
    intervals = librosa.effects.split(y, top_db=30)
    non_silent_duration = sum([(end - start) for start, end in intervals]) / sr
    silence_ratio = 1.0 - (non_silent_duration / duration)
    silence_ratio = max(0.0, min(1.0, silence_ratio))
    
    # 5. Words Per Minute (WPM)
    words = str(transcript).split()
    word_count = len(words)
    wpm = word_count / (duration / 60.0)
    
    return {
        "duration": duration,
        "rms_mean": rms_mean,
        "rms_std": rms_std,
        "zcr_mean": zcr_mean,
        "silence_ratio": silence_ratio,
        "word_count": word_count,
        "wpm": wpm
    }

def process_dataset(csv_path, audio_dir, out_filename):
    df = pd.read_csv(csv_path)
    features = []
    
    print(f"Extracting features for {out_filename}...")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        filename = row["filename"]
        transcript = row.get("transcript", "")
        audio_path = os.path.join(audio_dir, filename)
        
        feats = extract_features(audio_path, transcript)
        feats["filename"] = filename
        features.append(feats)
        
    feats_df = pd.DataFrame(features)
    # Merge with original transcripts to keep labels and transcripts
    final_df = pd.merge(df, feats_df, on="filename", how="left")
    
    out_path = os.path.join(OUT_DIR, out_filename)
    final_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}\n")

process_dataset(TRAIN_TRANSCRIPTS, TRAIN_AUDIO_DIR, "train_audio_features.csv")
process_dataset(TEST_TRANSCRIPTS, TEST_AUDIO_DIR, "test_audio_features.csv")
print("Acoustic feature extraction complete!")
