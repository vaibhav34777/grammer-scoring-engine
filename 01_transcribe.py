import os
import pandas as pd
from faster_whisper import WhisperModel

TRAIN_DIR = "/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final/train"
TEST_DIR  = "/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final/test"
BASE_DIR  = "/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final"
OUT_DIR   = "/kaggle/working"

model = WhisperModel("large-v3", device="cuda", compute_type="float16")

def transcribe_files(df, audio_dir):
    transcripts = []
    for i, row in df.iterrows():
        audio_path = os.path.join(audio_dir, row["filename"])
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            language="en",
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            temperature=0.0,
        )
        text = " ".join([seg.text.strip() for seg in segments])
        transcripts.append(text)
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(df)}")
    return transcripts

train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))

print("Transcribing train set...")
train_df["transcript"] = transcribe_files(train_df, TRAIN_DIR)
train_df.to_csv(os.path.join(OUT_DIR, "train_transcripts.csv"), index=False)
print("Saved train_transcripts.csv")

print("Transcribing test set...")
test_df["transcript"] = transcribe_files(test_df, TEST_DIR)
test_df.to_csv(os.path.join(OUT_DIR, "test_transcripts.csv"), index=False)
print("Saved test_transcripts.csv")

print("\nSample train transcripts:")
print(train_df[["filename", "label", "transcript"]].head(5).to_string())
