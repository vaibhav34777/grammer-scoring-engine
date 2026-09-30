import os
import pandas as pd
import librosa
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────
TRAIN_DIR  = "/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final/train"
TEST_DIR   = "/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final/test"
BASE_DIR   = "/kaggle/input/competitions/shl-hiring-assessment-2026/Dataset_Final"

# ── 1. List all CSVs at the base level ────────────────────────────────────
print("=== Files in BASE_DIR ===")
for f in os.listdir(BASE_DIR):
    print(f"  {f}")

# ── 2. Load and inspect train.csv ─────────────────────────────────────────
train_csv_path = os.path.join(BASE_DIR, "train.csv")
train_df = pd.read_csv(train_csv_path)
print("\n=== train.csv shape:", train_df.shape)
print("=== train.csv columns:", train_df.columns.tolist())
print("=== train.csv head:\n", train_df.head(10))
print("=== train.csv dtypes:\n", train_df.dtypes)

# ── 3. Label distribution ─────────────────────────────────────────────────
label_col = train_df.columns[-1]  # assumes label is last col
print(f"\n=== Label column: '{label_col}'")
print("  Value counts:\n", train_df[label_col].value_counts().sort_index())
print("  Unique values:", sorted(train_df[label_col].unique()))
print(f"  Min={train_df[label_col].min()}, Max={train_df[label_col].max()}, "
      f"Mean={train_df[label_col].mean():.3f}, Std={train_df[label_col].std():.3f}")

# ── 4. Load and inspect test.csv ──────────────────────────────────────────
test_csv_path = os.path.join(BASE_DIR, "test.csv")
test_df = pd.read_csv(test_csv_path)
print("\n=== test.csv shape:", test_df.shape)
print("=== test.csv columns:", test_df.columns.tolist())
print("=== test.csv head:\n", test_df.head(5))

# ── 5. Check audio files in train directory ───────────────────────────────
train_audio_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith(".wav")]
print(f"\n=== Number of .wav files in TRAIN_DIR: {len(train_audio_files)}")
print("=== First 5 audio files:", train_audio_files[:5])

# ── 6. Check a sample audio file's properties ─────────────────────────────
sample_file = os.path.join(TRAIN_DIR, train_audio_files[0])
y, sr = librosa.load(sample_file, sr=None)  # sr=None to preserve original sample rate
duration = librosa.get_duration(y=y, sr=sr)
print(f"\n=== Sample audio: {train_audio_files[0]}")
print(f"  Sample rate: {sr} Hz")
print(f"  Duration:    {duration:.2f} seconds")
print(f"  Shape:       {y.shape}")
print(f"  Channels:    {'mono' if y.ndim == 1 else 'stereo'}")

# ── 7. Quick duration stats across a subset of files (first 20) ───────────
print("\n=== Duration stats on first 20 train files ===")
durations = []
for f in train_audio_files[:20]:
    yy, ssr = librosa.load(os.path.join(TRAIN_DIR, f), sr=None)
    durations.append(librosa.get_duration(y=yy, sr=ssr))
print(f"  Min: {np.min(durations):.2f}s  Max: {np.max(durations):.2f}s  "
      f"Mean: {np.mean(durations):.2f}s")

# ── 8. Check if all train.csv filenames exist in TRAIN_DIR ────────────────
filename_col = train_df.columns[0]  # assumes filename is first col
print(f"\n=== Filename column: '{filename_col}'")
missing = [f for f in train_df[filename_col] if not os.path.exists(os.path.join(TRAIN_DIR, f))]
print(f"  Missing files in TRAIN_DIR: {len(missing)}")
if missing:
    print("  First 5 missing:", missing[:5])

print("\n✅ Exploration complete.")
