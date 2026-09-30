# Spoken Grammar Scoring Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97-Transformers-yellow)](https://huggingface.co/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-brightgreen.svg)](https://lightgbm.readthedocs.io/)

> **SHL Research Engineer Assessment**  
> An end-to-end multimodal machine learning pipeline for automated grammar scoring on spoken audio samples (MOS Likert scale: 0.0 to 5.0).

---

## 📌 Executive Summary & Results

Automated assessment of spoken grammar cannot rely on text transcripts alone: modern speech-to-text models clean up disfluencies, reconstruct sentence boundaries, and strip acoustic cues (pauses, hesitations, articulation rate). 

This solution uses a **Multimodal Stacking Architecture** combining:
1. **Raw-preserving ASR (`faster-whisper-large-v3`)** with disabled context-conditioning to retain verbatim speaker errors.
2. **5-Fold Cross-Validated `DeBERTa-v3-base`** fine-tuned on spoken transcripts using differential learning rates.
3. **Acoustic & Prosodic Feature Extraction** (`librosa`) capturing fluency, silence intervals, energy variability, and speech tempo (WPM).
4. **LightGBM Gradient Boosted Stacking** to fuse linguistic predictions with acoustic prosodic signals.

### 🏆 Benchmark Comparison

| Stage / Architecture | Test RMSE | OOF Pearson (r) | Key Takeaway |
| :--- | :---: | :---: | :--- |
| Baseline (Dataset Mean Predictor) | ~1.2390 | 0.0000 | Trivial reference baseline |
| DeBERTa-v3-base (Uniform LR) | ~0.5300 | 0.7420 | Text-only regression baseline |
| DeBERTa-v3-base (Diff LR + 5-Fold Ensemble) | ~0.5045 | 0.7910 | Enhanced textual syntactic representations |
| **Multimodal Stacking (DeBERTa + Acoustic + LightGBM)** | **0.4802** | **0.8274** | **Best performance; acoustic features bridge fluency gap** |

---

## 🏛️ Pipeline Architecture

```
                                  ┌───────────────────────────────┐
                                  │   Input Audio (.wav 45-60s)   │
                                  └───────────────┬───────────────┘
                                                  │
                         ┌────────────────────────┴────────────────────────┐
                         ▼                                                 ▼
        ┌───────────────────────────────────┐             ┌───────────────────────────────────┐
        │       ASR Transcription Stage     │             │     Acoustic Feature Extraction   │
        │    faster-whisper (large-v3)      │             │              (librosa)            │
        │  • condition_on_prev_text=False   │             │  • Words Per Minute (WPM)         │
        │  • verbatim error preservation    │             │  • Silence Duration Ratio         │
        └────────────────┬──────────────────┘             │  • RMS Energy (Mean / Std)        │
                         │                                │  • Zero Crossing Rate (ZCR)       │
                         ▼                                │  • Audio Duration & Word Count    │
        ┌───────────────────────────────────┐             └─────────────────┬─────────────────┘
        │   Syntactic / Text Modeling       │                               │
        │     DeBERTa-v3-base (5-Fold CV)   │                               │
        │  • Differential LR (Head: 1e-4)   │                               │
        │  • Encoder LR: 1e-5               │                               │
        └────────────────┬──────────────────┘                               │
                         │                                                  │
                         │  [OOF Predictions & Test Probabilities]          │
                         └────────────────────────┬─────────────────────────┘
                                                  ▼
                                 ┌───────────────────────────────────┐
                                 │       LightGBM Stacking Model     │
                                 │  • Non-linear multimodal fusion   │
                                 │  • 5-Fold Cross-Validation        │
                                 └────────────────┬──────────────────┘
                                                  ▼
                                 ┌───────────────────────────────────┐
                                 │  Continuous Grammar Score [0 - 5] │
                                 └───────────────────────────────────┘
```

---

## 📊 Feature Importance & Multimodal Insights

The LightGBM feature attribution confirms that acoustic signals are critical contributors to the final grammar score:

```
      Feature       Importance  Description
─────────────────────────────────────────────────────────────────────────────
     oof_pred           263     DeBERTa text-based syntactic/grammar score
     zcr_mean           205     Zero Crossing Rate (spectral / voice texture)
     rms_mean           173     Average speech volume & energy consistency
silence_ratio           118     Percentage of clip spent in silent hesitation
   word_count           115     Total spoken vocabulary volume in timeframe
      rms_std            93     Dynamic vocal range / confidence variation
          wpm            73     Words Per Minute (speaking tempo / fluency)
     duration            57     Valid audio duration
```

---

## 📂 Project Structure

```
shl-grammar-scoring/
├── README.md                     # Project overview and reproduction guide
├── requirements.txt              # Environment dependencies
├── competition_details.md        # Rubric and competition guidelines
├── 00_data_exploration.py        # EDA: Dataset distribution & audio properties
├── 01_transcribe.py              # Error-preserving ASR using Faster-Whisper
├── 02_train_deberta.py           # 5-fold DeBERTa-v3 fine-tuning
├── 03_train_qwen.py              # LLM QLoRA baseline experiment (Qwen2.5-3B)
├── 04_extract_audio_features.py  # Acoustic prosodic feature extraction (Librosa)
├── 05_train_lightgbm.py          # Multimodal LightGBM stacking & final submission
```

---

## 🚀 Reproduction Guide

### 1. Environment Setup
```bash
git clone <your-repo-url>
cd shl-grammar-scoring
pip install -r requirements.txt
```

### 2. Step-by-Step Execution

#### Step 0: Data Exploration
Inspect dataset shapes, missing files, duration distributions, and label balance:
```bash
python 00_data_exploration.py
```

#### Step 1: Speech-to-Text Transcription
Transcribe raw `.wav` files to verbatim transcripts using Faster-Whisper Large-v3:
```bash
python 01_transcribe.py
```
*Outputs: `train_transcripts.csv`, `test_transcripts.csv`*

#### Step 2: Fine-Tune DeBERTa-v3
Train 5-fold cross-validated DeBERTa-v3 with differential learning rates:
```bash
python 02_train_deberta.py
```
*Outputs: `train_oof_predictions.csv`, `submission.csv`, `deberta_fold*.pt`*

#### Step 3: Extract Acoustic Prosody Features
Compute WPM, silence ratios, RMS energy, and zero-crossing rates:
```bash
python 04_extract_audio_features.py
```
*Outputs: `train_audio_features.csv`, `test_audio_features.csv`*

#### Step 4: LightGBM Multimodal Stacking
Train the stacking regressor combining textual OOFs and acoustic features:
```bash
python 05_train_lightgbm.py
```
*Outputs: `submission_lgb.csv` (Final test predictions with **RMSE 0.4802**)*

---

## 📝 Scoring Rubric Reference

| Grammar Score | Official Description |
| :---: | :--- |
| **1.0** | Speech struggles with proper sentence structure; limited control over simple grammatical structures; relies on memorized patterns. |
| **2.0** | Limited understanding of sentence syntax; basic grammatical mistakes; sentences often left incomplete. |
| **3.0** | Decent grasp of sentence structure with grammatical errors, OR decent grammatical grasp with structural/syntax errors. |
| **4.0** | Strong understanding of grammar and syntax; minor occasional errors that do not impair understanding. |
| **5.0** | High grammatical accuracy; adept control of complex language structures; seldom makes noticeable mistakes. |

---

## 🛠️ Key Design Decisions & Research Learnings

1. **ASR Configuration:** Disabled `condition_on_previous_text` in Faster-Whisper. Enabling it causes Whisper to auto-correct ungrammatical phrases using language model priors, defeating downstream grammar evaluation.
2. **Differential Learning Rates:** Setting `LR_HEAD = 1e-4` (10x higher than `LR_ENCODER = 1e-5`) accelerated convergence on randomly initialized linear heads without distorting pretrained language weights.
3. **Continuous Probability Expectations:** Ground-truth MOS Likert scores represent annotator averages. Continuous outputs strictly outperform snapped discrete predictions under RMSE evaluation.
4. **Multimodal Synergy:** Text models excel at morphological and syntactic rules, while audio prosody models capture articulation fluency and disfluency pauses, creating a complementary ensemble.
