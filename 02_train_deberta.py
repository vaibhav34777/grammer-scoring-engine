import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

TRAIN_PATH = "/kaggle/input/datasets/vaibhav1908/transcripts/train_transcripts.csv"
TEST_PATH  = "/kaggle/input/datasets/vaibhav1908/transcripts/test_transcripts.csv"
MODEL_NAME = "microsoft/deberta-v3-base"
OUT_DIR    = "/kaggle/working/"
MAX_LEN    = 512
BATCH_SIZE = 8
EPOCHS     = 10
LR_ENCODER = 1e-5
LR_HEAD    = 1e-4
EARLY_STOPPING = 5
N_FOLDS    = 5
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

train_df["transcript"] = train_df["transcript"].fillna("").astype(str)
test_df["transcript"]  = test_df["transcript"].fillna("").astype(str)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class GrammarDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.texts  = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
        return item

def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def compute_pearson(y_true, y_pred):
    return pearsonr(np.array(y_true), np.array(y_pred))[0]

def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass in full fp32 (No AMP to prevent NaNs)
        logits = model(**batch).logits.squeeze(-1)
        
        # Simple MSE Regression
        loss = F.mse_loss(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
    return total_loss / len(loader)

def predict(model, loader, device):
    model.eval()
    preds = []
    
    with torch.no_grad():
        for batch in loader:
            batch.pop("labels", None)
            batch  = {k: v.to(device) for k, v in batch.items()}
            
            # Full fp32 inference
            logits = model(**batch).logits.squeeze(-1)
            preds.extend(logits.cpu().numpy().tolist())
            
    return np.array(preds)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

texts      = train_df["transcript"].tolist()
labels     = train_df["label"].tolist()
test_texts = test_df["transcript"].tolist()

oof_preds  = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
    print(f"\n{'='*50}")
    print(f"Fold {fold + 1} / {N_FOLDS}")
    print(f"{'='*50}")

    train_texts  = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts    = [texts[i] for i in val_idx]
    val_labels   = [labels[i] for i in val_idx]

    train_ds = GrammarDataset(train_texts, train_labels)
    val_ds   = GrammarDataset(val_texts,   val_labels)
    test_ds  = GrammarDataset(test_texts)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Simple Regression Head
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=1,
        problem_type="regression",
        ignore_mismatched_sizes=True,
    )
    model = model.to(device).float()

    # Differential Learning Rates
    optimizer = torch.optim.AdamW([
        {"params": model.deberta.parameters(),    "lr": LR_ENCODER},
        {"params": model.classifier.parameters(), "lr": LR_HEAD},
    ], weight_decay=0.01)
    
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * 0.1)
    scheduler    = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_val_rmse    = float("inf")
    best_val_preds   = None
    best_test_preds  = None
    patience_counter = 0

    for epoch in range(EPOCHS):
        train_loss  = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        val_preds   = predict(model, val_loader, device)
        val_preds_c = np.clip(val_preds, 0.0, 5.0)
        
        val_rmse    = compute_rmse(val_labels, val_preds_c)
        val_pearson = compute_pearson(val_labels, val_preds_c)
        print(f"  Epoch {epoch + 1}/{EPOCHS}  loss={train_loss:.4f}  val_rmse={val_rmse:.4f}  val_pearson={val_pearson:.4f}")

        if val_rmse < best_val_rmse:
            best_val_rmse    = val_rmse
            best_val_preds   = val_preds
            best_test_preds  = predict(model, test_loader, device)
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"deberta_fold{fold + 1}.pt"))
            print(f"    >> Best model saved (val_rmse={best_val_rmse:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING:
                print(f"    >> Early stopping at epoch {epoch + 1}")
                break

    oof_preds[val_idx] = best_val_preds
    test_preds        += best_test_preds / N_FOLDS

    del model
    torch.cuda.empty_cache()

oof_preds  = np.clip(oof_preds,  0.0, 5.0)
test_preds = np.clip(test_preds, 0.0, 5.0)

overall_rmse    = compute_rmse(labels, oof_preds)
overall_pearson = compute_pearson(labels, oof_preds)

print(f"\n{'='*50}")
print(f"OOF RMSE:    {overall_rmse:.4f}")
print(f"OOF Pearson: {overall_pearson:.4f}")
print(f"{'='*50}")

train_df["oof_pred"] = oof_preds
train_df.to_csv(os.path.join(OUT_DIR, "train_oof_predictions.csv"), index=False)

submission = test_df[["filename"]].copy()
submission["label"] = test_preds
submission.to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False)

print("\nsubmission.csv saved.")
print(submission.head(10))
