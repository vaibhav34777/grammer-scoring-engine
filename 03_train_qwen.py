import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

TRAIN_PATH = "/kaggle/working/train_transcripts.csv"
TEST_PATH  = "/kaggle/working/test_transcripts.csv"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUT_DIR    = "/kaggle/working/qwen_lora"
SEED       = 42
MAX_LEN    = 1024
EPOCHS     = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR         = 2e-4

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)

SYSTEM_PROMPT = """You are an expert English grammar evaluator. Given a spoken English transcript, assign a grammar score using the rubric below.

Rubric:
0   - Silent or completely unintelligible.
1   - Struggles with sentence structure; very limited grammatical control; relies on memorized patterns.
1.5 - Between 1 and 2.
2   - Limited understanding of structure; consistent basic mistakes; sentences may be incomplete.
2.5 - Between 2 and 3.
3   - Decent structure but grammatical errors, OR decent grammar but structural/syntax errors.
3.5 - Between 3 and 4.
4   - Strong grammar and structure; occasional minor errors that do not cause misunderstanding.
4.5 - Between 4 and 5.
5   - High grammatical accuracy; controls complex grammar; seldom makes noticeable mistakes.

Respond with ONLY a single number from: 0, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5"""

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def build_prompt_only(transcript):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Transcript:\n{transcript}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

class QwenGrammarDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=MAX_LEN):
        self.samples = []
        for _, row in df.iterrows():
            prompt_str = build_prompt_only(row["transcript"])
            target_str = f"{row['label']}<|im_end|>\n"
            
            prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
            target_ids = tokenizer.encode(target_str, add_special_tokens=False)
            
            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]
                
            attention_mask = [1] * len(input_ids)
            
            self.samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)
train_df["transcript"] = train_df["transcript"].fillna("").astype(str)
test_df["transcript"]  = test_df["transcript"].fillna("").astype(str)

train_split, val_split = train_test_split(train_df, test_size=0.15, random_state=SEED, shuffle=True)
train_split = train_split.reset_index(drop=True)
val_split   = val_split.reset_index(drop=True)

train_dataset = QwenGrammarDataset(train_split, tokenizer)
val_dataset   = QwenGrammarDataset(val_split, tokenizer)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    pad_to_multiple_of=8,
    return_tensors="pt"
)

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,
    seed=SEED,
    report_to="none",
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(os.path.join(OUT_DIR, "best_model"))
print("Model saved.")

VALID_SCORES = sorted([0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

def parse_score(generated_text):
    text = generated_text.strip()
    match = re.search(r'\b(\d+\.?\d*)\b', text)
    if match:
        val = float(match.group(1))
        closest = min(VALID_SCORES, key=lambda x: abs(x - val))
        return float(np.clip(closest, 0.0, 5.0))
    return 3.0

def predict_scores(texts, batch_size=4):
    model.eval()
    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        for j in range(len(batch_texts)):
            input_len = inputs["input_ids"].shape[1]
            new_tokens = outputs[j][input_len:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
            all_preds.append(parse_score(decoded))
        if (i // batch_size + 1) % 10 == 0:
            print(f"  Predicted {i + batch_size}/{len(texts)}")
    return np.array(all_preds)

print("\nRunning validation inference...")
val_inference_texts = [build_prompt_only(t) for t in val_split["transcript"].tolist()]
val_preds  = predict_scores(val_inference_texts)
val_labels = val_split["label"].tolist()

val_rmse    = np.sqrt(np.mean((val_preds - np.array(val_labels)) ** 2))
val_pearson = pearsonr(val_labels, val_preds)[0]
print(f"Val RMSE:    {val_rmse:.4f}")
print(f"Val Pearson: {val_pearson:.4f}")

print("\nRunning test inference...")
test_inference_texts = [build_prompt_only(t) for t in test_df["transcript"].tolist()]
test_preds = predict_scores(test_inference_texts)

submission = test_df[["filename"]].copy()
submission["label"] = test_preds
submission.to_csv(os.path.join(OUT_DIR, "submission.csv"), index=False)
print("\nsubmission.csv saved.")
print(submission.head(10))
