import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = "/kaggle/working"

# Load audio features
train_df = pd.read_csv(os.path.join(OUT_DIR, "train_audio_features.csv"))
test_df  = pd.read_csv(os.path.join(OUT_DIR, "test_audio_features.csv"))

# Load DeBERTa predictions
# train_oof_predictions.csv should contain 'filename', 'label', 'oof_pred'
deberta_train = pd.read_csv(os.path.join(OUT_DIR, "train_oof_predictions.csv"))
# The DeBERTa test predictions are stored in submission.csv under 'label'
deberta_test = pd.read_csv(os.path.join(OUT_DIR, "submission.csv"))
deberta_test = deberta_test.rename(columns={"label": "deberta_pred"})

# Merge DeBERTa predictions into our feature dataframes
train_df = pd.merge(train_df, deberta_train[["filename", "oof_pred"]], on="filename", how="left")
test_df  = pd.merge(test_df, deberta_test[["filename", "deberta_pred"]], on="filename", how="left")
# Rename for consistency
test_df = test_df.rename(columns={"deberta_pred": "oof_pred"})

# Define features to use
FEATURES = [
    "oof_pred",        # DeBERTa's text-based prediction (Most important)
    "duration",        # Total audio length
    "rms_mean",        # Average volume/energy
    "rms_std",         # Volume variation
    "zcr_mean",        # Voice character
    "silence_ratio",   # % of audio that is silent pauses
    "word_count",      # Total words spoken
    "wpm"              # Words per minute (fluency proxy)
]

TARGET = "label"

# We will apply a rule-based override for complete silence (0.0 labels)
def override_silence(df, preds):
    # If the audio has very low energy or 0 words, snap prediction to 0.0
    silence_mask = (df["word_count"] == 0) | (df["rms_mean"] < 0.001) | (df["silence_ratio"] > 0.95)
    preds = np.where(silence_mask, 0.0, preds)
    return preds

X = train_df[FEATURES]
y = train_df[TARGET]
X_test = test_df[FEATURES]

N_FOLDS = 5
SEED = 42
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.03,
    'num_leaves': 15,
    'max_depth': 4,
    'feature_fraction': 0.8,
    'random_state': SEED,
    'n_estimators': 500,
    'verbose': -1
}

print("Training LightGBM on Audio + DeBERTa Stacking...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(**lgb_params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    val_pred = model.predict(X_val)
    oof_preds[val_idx] = val_pred
    test_preds += model.predict(X_test) / N_FOLDS
    
    val_rmse = np.sqrt(np.mean((val_pred - y_val) ** 2))
    print(f"Fold {fold+1} | Best Iteration: {model.best_iteration_} | Val RMSE: {val_rmse:.4f}")

# Apply 0.0 post-processing logic
oof_preds = override_silence(train_df, oof_preds)
test_preds = override_silence(test_df, test_preds)

# Clip final predictions
oof_preds = np.clip(oof_preds, 0.0, 5.0)
test_preds = np.clip(test_preds, 0.0, 5.0)

final_rmse = np.sqrt(np.mean((oof_preds - y) ** 2))
final_pearson = pearsonr(y, oof_preds)[0]

print("\n" + "="*40)
print(f"Final Stacking OOF RMSE:    {final_rmse:.4f}")
print(f"Final Stacking OOF Pearson: {final_pearson:.4f}")
print("="*40)

# Feature importance
feature_imp = pd.DataFrame({'Feature': FEATURES, 'Importance': model.feature_importances_})
feature_imp = feature_imp.sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(feature_imp.to_string(index=False))

# Create final submission
submission = test_df[["filename"]].copy()
submission["label"] = test_preds
submission.to_csv(os.path.join(OUT_DIR, "submission_lgb.csv"), index=False)
print("\nFinal submission saved to 'submission_lgb.csv'")
