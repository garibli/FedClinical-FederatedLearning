"""
Stage 3: Baselines for the FedClinical thesis project.

Two cheap baselines that anchor the results table:

1. Frozen baseline: the same RoBERTa-base with a fresh untrained
   classification head, evaluated with ZERO training. Shows what the
   setup produces without any fine-tuning. Expected: near random (~25%).

2. Classical baseline: TF-IDF features + logistic regression, trained on
   the training data. Shows whether a transformer was needed at all.

Usage:
    python src/stage3_baselines.py
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---------------- settings (must match stage 2) ----------------
SEED = 42
MODEL_NAME = "roberta-base"
LABEL2ID = {"Anxiety": 0, "Depression": 1, "Normal": 2, "Suicidal": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
MAX_LENGTH = 256
BATCH_SIZE = 32

TRAIN_PATH = Path("data/processed/train_full_centralised.csv")
TEST_PATH = Path("data/processed/test_balanced.csv")
RESULTS_DIR = Path("results")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.texts = df["text"].tolist()
        self.labels = [LABEL2ID[s] for s in df["status"].tolist()]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def metrics_dict(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(LABEL2ID))),
        target_names=[ID2LABEL[i] for i in range(len(LABEL2ID))],
        output_dict=True, zero_division=0,
    )
    per_class = {c: report[c]["f1-score"] for c in LABEL2ID}
    return acc, macro, per_class


def frozen_baseline(test_df: pd.DataFrame, device) -> dict:
    """Evaluate RoBERTa with a fresh untrained head. No training at all."""
    print("\n--- Frozen baseline (no training) ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)
    model.eval()

    loader = DataLoader(TextDataset(test_df, tokenizer), batch_size=BATCH_SIZE)
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            labels.extend(batch["labels"].numpy().tolist())

    acc, macro, per_class = metrics_dict(labels, preds)
    print(f"accuracy {acc:.4f}, macro F1 {macro:.4f}")
    for c, f in per_class.items():
        print(f"  {c}: F1 {f:.4f}")
    return {"condition": "frozen_baseline", "accuracy": acc,
            "macro_f1": macro, "per_class": per_class}


def classical_baseline(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """TF-IDF + logistic regression. The 'did we need a transformer' check."""
    print("\n--- Classical baseline (TF-IDF + logistic regression) ---")
    start = time.time()

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                                 sublinear_tf=True)
    x_train = vectorizer.fit_transform(train_df["text"])
    x_test = vectorizer.transform(test_df["text"])
    y_train = [LABEL2ID[s] for s in train_df["status"]]
    y_test = [LABEL2ID[s] for s in test_df["status"]]

    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)

    acc, macro, per_class = metrics_dict(y_test, preds)
    minutes = (time.time() - start) / 60
    print(f"accuracy {acc:.4f}, macro F1 {macro:.4f} "
          f"(took {minutes:.1f} minutes)")
    for c, f in per_class.items():
        print(f"  {c}: F1 {f:.4f}")
    return {"condition": "classical_tfidf_logreg", "accuracy": acc,
            "macro_f1": macro, "per_class": per_class,
            "train_minutes": round(minutes, 1)}


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train rows: {len(train_df)}, test rows: {len(test_df)}")

    results = {
        "frozen": frozen_baseline(test_df, device),
        "classical": classical_baseline(train_df, test_df),
        "seed": SEED,
        "model": MODEL_NAME,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "stage3_baselines.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved to {out_path}")


if __name__ == "__main__":
    main()