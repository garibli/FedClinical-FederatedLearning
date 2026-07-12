"""
Stage 6a: Local-only baselines. Each client trains on its OWN data alone,
no federation, no privacy. 6 local epochs = the same number of passes over
local data that 6 federated rounds gave, so the only difference left
versus stage 4 is collaboration itself.

Each local model is evaluated on:
  - the client's own local held-out set  -> for "gain from joining"
  - the global balanced test set         -> generalisation of isolated models

Usage:
    python src/stage6_local_baselines.py            # ~1.5 hours
    python src/stage6_local_baselines.py --smoke
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, classification_report

SEED = 42
MODEL_NAME = "roberta-base"
LABEL2ID = {"Anxiety": 0, "Depression": 1, "Normal": 2, "Suicidal": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

MAX_LENGTH = 256
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

NUM_CLIENTS = 5
LOCAL_EPOCHS_TOTAL = 6          # matches 6 rounds x 1 local epoch of federation
LOCAL_HOLDOUT_FRAC = 0.10

DATA_DIR = Path("data/processed")
TEST_PATH = DATA_DIR / "test_balanced.csv"
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
            self.texts[idx], truncation=True, max_length=MAX_LENGTH,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def build_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL2ID),
        id2label=ID2LABEL, label2id=LABEL2ID,
    )
    lora_config = LoraConfig(
        task_type="SEQ_CLS", r=LORA_RANK, lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT, target_modules=["query", "value"],
    )
    return get_peft_model(model, lora_config)


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)).logits
            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            labels.extend(batch["labels"].numpy().tolist())
    acc = accuracy_score(labels, preds)
    macro = f1_score(labels, preds, average="macro")
    report = classification_report(
        labels, preds,
        labels=list(range(len(LABEL2ID))),
        target_names=[ID2LABEL[i] for i in range(len(LABEL2ID))],
        output_dict=True, zero_division=0)
    per_class = {c: report[c]["f1-score"] for c in LABEL2ID}
    return acc, macro, per_class


def train(model, loader, device, epochs):
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE)
    total_steps = len(loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps)
    for _ in range(epochs):
        for batch in loader:
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device))
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    epochs = 1 if args.smoke else LOCAL_EPOCHS_TOTAL
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    test_df = pd.read_csv(TEST_PATH)
    if args.smoke:
        test_df = test_df.sample(n=80, random_state=SEED)
    test_loader = DataLoader(TextDataset(test_df, tokenizer),
                             batch_size=BATCH_SIZE)

    all_results = {}
    start_all = time.time()

    for i in range(NUM_CLIENTS):
        df = pd.read_csv(DATA_DIR / f"client_{i}.csv")
        if args.smoke:
            df = df.sample(n=min(150, len(df)), random_state=SEED)
        # identical deterministic split to stages 4 and 5
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        cut = int(len(df) * (1 - LOCAL_HOLDOUT_FRAC))
        train_df, eval_df = df.iloc[:cut], df.iloc[cut:]

        train_loader = DataLoader(TextDataset(train_df, tokenizer),
                                  batch_size=BATCH_SIZE, shuffle=True)
        local_eval_loader = DataLoader(TextDataset(eval_df, tokenizer),
                                       batch_size=BATCH_SIZE)

        print(f"\n=== client_{i}: training locally on {len(train_df)} rows, "
              f"{epochs} epochs ===")
        set_seed(SEED)  # same init for every client for fairness
        model = build_model().to(device)

        t0 = time.time()
        train(model, train_loader, device, epochs)
        minutes = (time.time() - t0) / 60

        l_acc, l_macro, l_per_class = evaluate(model, local_eval_loader, device)
        g_acc, g_macro, g_per_class = evaluate(model, test_loader, device)
        print(f"client_{i}: local holdout acc {l_acc:.4f}, macro F1 {l_macro:.4f}"
              f" | global test acc {g_acc:.4f}, macro F1 {g_macro:.4f}"
              f" | {minutes:.1f} min")

        all_results[f"client_{i}"] = {
            "train_rows": len(train_df),
            "epochs": epochs,
            "local_holdout": {"accuracy": l_acc, "macro_f1": l_macro,
                              "per_class": l_per_class},
            "global_test": {"accuracy": g_acc, "macro_f1": g_macro,
                            "per_class": g_per_class},
            "train_minutes": round(minutes, 1),
        }

        if not args.smoke:
            adapter_path = RESULTS_DIR / "adapters" / f"local_only_client_{i}.pt"
            adapter_path.parent.mkdir(parents=True, exist_ok=True)
            state = {name: p.detach().cpu().clone()
                     for name, p in model.named_parameters() if p.requires_grad}
            torch.save(state, adapter_path)

        del model
        torch.cuda.empty_cache()

    total_minutes = (time.time() - start_all) / 60
    print(f"\nAll local baselines finished in {total_minutes:.1f} minutes")

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    out = {"condition": "local_only_baselines", "run": tag,
           "clients": all_results, "seed": SEED,
           "total_minutes": round(total_minutes, 1)}
    out_path = RESULTS_DIR / f"stage6_local_baselines_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Metrics saved to {out_path}")


if __name__ == "__main__":
    main()