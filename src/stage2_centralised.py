"""
Stage 2: Centralised baseline training for the FedClinical thesis project.

What this script does:
1. Loads the cleaned training data and the balanced test set from stage 1.
2. Fine-tunes RoBERTa-base with LoRA adapters on the full training data.
3. Evaluates on the balanced test set: accuracy, macro F1, per-class F1.
4. Saves metrics to results/ and the trained LoRA adapter to results/adapters/.

This produces the centralised upper bound, one of the four thesis conditions.

Usage:
    python src/stage2_centralised.py            # full run (use GPU / Colab)
    python src/stage2_centralised.py --smoke    # tiny run to verify the code works

The label mapping below is THE single source of truth for the whole project.
Every later stage (federated, DP) must import or copy it unchanged.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---------------- settings ----------------
SEED = 42
MODEL_NAME = "roberta-base"
LABEL2ID = {"Anxiety": 0, "Depression": 1, "Normal": 2, "Suicidal": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

MAX_LENGTH = 256          # tokens per text; covers the vast majority of posts
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 2e-4      # LoRA trains small adapters, so a higher lr is normal
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

TRAIN_PATH = Path("data/processed/train_full_centralised.csv")
TEST_PATH = Path("data/processed/test_balanced.csv")
RESULTS_DIR = Path("results")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    """Wraps a dataframe so PyTorch can read it in batches."""

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


def build_model():
    """Load RoBERTa-base and wrap it with LoRA adapters.

    The base model stays frozen. Only the small LoRA matrices train,
    which is what makes this cheap enough for a student budget and,
    later, small enough to send between federated clients.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    lora_config = LoraConfig(
        task_type="SEQ_CLS",
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["query", "value"],  # attention layers of RoBERTa
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def evaluate(model, loader, device):
    """Run the model on a dataloader and return metrics."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch["labels"].numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(
        all_labels, all_preds,
        labels=list(range(len(LABEL2ID))),
        target_names=[ID2LABEL[i] for i in range(len(LABEL2ID))],
        output_dict=True, zero_division=0,
    )
    return acc, macro_f1, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="tiny fast run to check the code works")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    if args.smoke:
        print("SMOKE TEST MODE: tiny subset, one epoch, results are meaningless")
        train_df = train_df.sample(n=200, random_state=SEED)
        test_df = test_df.sample(n=80, random_state=SEED)

    print(f"Training rows: {len(train_df)}, test rows: {len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_loader = DataLoader(TextDataset(train_df, tokenizer),
                              batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TextDataset(test_df, tokenizer),
                             batch_size=BATCH_SIZE)

    model = build_model().to(device)

    epochs = 1 if args.smoke else EPOCHS
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    start = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels)
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += out.loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(train_loader)} "
                      f"loss {running_loss / (step + 1):.4f}")

    train_minutes = (time.time() - start) / 60
    print(f"\nTraining finished in {train_minutes:.1f} minutes")

    acc, macro_f1, report = evaluate(model, test_loader, device)
    print(f"\nTest accuracy: {acc:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")
    for cls in LABEL2ID:
        print(f"  {cls}: F1 {report[cls]['f1-score']:.4f}")

    # save everything
    RESULTS_DIR.mkdir(exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    metrics = {
        "condition": "centralised",
        "run": tag,
        "model": MODEL_NAME,
        "epochs": epochs,
        "train_rows": len(train_df),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class": {c: report[c]["f1-score"] for c in LABEL2ID},
        "train_minutes": round(train_minutes, 1),
        "seed": SEED,
    }
    out_path = RESULTS_DIR / f"stage2_centralised_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {out_path}")

    if not args.smoke:
        adapter_dir = RESULTS_DIR / "adapters" / "centralised"
        model.save_pretrained(adapter_dir)
        print(f"LoRA adapter saved to {adapter_dir}")


if __name__ == "__main__":
    main()