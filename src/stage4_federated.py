"""
Stage 4: Federated training (FedAvg, no privacy) for the FedClinical thesis.

What this script does:
1. Loads the 5 client datasets from stage 1. Each client keeps 10% of its
   own data as a local held-out set (deterministic, seed 42).
2. Runs FedAvg: each round, every client receives the global trainable
   parameters (LoRA adapters + classifier head), trains locally, and sends
   its updated parameters back. The server averages them weighted by
   client dataset size.
3. Evaluates the global model on the balanced test set after every round,
   so convergence over rounds is visible.
4. At the end, evaluates the final global model on each client's local
   held-out set (the per-client view, research question 2).
5. Tracks communication cost: bytes sent per round and in total.
6. Saves metrics json and the final federated adapter state.

Usage:
    python src/stage4_federated.py            # full run
    python src/stage4_federated.py --smoke    # tiny verification run
"""

import argparse
import copy
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

# ---------------- settings (must match stage 2) ----------------
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
ROUNDS = 6                # federated rounds
LOCAL_EPOCHS = 1          # epochs each client trains per round
LOCAL_HOLDOUT_FRAC = 0.10 # share of each client's data kept for local eval

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


# ---------- the three federated primitives ----------

def get_trainable_state(model) -> dict:
    """Copy of every trainable parameter (LoRA + classifier head).
    This is exactly what travels between server and clients."""
    return {name: p.detach().cpu().clone()
            for name, p in model.named_parameters() if p.requires_grad}


def set_trainable_state(model, state: dict) -> None:
    """Load global parameters into the model before local training."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(state[name].to(p.device))


def fedavg(states: list[dict], weights: list[int]) -> dict:
    """Weighted average of client parameter states. This is FedAvg."""
    total = sum(weights)
    avg = {}
    for name in states[0]:
        stacked = torch.stack(
            [s[name].float() * (w / total) for s, w in zip(states, weights)])
        avg[name] = stacked.sum(dim=0)
    return avg


def state_num_bytes(state: dict) -> int:
    """Size of one parameter package in bytes (fp32)."""
    return sum(t.numel() * 4 for t in state.values())


# ---------- training and evaluation ----------

def train_local(model, loader, device, epochs: int):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rounds = 1 if args.smoke else ROUNDS

    # ----- load clients, carve local held-out sets -----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    client_train_loaders, client_eval_loaders, client_sizes = [], [], []
    for i in range(NUM_CLIENTS):
        df = pd.read_csv(DATA_DIR / f"client_{i}.csv")
        if args.smoke:
            df = df.sample(n=min(150, len(df)), random_state=SEED)
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        cut = int(len(df) * (1 - LOCAL_HOLDOUT_FRAC))
        train_df, eval_df = df.iloc[:cut], df.iloc[cut:]
        client_train_loaders.append(DataLoader(
            TextDataset(train_df, tokenizer), batch_size=BATCH_SIZE, shuffle=True))
        client_eval_loaders.append(DataLoader(
            TextDataset(eval_df, tokenizer), batch_size=BATCH_SIZE))
        client_sizes.append(len(train_df))
        print(f"client_{i}: {len(train_df)} train rows, {len(eval_df)} local eval rows")

    test_df = pd.read_csv(TEST_PATH)
    if args.smoke:
        test_df = test_df.sample(n=80, random_state=SEED)
    test_loader = DataLoader(TextDataset(test_df, tokenizer), batch_size=BATCH_SIZE)

    # ----- global model -----
    model = build_model().to(device)
    model.print_trainable_parameters()
    global_state = get_trainable_state(model)

    package_bytes = state_num_bytes(global_state)
    print(f"Parameter package size: {package_bytes / 1e6:.2f} MB "
          f"(vs full model {125537288 * 4 / 1e6:.0f} MB)")

    # ----- federated rounds -----
    history = []
    total_comm_bytes = 0
    start = time.time()

    for rnd in range(1, rounds + 1):
        client_states = []
        for i in range(NUM_CLIENTS):
            set_trainable_state(model, global_state)      # server -> client
            train_local(model, client_train_loaders[i], device, LOCAL_EPOCHS)
            client_states.append(get_trainable_state(model))  # client -> server
            total_comm_bytes += 2 * package_bytes         # down + up
            print(f"  round {rnd}: client_{i} done")

        global_state = fedavg(client_states, client_sizes)

        set_trainable_state(model, global_state)
        acc, macro, per_class = evaluate(model, test_loader, device)
        history.append({"round": rnd, "accuracy": acc, "macro_f1": macro})
        print(f"round {rnd}/{rounds}: test accuracy {acc:.4f}, "
              f"macro F1 {macro:.4f}")

    minutes = (time.time() - start) / 60
    print(f"\nFederated training finished in {minutes:.1f} minutes")

    # ----- final global metrics -----
    set_trainable_state(model, global_state)
    acc, macro, per_class = evaluate(model, test_loader, device)
    print(f"\nFinal global model: accuracy {acc:.4f}, macro F1 {macro:.4f}")
    for c, f in per_class.items():
        print(f"  {c}: F1 {f:.4f}")

    # ----- per-client view: global model on each client's local data -----
    print("\nPer-client evaluation (global model on local held-out data):")
    per_client = {}
    for i in range(NUM_CLIENTS):
        c_acc, c_macro, _ = evaluate(model, client_eval_loaders[i], device)
        per_client[f"client_{i}"] = {"accuracy": c_acc, "macro_f1": c_macro,
                                     "train_rows": client_sizes[i]}
        print(f"  client_{i} ({client_sizes[i]} train rows): "
              f"accuracy {c_acc:.4f}, macro F1 {c_macro:.4f}")

    # ----- save -----
    RESULTS_DIR.mkdir(exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    metrics = {
        "condition": "federated_fedavg",
        "run": tag,
        "rounds": rounds,
        "local_epochs": LOCAL_EPOCHS,
        "num_clients": NUM_CLIENTS,
        "final_accuracy": acc,
        "final_macro_f1": macro,
        "per_class": per_class,
        "round_history": history,
        "per_client": per_client,
        "package_mb": round(package_bytes / 1e6, 2),
        "total_comm_mb": round(total_comm_bytes / 1e6, 1),
        "train_minutes": round(minutes, 1),
        "seed": SEED,
    }
    out_path = RESULTS_DIR / f"stage4_federated_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {out_path}")

    if not args.smoke:
        adapter_path = RESULTS_DIR / "adapters" / "federated_fedavg.pt"
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(global_state, adapter_path)
        print(f"Federated adapter state saved to {adapter_path}")


if __name__ == "__main__":
    main()