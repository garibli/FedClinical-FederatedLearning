"""
Stage 5: Federated training WITH differential privacy (DP-FedAvg).

Same federation as stage 4 (5 clients, 6 rounds, FedAvg over LoRA adapters),
but every client now trains with DP-SGD via Opacus:
  - per-example gradients are clipped to MAX_GRAD_NORM,
  - Gaussian noise is added to the summed gradients,
  - a per-client privacy accountant tracks epsilon across ALL rounds.

The noise level is calibrated ONCE per client for the whole training
(ROUNDS x LOCAL_EPOCHS epochs), so the final accumulated epsilon lands at
the target. Each client keeps its own persistent accountant; budgets are
never reset between rounds.

Usage:
    python src/stage5_federated_dp.py --epsilon 8            # full run
    python src/stage5_federated_dp.py --epsilon 3
    python src/stage5_federated_dp.py --epsilon 1
    python src/stage5_federated_dp.py --epsilon 8 --smoke    # plumbing check
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, classification_report
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager

# ---------------- settings (must match stage 4) ----------------
SEED = 42
MODEL_NAME = "roberta-base"
LABEL2ID = {"Anxiety": 0, "Depression": 1, "Normal": 2, "Suicidal": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

MAX_LENGTH = 256
LOGICAL_BATCH_SIZE = 64    # batch size for privacy accounting and averaging
PHYSICAL_BATCH_SIZE = 16   # what the GPU actually processes at once
DP_LEARNING_RATE = 1e-3    # DP needs a higher lr than non-private training
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

NUM_CLIENTS = 5
ROUNDS = 6
LOCAL_EPOCHS = 1
LOCAL_HOLDOUT_FRAC = 0.10

# ---------------- DP settings ----------------
DELTA = 1e-5            # standard delta; conservative for all client sizes
MAX_GRAD_NORM = 1.0     # per-example gradient clipping bound

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


def get_trainable_state(model) -> dict:
    return {name: p.detach().cpu().clone()
            for name, p in model.named_parameters() if p.requires_grad}


def set_trainable_state(model, state: dict) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(state[name].to(p.device))


def fedavg(states: list[dict], weights: list[int]) -> dict:
    total = sum(weights)
    avg = {}
    for name in states[0]:
        stacked = torch.stack(
            [s[name].float() * (w / total) for s, w in zip(states, weights)])
        avg[name] = stacked.sum(dim=0)
    return avg


def state_num_bytes(state: dict) -> int:
    return sum(t.numel() * 4 for t in state.values())


def train_local_dp(model, loader, device, engine, noise_multiplier):
    """One client's local training under DP-SGD.
    Wraps model+optimizer with the client's persistent PrivacyEngine,
    trains LOCAL_EPOCHS, then removes hooks so the next client can wrap.
    """
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=DP_LEARNING_RATE)

    wrapped_model, wrapped_opt, wrapped_loader = engine.make_private(
        module=model, optimizer=optimizer, data_loader=loader,
        noise_multiplier=noise_multiplier, max_grad_norm=MAX_GRAD_NORM)

    # BatchMemoryManager: privacy accounting sees the LOGICAL batch (64),
    # while the GPU physically processes PHYSICAL_BATCH_SIZE (16) at a time.
    for _ in range(LOCAL_EPOCHS):
        with BatchMemoryManager(
                data_loader=wrapped_loader,
                max_physical_batch_size=PHYSICAL_BATCH_SIZE,
                optimizer=wrapped_opt) as mem_loader:
            for batch in mem_loader:
                if len(batch["labels"]) == 0:
                    continue  # Poisson sampling can produce empty batches
                out = wrapped_model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device))
                out.loss.backward()
                wrapped_opt.step()
                wrapped_opt.zero_grad()

    wrapped_model.remove_hooks()
    del wrapped_opt, wrapped_loader
    torch.cuda.empty_cache()


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
    parser.add_argument("--epsilon", type=float, required=True,
                        help="target privacy budget for the whole training")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | target epsilon: {args.epsilon}, "
          f"delta: {DELTA}, clip norm: {MAX_GRAD_NORM}")

    rounds = 1 if args.smoke else ROUNDS

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
            TextDataset(train_df, tokenizer), batch_size=LOGICAL_BATCH_SIZE, shuffle=True))
        client_eval_loaders.append(DataLoader(
            TextDataset(eval_df, tokenizer), batch_size=PHYSICAL_BATCH_SIZE))
        client_sizes.append(len(train_df))
        print(f"client_{i}: {len(train_df)} train rows, {len(eval_df)} local eval rows")

    test_df = pd.read_csv(TEST_PATH)
    if args.smoke:
        test_df = test_df.sample(n=80, random_state=SEED)
    test_loader = DataLoader(TextDataset(test_df, tokenizer), batch_size=PHYSICAL_BATCH_SIZE)

    # ----- per-client noise calibration for the WHOLE training -----
    total_epochs = rounds * LOCAL_EPOCHS
    noise_multipliers = []
    for i in range(NUM_CLIENTS):
        sample_rate = LOGICAL_BATCH_SIZE / client_sizes[i]
        nm = get_noise_multiplier(
            target_epsilon=args.epsilon, target_delta=DELTA,
            sample_rate=sample_rate, epochs=total_epochs)
        noise_multipliers.append(nm)
        print(f"client_{i}: sample_rate {sample_rate:.4f}, "
              f"noise_multiplier {nm:.3f}")

    # one persistent privacy engine (accountant) per client
    engines = [PrivacyEngine(accountant="rdp") for _ in range(NUM_CLIENTS)]

    model = build_model().to(device)
    model.print_trainable_parameters()
    global_state = get_trainable_state(model)
    package_bytes = state_num_bytes(global_state)
    print(f"Parameter package size: {package_bytes / 1e6:.2f} MB")

    history = []
    total_comm_bytes = 0
    start = time.time()

    for rnd in range(1, rounds + 1):
        client_states = []
        for i in range(NUM_CLIENTS):
            set_trainable_state(model, global_state)
            train_local_dp(model, client_train_loaders[i], device,
                           engines[i], noise_multipliers[i])
            client_states.append(get_trainable_state(model))
            total_comm_bytes += 2 * package_bytes
            eps_now = engines[i].accountant.get_epsilon(delta=DELTA)
            print(f"  round {rnd}: client_{i} done, epsilon so far {eps_now:.2f}")

        global_state = fedavg(client_states, client_sizes)
        set_trainable_state(model, global_state)
        acc, macro, per_class = evaluate(model, test_loader, device)
        history.append({"round": rnd, "accuracy": acc, "macro_f1": macro})
        print(f"round {rnd}/{rounds}: test accuracy {acc:.4f}, "
              f"macro F1 {macro:.4f}")

    minutes = (time.time() - start) / 60
    print(f"\nDP federated training finished in {minutes:.1f} minutes")

    set_trainable_state(model, global_state)
    acc, macro, per_class = evaluate(model, test_loader, device)
    print(f"\nFinal global model: accuracy {acc:.4f}, macro F1 {macro:.4f}")
    for c, f in per_class.items():
        print(f"  {c}: F1 {f:.4f}")

    print("\nPer-client evaluation (global model on local held-out data):")
    per_client = {}
    for i in range(NUM_CLIENTS):
        c_acc, c_macro, _ = evaluate(model, client_eval_loaders[i], device)
        eps_spent = engines[i].accountant.get_epsilon(delta=DELTA)
        per_client[f"client_{i}"] = {
            "accuracy": c_acc, "macro_f1": c_macro,
            "train_rows": client_sizes[i],
            "epsilon_spent": round(eps_spent, 2),
        }
        print(f"  client_{i} ({client_sizes[i]} rows): accuracy {c_acc:.4f}, "
              f"macro F1 {c_macro:.4f}, epsilon spent {eps_spent:.2f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    tag = "smoke" if args.smoke else "full"
    eps_tag = str(args.epsilon).replace(".", "p")
    metrics = {
        "condition": "federated_dp",
        "run": tag,
        "target_epsilon": args.epsilon,
        "delta": DELTA,
        "max_grad_norm": MAX_GRAD_NORM,
        "rounds": rounds,
        "local_epochs": LOCAL_EPOCHS,
        "num_clients": NUM_CLIENTS,
        "noise_multipliers": [round(n, 3) for n in noise_multipliers],
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
    out_path = RESULTS_DIR / f"stage5_dp_eps{eps_tag}_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {out_path}")

    if not args.smoke:
        adapter_path = RESULTS_DIR / "adapters" / f"federated_dp_eps{eps_tag}.pt"
        adapter_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(global_state, adapter_path)
        print(f"DP federated adapter saved to {adapter_path}")


if __name__ == "__main__":
    main()