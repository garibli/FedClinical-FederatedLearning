"""
Stage 6b: Analysis layer. Trains nothing. Loads saved results and adapters,
then produces every remaining table and figure of the thesis:

  1. Gain-from-joining table: local-only vs federated vs federated+DP,
     each measured on the client's own local held-out data.
  2. Present-classes-only macro F1 per client (decomposes the mechanical
     zero for absent classes from real model weakness).
  3. Confusion matrices on the balanced test set for: federated, DP eps 8/3/1.
  4. Figures (results/figures/): privacy-accuracy curve, convergence
     comparison, per-client grouped bars, confusion matrix grid.

Run AFTER stage6_local_baselines.py:
    python src/stage6_analysis.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix)

SEED = 42
MODEL_NAME = "roberta-base"
LABEL2ID = {"Anxiety": 0, "Depression": 1, "Normal": 2, "Suicidal": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
CLASSES = [ID2LABEL[i] for i in range(4)]

MAX_LENGTH = 256
BATCH_SIZE = 32
NUM_CLIENTS = 5
LOCAL_HOLDOUT_FRAC = 0.10

DATA_DIR = Path("data/processed")
TEST_PATH = DATA_DIR / "test_balanced.csv"
RESULTS_DIR = Path("results")
ADAPTER_DIR = RESULTS_DIR / "adapters"
FIG_DIR = RESULTS_DIR / "figures"


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
            padding="max_length", return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def build_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4, id2label=ID2LABEL, label2id=LABEL2ID)
    cfg = LoraConfig(task_type="SEQ_CLS", r=8, lora_alpha=16,
                     lora_dropout=0.1, target_modules=["query", "value"])
    return get_peft_model(model, cfg)


def load_adapter(model, path: Path):
    state = torch.load(path, map_location="cpu")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(state[name].to(p.device))


def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device)).logits
            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            labels.extend(batch["labels"].numpy().tolist())
    return np.array(labels), np.array(preds)


def macro_all_and_present(y_true, y_pred):
    """Macro F1 two ways: over all 4 classes, and only over classes
    actually present in y_true. The gap is the mechanical penalty."""
    macro_all = f1_score(y_true, y_pred, average="macro",
                         labels=list(range(4)), zero_division=0)
    present = sorted(set(y_true.tolist()))
    macro_present = f1_score(y_true, y_pred, average="macro",
                             labels=present, zero_division=0)
    return macro_all, macro_present, [ID2LABEL[i] for i in present]


def client_eval_loaders(tokenizer):
    loaders = []
    for i in range(NUM_CLIENTS):
        df = pd.read_csv(DATA_DIR / f"client_{i}.csv")
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        cut = int(len(df) * (1 - LOCAL_HOLDOUT_FRAC))
        eval_df = df.iloc[cut:]
        loaders.append(DataLoader(TextDataset(eval_df, tokenizer),
                                  batch_size=BATCH_SIZE))
    return loaders


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_df = pd.read_csv(TEST_PATH)
    test_loader = DataLoader(TextDataset(test_df, tokenizer),
                             batch_size=BATCH_SIZE)
    local_loaders = client_eval_loaders(tokenizer)

    model = build_model().to(device)

    adapters = {
        "federated": ADAPTER_DIR / "federated_fedavg.pt",
        "dp_eps8": ADAPTER_DIR / "federated_dp_eps8p0.pt",
        "dp_eps3": ADAPTER_DIR / "federated_dp_eps3p0.pt",
        "dp_eps1": ADAPTER_DIR / "federated_dp_eps1p0.pt",
    }
    local_adapters = {f"client_{i}": ADAPTER_DIR / f"local_only_client_{i}.pt"
                      for i in range(NUM_CLIENTS)}

    # ---------- 1 + 2: per-client table across conditions ----------
    print("\n=== Per-client analysis ===")
    per_client_rows = []
    global_preds = {}   # condition -> (y_true, y_pred) on balanced test

    for cond, path in adapters.items():
        load_adapter(model, path)
        global_preds[cond] = predict(model, test_loader, device)
        for i in range(NUM_CLIENTS):
            y, p = predict(model, local_loaders[i], device)
            m_all, m_present, present = macro_all_and_present(y, p)
            per_client_rows.append({
                "condition": cond, "client": f"client_{i}",
                "macro_f1_all": round(m_all, 4),
                "macro_f1_present_only": round(m_present, 4),
                "present_classes": present,
            })

    # local-only models: each evaluated on its own holdout
    for i in range(NUM_CLIENTS):
        load_adapter(model, local_adapters[f"client_{i}"])
        y, p = predict(model, local_loaders[i], device)
        m_all, m_present, present = macro_all_and_present(y, p)
        per_client_rows.append({
            "condition": "local_only", "client": f"client_{i}",
            "macro_f1_all": round(m_all, 4),
            "macro_f1_present_only": round(m_present, 4),
            "present_classes": present,
        })

    pc_df = pd.DataFrame(per_client_rows)
    print(pc_df.to_string(index=False))

    # gain from joining (all-classes and present-only views)
    print("\n=== Gain from joining (macro F1 on own local holdout) ===")
    gain_rows = []
    for i in range(NUM_CLIENTS):
        c = f"client_{i}"
        loc = pc_df[(pc_df.condition == "local_only") & (pc_df.client == c)]
        fed = pc_df[(pc_df.condition == "federated") & (pc_df.client == c)]
        dp8 = pc_df[(pc_df.condition == "dp_eps8") & (pc_df.client == c)]
        gain_rows.append({
            "client": c,
            "local_only": loc.macro_f1_all.iloc[0],
            "federated": fed.macro_f1_all.iloc[0],
            "gain_fed": round(fed.macro_f1_all.iloc[0] - loc.macro_f1_all.iloc[0], 4),
            "dp_eps8": dp8.macro_f1_all.iloc[0],
            "gain_dp8": round(dp8.macro_f1_all.iloc[0] - loc.macro_f1_all.iloc[0], 4),
            "local_present_only": loc.macro_f1_present_only.iloc[0],
            "fed_present_only": fed.macro_f1_present_only.iloc[0],
        })
    gain_df = pd.DataFrame(gain_rows)
    print(gain_df.to_string(index=False))

    # ---------- 3: confusion matrices ----------
    print("\nBuilding confusion matrices...")
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, (cond, (y, p)) in zip(axes.flat, global_preds.items()):
        cm = confusion_matrix(y, p, labels=list(range(4)), normalize="true")
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(4), CLASSES, rotation=45, ha="right")
        ax.set_yticks(range(4), CLASSES)
        ax.set_title(cond)
        for r in range(4):
            for c_ in range(4):
                ax.text(c_, r, f"{cm[r, c_]:.2f}", ha="center", va="center",
                        color="white" if cm[r, c_] > 0.5 else "black",
                        fontsize=9)
    fig.suptitle("Row-normalised confusion matrices (balanced test set)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrices.png", dpi=200)
    plt.close(fig)

    # ---------- 4: figures ----------
    print("Building figures...")

    # privacy-accuracy curve
    fed_json = json.load(open(RESULTS_DIR / "stage4_federated_full.json"))
    curve = {
        "No DP": fed_json["final_macro_f1"],
        "eps=8": json.load(open(RESULTS_DIR / "stage5_dp_eps8p0_full.json"))["final_macro_f1"],
        "eps=3": json.load(open(RESULTS_DIR / "stage5_dp_eps3p0_full.json"))["final_macro_f1"],
        "eps=1": json.load(open(RESULTS_DIR / "stage5_dp_eps1p0_full.json"))["final_macro_f1"],
    }
    classical = json.load(open(RESULTS_DIR / "stage3_baselines.json"))["classical"]["macro_f1"]
    centralised = json.load(open(RESULTS_DIR / "stage2_centralised_full.json"))["macro_f1"]

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = list(range(len(curve)))
    ax.plot(xs, list(curve.values()), marker="o", linewidth=2, label="Federated")
    ax.axhline(centralised, linestyle="--", color="green",
               label=f"Centralised ({centralised:.3f})")
    ax.axhline(classical, linestyle=":", color="gray",
               label=f"Classical TF-IDF ({classical:.3f})")
    ax.set_xticks(xs, list(curve.keys()))
    ax.set_xlabel("Privacy level (stricter to the right)")
    ax.set_ylabel("Macro F1 (balanced test)")
    ax.set_title("The price of privacy")
    for x, v in zip(xs, curve.values()):
        ax.annotate(f"{v:.3f}", (x, v), textcoords="offset points",
                    xytext=(0, 8), ha="center")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "privacy_accuracy_curve.png", dpi=200)
    plt.close(fig)

    # convergence comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, fname in [("No DP", "stage4_federated_full.json"),
                         ("eps=8", "stage5_dp_eps8p0_full.json"),
                         ("eps=3", "stage5_dp_eps3p0_full.json"),
                         ("eps=1", "stage5_dp_eps1p0_full.json")]:
        hist = json.load(open(RESULTS_DIR / fname))["round_history"]
        ax.plot([h["round"] for h in hist], [h["macro_f1"] for h in hist],
                marker="o", label=label)
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Macro F1 (balanced test)")
    ax.set_title("Convergence under increasing privacy")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "convergence_comparison.png", dpi=200)
    plt.close(fig)

    # per-client grouped bars
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.28
    x = np.arange(NUM_CLIENTS)
    for k, cond in enumerate(["local_only", "federated", "dp_eps8"]):
        vals = [pc_df[(pc_df.condition == cond) &
                      (pc_df.client == f"client_{i}")].macro_f1_all.iloc[0]
                for i in range(NUM_CLIENTS)]
        ax.bar(x + (k - 1) * width, vals, width, label=cond)
    ax.set_xticks(x, [f"client_{i}" for i in range(NUM_CLIENTS)])
    ax.set_ylabel("Macro F1 on own local holdout")
    ax.set_title("Per-client view: alone vs federation vs private federation")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_client_bars.png", dpi=200)
    plt.close(fig)

    # present-only vs all-classes per client (federated model)
    fig, ax = plt.subplots(figsize=(9, 5))
    fed_rows = pc_df[pc_df.condition == "federated"]
    x = np.arange(NUM_CLIENTS)
    ax.bar(x - 0.2, fed_rows.macro_f1_all.values, 0.4, label="all 4 classes")
    ax.bar(x + 0.2, fed_rows.macro_f1_present_only.values, 0.4,
           label="present classes only")
    ax.set_xticks(x, fed_rows.client.values)
    ax.set_ylabel("Macro F1 on own local holdout")
    ax.set_title("Mechanical penalty of absent classes (federated model)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "present_only_decomposition.png", dpi=200)
    plt.close(fig)

    # ---------- save all tables ----------
    out = {
        "per_client_table": per_client_rows,
        "gain_from_joining": gain_rows,
    }
    with open(RESULTS_DIR / "stage6_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    pc_df.to_csv(RESULTS_DIR / "stage6_per_client_table.csv", index=False)
    gain_df.to_csv(RESULTS_DIR / "stage6_gain_from_joining.csv", index=False)

    print(f"\nSaved: stage6_analysis.json, two csv tables, and 4 figures "
          f"in {FIG_DIR}/")


if __name__ == "__main__":
    main()