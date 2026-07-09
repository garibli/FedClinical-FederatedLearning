"""
Stage 1: Data pipeline for the FedClinical thesis project.

What this script does:
1. Loads the raw mental health dataset.
2. Cleans it: removes duplicate texts and empty rows.
3. Splits it into 5 simulated clients using a Dirichlet distribution,
   so clients differ in size and in class mix (non IID).
4. Saves the cleaned test set and one csv per client into data/processed.

Run it once. Every later experiment reads the files this script produces,
so the split stays identical across the whole thesis.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------- settings ----------------
SEED = 42                  # fixed seed so the split is reproducible
NUM_CLIENTS = 5            # how many simulated clinics
DIRICHLET_ALPHA = 0.5      # lower = clients more different, higher = more similar
CLIENT_SIZE_WEIGHTS = [0.35, 0.25, 0.18, 0.12, 0.10]  # uneven client sizes

RAW_TRAIN = Path("data/raw/mental_heath_unbanlanced.csv")
RAW_TEST = Path("data/raw/mental_health_combined_test.csv")
OUT_DIR = Path("data/processed")

rng = np.random.default_rng(SEED)


def load_and_clean(path: Path) -> pd.DataFrame:
    """Load a csv, keep only text and status, drop duplicates and empty rows."""
    df = pd.read_csv(path)
    df = df[["text", "status"]]

    before = len(df)
    df = df.dropna(subset=["text", "status"])          # remove empty rows
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0]                  # remove blank texts
    df = df.drop_duplicates(subset=["text"])           # remove duplicate texts
    after = len(df)

    print(f"{path.name}: {before} rows -> {after} rows after cleaning "
          f"({before - after} removed)")
    return df.reset_index(drop=True)


def dirichlet_split(df: pd.DataFrame) -> list[pd.DataFrame]:
    """
    Split the dataframe into NUM_CLIENTS parts.

    Two ideas combined:
    - CLIENT_SIZE_WEIGHTS makes clients different sizes (a big clinic, a small one).
    - The Dirichlet distribution decides, for every class, what share of that
      class goes to each client. With a low alpha the shares are very uneven,
      which makes clients specialise (one sees mostly Suicidal, another mostly
      Anxiety). This is the non IID split the research questions need.
    """
    classes = df["status"].unique()
    client_indices: list[list[int]] = [[] for _ in range(NUM_CLIENTS)]

    for cls in classes:
        cls_idx = df.index[df["status"] == cls].to_numpy().copy()
        rng.shuffle(cls_idx)

        # base proportions from client sizes, tilted by Dirichlet randomness
        proportions = rng.dirichlet(DIRICHLET_ALPHA * np.array(CLIENT_SIZE_WEIGHTS) * NUM_CLIENTS)

        # turn proportions into cut points inside this class's rows
        cuts = (np.cumsum(proportions)[:-1] * len(cls_idx)).astype(int)
        parts = np.split(cls_idx, cuts)

        for client_id, part in enumerate(parts):
            client_indices[client_id].extend(part.tolist())

    clients = []
    for client_id, idx in enumerate(client_indices):
        client_df = df.loc[idx].sample(frac=1, random_state=SEED).reset_index(drop=True)
        clients.append(client_df)
    return clients


def report(clients: list[pd.DataFrame]) -> None:
    """Print a small table showing size and class mix of every client."""
    print("\nClient summary:")
    header = f"{'client':<8}{'rows':<8}" + "".join(f"{c:<12}" for c in sorted(clients[0]['status'].unique()))
    print(header)
    for i, cdf in enumerate(clients):
        counts = cdf["status"].value_counts()
        row = f"client_{i:<2}{len(cdf):<8}"
        for cls in sorted(clients[0]["status"].unique()):
            row += f"{counts.get(cls, 0):<12}"
        print(row)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train = load_and_clean(RAW_TRAIN)
    test = load_and_clean(RAW_TEST)

    # safety check: no test text may appear in training data (leakage)
    leaked = test["text"].isin(set(train["text"])).sum()
    if leaked > 0:
        print(f"WARNING: {leaked} test texts also exist in training data, removing them from train")
        train = train[~train["text"].isin(set(test["text"]))].reset_index(drop=True)

    clients = dirichlet_split(train)
    report(clients)

    # save everything
    test.to_csv(OUT_DIR / "test_balanced.csv", index=False)
    train.to_csv(OUT_DIR / "train_full_centralised.csv", index=False)
    for i, cdf in enumerate(clients):
        cdf.to_csv(OUT_DIR / f"client_{i}.csv", index=False)

    print(f"\nSaved: test_balanced.csv, train_full_centralised.csv and "
          f"{NUM_CLIENTS} client files into {OUT_DIR}/")


if __name__ == "__main__":
    main()