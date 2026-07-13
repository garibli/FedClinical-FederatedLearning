# FedClinical: Federated Learning for Mental Health Text Classification

Measuring what privacy actually costs. This repository contains the full experimental framework and results of an MSc dissertation at the University of Glasgow that studies the effects of differential privacy and parameter-efficient fine-tuning on federated learning, using mental health text classification as the task.

## The idea in one paragraph

Depression, anxiety and suicidal thoughts leave visible traces in the way people write, and a text classifier can learn to recognise them. But the text needed to train such a model is the most private data that exists, and clinics or platforms cannot pool it on a central server. Federated learning trains a shared model without moving the data, and differential privacy adds a formal guarantee that no individual post can be recovered from training. This project measures what those protections cost in prediction quality, and, unusually, who inside the federation pays that cost.

## The task

Four-class classification of social media posts: **Normal, Depression, Anxiety, Suicidal**. Data comes from a public, openly licensed combined mental health corpus (~48,500 posts after cleaning), split into five simulated clients with deliberately unequal sizes and class mixtures via a Dirichlet partition. The model is RoBERTa-base, fine-tuned with LoRA adapters (only 0.71 percent of parameters train), federated with a custom FedAvg engine, and made private with DP-SGD through Opacus.

## Headline results

| Condition | Macro F1 |
|---|---|
| Centralised (upper bound) | 0.878 |
| Federated, no privacy | 0.861 |
| Federated + DP, epsilon 8 | 0.720 |
| Federated + DP, epsilon 3 | 0.684 |
| Federated + DP, epsilon 1 | 0.589 |
| Classical TF-IDF + LogReg | 0.692 |
| Frozen model (lower bound) | 0.100 |

Four findings stand out:

1. **Federation is nearly free, privacy is not, and its price is non-linear.** Splitting data across five unequal clients cost 1.7 macro F1 points. Entering differential privacy cost 14.1 points at epsilon 8, tightening to epsilon 3 cost only 3.6 more, and going strict to epsilon 1 cost a further 9.5.
2. **Privacy noise destroys the clinically important distinctions first.** Increasing noise collapses Anxiety and Suicidal predictions into Depression (at epsilon 1, half of Anxiety posts are labelled Depression), while the coarse Normal boundary survives every budget tested.
3. **DP training lives or dies by DP-specific tuning.** The same epsilon 8 budget scored 0.288 with naive hyperparameters and 0.720 with a DP-appropriate learning rate and large logical batches. Mistuning masquerades as privacy cost.
4. **The real value of federation is not better home performance.** On their own data, clients gained almost nothing by joining. What federation actually provides is coverage of blind-spot classes (one client's solo model had suicidal-detection F1 of exactly 0.0) and generalisation (solo models dropped up to 56 points on the global test).

Figures for all of this live in `results/figures/`.

## Repository structure

```
src/
  stage1_data_pipeline.py      cleaning, leakage check, Dirichlet client split
  stage2_centralised.py        centralised LoRA fine-tuning (upper bound)
  stage3_baselines.py          frozen baseline + classical TF-IDF baseline
  stage4_federated.py          custom FedAvg federation, per-client evaluation
  stage5_federated_dp.py       DP-FedAvg with per-client privacy accounting
  stage6_local_baselines.py    each client training alone (gain-from-joining)
  stage6_analysis.py           all tables, confusion matrices and figures
results/
  *.json                       metrics of every run
  *.csv                        per-client and gain-from-joining tables
  figures/                     the five result figures
```

Raw data and trained adapters are intentionally not tracked; see below.

## Reproducing the experiments

Requirements: Python 3.11+, an NVIDIA GPU with ~8 GB VRAM (all experiments ran on a single RTX 4060 laptop GPU), and:

```
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install transformers peft opacus pandas scikit-learn matplotlib
```

Download the dataset (see Dataset section), place the two csv files in `data/raw/`, then run the stages in order:

```
python src/stage1_data_pipeline.py
python src/stage2_centralised.py
python src/stage3_baselines.py
python src/stage4_federated.py
python src/stage5_federated_dp.py --epsilon 8
python src/stage5_federated_dp.py --epsilon 3
python src/stage5_federated_dp.py --epsilon 1
python src/stage6_local_baselines.py
python src/stage6_analysis.py
```

Every training script has a `--smoke` flag for a fast plumbing check. Everything is seeded (42), so the client split and results are reproducible. Total compute for the full matrix is roughly 8 hours on the GPU above.

## Dataset

The dataset is a publicly available combined mental health text corpus distributed under an open licence via Kaggle, assembled from three earlier public corpora. It is not redistributed in this repository; download it from the source and place it in `data/raw/`. All data is public and de-identified. The models trained here are research artefacts, not diagnostic tools.

## Technical notes worth knowing

- The federation is a transparent sequential FedAvg implementation rather than an off-the-shelf framework, which made integrating Opacus into the client loop straightforward and keeps every step inspectable.
- Only LoRA adapters and the classification head travel between clients and server: 3.55 MB per package against ~502 MB for the full model, a 141x reduction.
- Privacy budgets accumulate across rounds. Each client owns a persistent RDP accountant, and noise is calibrated once per client for the whole training so the final epsilon lands on target (verified in every run's json).
- The stage 1 pipeline caught real train/test leakage in the source data (496 of 992 test texts appeared in the training file) and repairs it automatically.

## Author

Fuad Garibli, MSc Data Science, School of Computing Science, University of Glasgow. Dissertation: *The Effects of Differential Privacy and Fine-Tuning on Federated Learning for Mental Health Text Classification: A Study Using the FedClinical Framework* (2026).

## License

MIT. See LICENSE.
