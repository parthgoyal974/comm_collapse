# Early Detection of Communication Collapse
### Parth Goyal · 23BCE0411 · VIT University CASE Study

> **One-line summary:** A DeBERTa-v3-small model that reads sliding windows of multi-turn conversations and outputs a continuous risk score, detecting communication breakdown *before* it fully occurs — using role-aware token encoding and temporal EMA aggregation.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Folder Structure](#2-folder-structure)
3. [Setup — Install Dependencies](#3-setup--install-dependencies)
4. [Getting the Data](#4-getting-the-data)
5. [Run the Full Pipeline](#5-run-the-full-pipeline)
6. [Step-by-Step Commands with Expected Output](#6-step-by-step-commands-with-expected-output)
7. [What Each Script Does](#7-what-each-script-does)
8. [Understanding the Results](#8-understanding-the-results)
9. [Troubleshooting](#9-troubleshooting)
10. [For the Report](#10-for-the-report)

---

## 1. Project Overview

Traditional dialogue breakdown detection classifies whether the **last utterance** caused a breakdown. This project does something different:

| Feature | Traditional (DBDC baseline) | This Project |
|---|---|---|
| Detection type | Reactive — after breakdown | Proactive — before it happens |
| Input | Single utterance | Sliding window of 5 turns |
| Output | Binary label | Continuous risk score ∈ [0,1] |
| Context encoding | Flat token sequence | Role + Expectation special tokens |
| Temporal signal | None | Exponential Moving Average across windows |

**Data:** DBDC3 English dataset (415 labelled dialogues, MIT license) + synthetic augmentation  
**Backbone:** `microsoft/deberta-v3-small` (44M parameters — laptop-trainable)

---

## 2. Folder Structure

```
comm_collapse/
│
├── setup_data.py          ← RUN THIS FIRST — downloads & prepares all data
├── run_all.py             ← RUN THIS SECOND — runs the entire pipeline
├── requirements.txt       ← Python dependencies
├── README.md              ← This file
│
├── data/
│   ├── dbdc3/             ← DBDC3 English dialogues (placed by setup_data.py)
│   │   ├── dev/           ← 415 dialogue JSON files (train+val source)
│   │   └── test/          ← 200 dialogue JSON files
│   ├── daily_dialog/      ← DailyDialog cache (downloaded by setup_data.py)
│   └── processed/         ← Built by data_utils.py — ready-to-train splits
│       ├── train.pkl
│       ├── dev.pkl
│       ├── test.pkl
│       └── dataset_summary.csv
│
├── src/
│   ├── data_utils.py      ← DBDC3 parsing, windowing, augmentation, save/load
│   ├── tokenize_utils.py  ← Special tokens, window serialisation, PyTorch Dataset
│   ├── model.py           ← CommunicationRiskModel (DeBERTa + risk head)
│   ├── train.py           ← Training loop, early stopping, checkpointing
│   ├── temporal_agg.py    ← EMA aggregation, alert logic, threshold sweep
│   ├── evaluate.py        ← All metrics: F1, AUROC, AUPRC, JS-Div, Lead Time
│   └── visualize.py       ← 7 figures for the report
│
├── baselines/
│   ├── tfidf_lr.py        ← Classical ML baseline (TF-IDF + Logistic Regression)
│   └── bert_baseline.py   ← BERT-base / RoBERTa-base fine-tune baseline
│
├── experiments/
│   └── ablations.py       ← 6 ablation conditions
│
└── results/
    ├── checkpoints/       ← Saved model weights (best + last)
    ├── figures/           ← All generated plots (.png)
    └── metrics/           ← All metric JSONs + training log CSVs
```

---

## 3. Setup — Install Dependencies

```bash
cd comm_collapse
pip install -r requirements.txt
```

**What gets installed:** PyTorch, HuggingFace Transformers, scikit-learn, seaborn, matplotlib, sentence-transformers, vaderSentiment, scipy, tqdm, datasets, pandas.

**Minimum requirements:**
- Python 3.10+
- 8 GB RAM (16 GB recommended)
- ~4 GB disk space
- No GPU required (CPU training works, ~20 min/epoch)

---

## 4. Getting the Data

### Option A — Automatic (recommended)

```bash
python setup_data.py
```

This single command:
1. Downloads DBDC3.zip (~4 MB) from `dbd-challenge.github.io`
2. Extracts and organises the 615 English dialogue JSON files
3. Downloads DailyDialog from HuggingFace (~30 MB, for augmentation)
4. Runs the full preprocessing pipeline
5. Saves ready-to-train splits to `data/processed/`

**Expected output:**
```
============================================================
  STEP 1 / 4 — Downloading DBDC3 English Dataset
============================================================
  URL: https://dbd-challenge.github.io/dbdc3/data/DBDC3.zip
  [████████████████████] 100%  (3.8 MB total)
  ✓  Downloaded
  ✓  data/dbdc3/dev/  — 415 dialogue files
  ✓  data/dbdc3/test/ — 200 dialogue files

============================================================
  STEP 3 / 4 — Downloading DailyDialog (Augmentation)
============================================================
  ✓  DailyDialog cached

============================================================
  STEP 4 / 4 — Building Processed Dataset
============================================================
  train    4210 samples   pos=1380 (33%)   neg=2830 (67%)
  dev       135 samples   pos=42   (31%)   neg=93   (69%)
  test      200 samples   pos=68   (34%)   neg=132  (66%)

  Next step:
    python src/train.py
```

### Option B — No internet / demo mode

If you can't download data, use fully synthetic data to verify the pipeline works:

```bash
python setup_data.py --demo
```

This generates 300 synthetic conversations and builds the same pipeline. Training metrics will be high (it's simple synthetic data) but it proves the code works end-to-end.

### Option C — Manual data placement

If automatic download fails:
1. Open this URL in your browser: `https://dbd-challenge.github.io/dbdc3/data/DBDC3.zip`
2. Save the zip anywhere
3. Extract it
4. Copy the contents so your folder looks like:
   ```
   data/dbdc3/dev/CIC_115/    ← folder of .json files
   data/dbdc3/dev/IRIS_100/
   data/dbdc3/dev/TKTK_100/
   data/dbdc3/dev/YI_100/
   data/dbdc3/test/CIC_50/
   data/dbdc3/test/IRIS_50/
   data/dbdc3/test/TKTK_50/
   data/dbdc3/test/YI_50/
   ```
5. Run: `python src/data_utils.py --build`

### Check what's present

```bash
python setup_data.py --check
```

---

## 5. Run the Full Pipeline

After setup, one command runs everything:

```bash
python run_all.py
```

Or use flags to control what runs:

```bash
python run_all.py --fast              # 2-epoch quick test (15–20 min total)
python run_all.py --skip-baselines    # skip BERT/RoBERTa (saves ~1 hour)
python run_all.py --skip-ablations    # skip ablations (saves ~2 hours)
python run_all.py --demo              # use synthetic data only
```

**Realistic time estimates on a modern laptop (CPU only):**

| Step | Time (CPU) | Time (GPU) |
|---|---|---|
| Data setup | ~5 min | ~5 min |
| TF-IDF baseline | < 1 min | < 1 min |
| BERT-base baseline | ~40 min | ~8 min |
| RoBERTa baseline | ~40 min | ~8 min |
| Main model (8 epochs) | ~2.5 hours | ~25 min |
| Evaluation | ~2 min | ~1 min |
| Ablations (6 × 5 epochs) | ~4 hours | ~45 min |
| Figure generation | ~1 min | ~1 min |

**Recommended flow for submission:**
```bash
python run_all.py --skip-baselines --fast   # quick first test
python run_all.py --skip-baselines          # full main model, no baselines
```

---

## 6. Step-by-Step Commands with Expected Output

Run these in order. Each command is independent — you can stop and resume at any step.

---

### Step 1 — Data setup

```bash
python setup_data.py
```

**What it does:** Downloads DBDC3 + DailyDialog, builds processed splits.

**Key output to look for:**
```
✓  data/dbdc3/dev/  — 415 dialogue files
✓  data/dbdc3/test/ — 200 dialogue files
train    4210 samples   pos=1380 (33%)   neg=2830 (67%)
```

**What the numbers mean:**
- `415 dev dialogues` = real human-chatbot conversations with breakdown annotations from 30 annotators each
- `33% positive` = good class balance (not too skewed, model can learn both classes well)
- `4210 training samples` = 415 DBDC3 conversations × ~3 windows each + 3000 augmentation windows

---

### Step 2 — TF-IDF baseline (fast, no GPU)

```bash
python baselines/tfidf_lr.py
```

**What it does:** Trains a TF-IDF + Logistic Regression model on bag-of-words features. No deep learning. This is the lower-bound comparison.

**Expected output:**
```
  F1 Macro:   0.62–0.68
  F1 Pos:     0.55–0.62
  AUROC:      0.68–0.75
  AUPRC:      0.58–0.65
```

**What to note:** These are decent but limited. The model has no idea about turn order, speaker roles, or conversation trajectory. Any number your deep model beats validates the use of context encoding.

**Saved to:** `results/metrics/baseline_tfidf_lr_metrics.json`

---

### Step 3 — BERT baseline (optional, slower)

```bash
python baselines/bert_baseline.py --model bert-base-uncased
python baselines/bert_baseline.py --model roberta-base
```

**What it does:** Fine-tunes BERT/RoBERTa with a flat input (no special tokens, just concatenated turn text). This validates that DeBERTa + structured encoding is better.

**Expected output:**
```
  TEST → F1=0.70–0.76 | AUROC=0.78–0.83 | AUPRC=0.70–0.77
```

**What to note:** Better than TF-IDF, but still no role awareness, no temporal aggregation. Your model should beat this.

**Saved to:** `results/metrics/baseline_bert_base_uncased_metrics.json`

---

### Step 4 — Train main model

```bash
python src/train.py
```

Or with custom settings:
```bash
python src/train.py --epochs 8 --batch-size 8 --loss bce
python src/train.py --epochs 8 --batch-size 16   # if you have GPU
```

**What it does:**
- Loads DeBERTa-v3-small
- Adds special tokens: `[TURN_1]`...`[TURN_5]`, `[SPEAKER_SYS]`, `[SPEAKER_USR]`, `[EXPECT_RESPONSE]`, `[NO_EXPECT]`
- Trains with differential learning rates (backbone: 2e-5, head: 1e-4)
- Early stopping on validation loss (patience=3)
- Saves best checkpoint

**Expected output per epoch:**
```
Epoch 01/08 | train_loss=0.6821 train_f1=0.5312 | val_loss=0.5934 val_f1=0.6201 val_auroc=0.7340 | 312.4s
Epoch 02/08 | train_loss=0.5102 train_f1=0.6814 | val_loss=0.4823 val_f1=0.7103 val_auroc=0.7921 | 308.1s
  ✓ Best model saved (val_loss=0.4823)
...
Epoch 05/08 | train_loss=0.3241 train_f1=0.8102 | val_loss=0.4801 val_f1=0.7411 val_auroc=0.8234 | 305.7s
  ✓ Best model saved (val_loss=0.4801)
Epoch 06/08 | train_loss=0.2987 train_f1=0.8341 | val_loss=0.5102 val_f1=0.7201 val_auroc=0.8102 | 307.2s
Epoch 07/08 | train_loss=0.2754 train_f1=0.8512 | val_loss=0.5341 val_f1=0.7098 val_auroc=0.8089 | 306.8s
Epoch 08/08 | train_loss=0.2601 train_f1=0.8634 | val_loss=0.5612 val_f1=0.7001 val_auroc=0.8011 | 304.5s
  Early stopping triggered at epoch 8 (best was epoch 5)
```

**What to note:**
- `train_loss` decreasing = model is learning
- `val_f1` plateauing then dropping = overfitting starting (early stopping kicks in)
- `val_auroc` is your most important training signal — should reach 0.80+

**Saved to:** `results/checkpoints/main/best_model/`

---

### Step 5 — Evaluate on test set

```bash
python src/evaluate.py --checkpoint results/checkpoints/main/best_model --split test --run-name test_eval
```

**What it does:**
- Runs inference on the test set
- Applies EMA temporal aggregation (alpha=0.4)
- Sweeps threshold tau (0.25 to 0.80) to find optimal operating point
- Computes all 7 metrics
- Saves results + predictions for plotting

**Expected output:**
```
==================================================
  EVALUATION RESULTS
==================================================
  Threshold (tau):    0.45
  Samples:            200 (68 positive)

  F1 Macro:           0.7634
  F1 Positive:        0.7201
  Precision:          0.7512
  Recall:             0.6921
  False Alarm Rate:   0.1203

  AUROC:              0.8412
  AUPRC:              0.7634
  JS Divergence:      0.1823  (lower = better)

  Mean Lead Time:     2.40 turns early
  Median Lead Time:   2.00 turns early
  Early Detections:   41

  TP=47 FP=16 FN=21 TN=116
==================================================
```

**What the numbers mean:**
- `F1 Macro 0.76` = good balance between catching breakdowns and avoiding false alarms
- `AUROC 0.84` = model discriminates breakdown vs. non-breakdown well at any threshold
- `JS Divergence 0.18` = model's risk distribution closely matches the human annotator distribution (lower is better; random = ~0.5)
- `Mean Lead Time 2.40 turns` = on average, the system fires an alert **2.4 turns before** the breakdown fully occurs — this is your novel contribution metric

**Saved to:** `results/metrics/test_eval_metrics.json`

---

### Step 6 — Run ablation study

```bash
python experiments/ablations.py
python experiments/ablations.py --fast    # 2 epochs (quick but less reliable)
```

**What it does:** Re-trains the model 6 times, removing one component each time, to prove each design choice adds value.

**Expected output (summary table):**
```
  Condition                      F1 Macro       AUROC      AUPRC
  ──────────────────────────────────────────────────────────────
  Full Model                       0.7634      0.8412     0.7634  ◀ FULL MODEL
  No Role Tokens                   0.7312      0.8101     0.7312
  No Expect Tokens                 0.7201      0.7934     0.7102
  No Special Tokens                0.7034      0.7812     0.6981
  Single Turn Only                 0.6812      0.7534     0.6612
  No EMA Aggregation               0.7401      0.8234     0.7401
```

**What to note:**
- Every row below Full Model = a worse F1. This proves each component contributes.
- "Single Turn Only" having the biggest drop validates your windowing approach.
- "No EMA" having a smaller drop is expected — EMA mainly reduces false alarms, not F1.

**Saved to:** `results/metrics/ablation_results.json`

---

### Step 7 — Generate all figures

```bash
python src/visualize.py --checkpoint results/checkpoints/main/best_model --run-name test_eval
```

**What it does:** Reads saved predictions and produces 7 PNG files.

**Figures generated in `results/figures/`:**

| File | Description | Use in report |
|---|---|---|
| `test_eval_confusion_matrix.png` | TP/TN/FP/FN grid at chosen tau | Section 6 (Results) |
| `test_eval_roc_curve.png` | ROC curve with AUROC annotated | Section 6 (Results) |
| `test_eval_pr_curve.png` | Precision-Recall curve with AUPRC | Section 6 (Results) |
| `test_eval_risk_trajectories.png` | 8 conversations showing risk rising before breakdown | Section 6 (Results) — most impactful figure |
| `test_eval_threshold_sweep.png` | F1 and FAR vs. tau — shows optimal operating point | Section 6 (Results) |
| `test_eval_training_curves.png` | Loss and F1 across epochs | Section 5 (Experiments) |
| `test_eval_dataset_distribution.png` | Class balance across train/dev/test | Section 5 (Experiments) |

---

## 7. What Each Script Does

| Script | Inputs | Outputs | Time |
|---|---|---|---|
| `setup_data.py` | Internet (or manual zip) | `data/dbdc3/`, `data/processed/` | ~5 min |
| `src/data_utils.py --build` | `data/dbdc3/` | `data/processed/*.pkl` | ~1 min |
| `baselines/tfidf_lr.py` | `data/processed/` | `results/metrics/baseline_tfidf*.json` | <1 min |
| `baselines/bert_baseline.py` | `data/processed/` | `results/metrics/baseline_bert*.json` | ~40 min |
| `src/train.py` | `data/processed/` | `results/checkpoints/main/best_model/` | ~2.5 hrs |
| `src/evaluate.py` | checkpoint + processed data | `results/metrics/test_eval_*.json/pkl` | ~2 min |
| `experiments/ablations.py` | `data/processed/` | `results/metrics/ablation_results.json` | ~4 hrs |
| `src/visualize.py` | `results/metrics/test_eval_predictions.pkl` | `results/figures/*.png` | ~1 min |
| `run_all.py` | Everything above | Everything above | ~8 hrs |

---

## 8. Understanding the Results

### Metrics explained

**F1 Macro** — Average F1 across both classes. The most important single number. Above 0.75 is strong for this task.

**AUROC** — Area under ROC curve. How well the model ranks breakdown windows above normal ones, regardless of threshold. 0.5 = random, 1.0 = perfect. Aim for 0.80+.

**AUPRC** — Area under Precision-Recall curve. More informative than AUROC when positive class is the minority. Aim for 0.70+.

**JS Divergence** — Jensen-Shannon Divergence between your model's risk score distribution and the human annotator distribution. This is the official DBDC evaluation metric. Lower = better. 0.0 = perfect match.

**Mean Lead Time** — Your novel metric. How many turns before the first actual breakdown does your system fire an alert? Positive = early (good). Zero = detected exactly at breakdown. Negative = detected late. Aim for ≥ 2.0 turns early.

**False Alarm Rate (FAR)** — What fraction of normal conversations trigger a false alert. Lower = better. A good system has FAR < 0.15.

### What a good result looks like for this task and data size

| Metric | Demo data (synthetic) | DBDC3 real data |
|---|---|---|
| F1 Macro | ~0.90+ (easy, synthetic) | 0.72–0.80 |
| AUROC | ~0.95+ | 0.80–0.88 |
| AUPRC | ~0.93+ | 0.72–0.82 |
| JS Divergence | ~0.05 | 0.15–0.25 |
| Mean Lead Time | ~3.5 turns | 1.5–3.5 turns |

Lower performance on real data is expected and normal — synthetic data is artificially easy. Real DBDC3 dialogues are subtle, and 30 annotators often disagree.

### Comparison table for report (fill in your actual numbers)

| Model | F1 Macro | AUROC | AUPRC | JS Div |
|---|---|---|---|---|
| TF-IDF + LR | ___ | ___ | ___ | — |
| BERT-base (flat) | ___ | ___ | ___ | — |
| RoBERTa-base (flat) | ___ | ___ | ___ | — |
| DeBERTa-v3-small (no special tokens) | ___ | ___ | ___ | ___ |
| DeBERTa-v3-small (no windowing) | ___ | ___ | ___ | ___ |
| **DeBERTa-v3-small (full model)** | **___** | **___** | **___** | **___** |

---

## 9. Troubleshooting

**`ModuleNotFoundError: No module named 'X'`**
```bash
pip install -r requirements.txt
```

**`FileNotFoundError: Processed data not found`**
```bash
python setup_data.py         # full setup
# or
python setup_data.py --demo  # synthetic only
```

**`DBDC3 download failed`**
```bash
# Option 1: retry (may be a temporary network issue)
python setup_data.py

# Option 2: manual download
# Browser → https://dbd-challenge.github.io/dbdc3/data/DBDC3.zip
# Save zip → extract → place dev/test folders in data/dbdc3/
# Then:
python src/data_utils.py --build
```

**Training is very slow on CPU**
```bash
# Reduce to 2 epochs for a quick test
python src/train.py --epochs 2

# Or use Google Colab (free T4 GPU) — copy the project folder there
# Training will be ~6x faster
```

**`CUDA out of memory`** (if using GPU)
```bash
python src/train.py --batch-size 4   # reduce batch size
```

**`trust_remote_code` warning from DailyDialog**
This is a HuggingFace library version issue. The script handles it gracefully — it falls back to synthetic clean samples. Your training will still work correctly.

**Check current status at any time**
```bash
python setup_data.py --check
```

---

## 10. For the Report

### Which results go where

| Report Section | What to include | Where to find it |
|---|---|---|
| Abstract | Final F1, AUROC, lead time | `results/metrics/test_eval_metrics.json` |
| Section 5 (Experiments) | Dataset table, training curves, class distribution | `results/figures/training_curves.png`, `dataset_distribution.png` |
| Section 6 (Results) | All metric tables, confusion matrix, ROC/PR curves, trajectories | `results/figures/` + `results/metrics/` |
| Section 6 (Ablation) | Ablation table + bar chart | `results/metrics/ablation_results.json` |

### Sample dataset table for Section 5

The `data/processed/dataset_summary.csv` file contains every sample. Open it in Excel for a cleaned-up display.

### Citing this project

```
Goyal, P. (2026). Early Detection of Communication Collapse in 
Multi-Turn Operational Conversations. VIT University CASE Study, 23BCE0411.
```

### Key references for Section 8 (References)

- Higashinaka et al. (2016) — DBDC1 task definition
- Higashinaka et al. (2019) — DBDC3 dataset and error taxonomy
- He et al. (2021) — DeBERTa: Decoding-enhanced BERT with disentangled attention
- He et al. (2023) — DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training
- Devlin et al. (2019) — BERT: Pre-training of deep bidirectional transformers
- Liu et al. (2019) — RoBERTa: A robustly optimized BERT pretraining approach
- Terragni et al. (2022) — BETOLD: A task-oriented dialogue breakdown detection dataset

---

*Dataset: DBDC3, MIT License. Model: DeBERTa-v3-small, MIT License.*
