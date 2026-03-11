"""
train.py
========
Full training pipeline for CommunicationRiskModel.

Features:
  - Differential learning rates (backbone vs. head)
  - Linear LR warmup + decay scheduler
  - Early stopping on validation loss
  - Gradient clipping
  - Checkpointing (best + last)
  - Loss/metric logging to CSV
  - CPU optimizations: thread pinning, DataLoader prefetch

Run:
    python src/train.py
"""

import sys
import os
import argparse
import json
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from tqdm import tqdm

import torch._dynamo
torch._dynamo.config.suppress_errors = True

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_utils import load_processed
from tokenize_utils import load_tokenizer, build_dataloaders
from model import CommunicationRiskModel, get_loss_fn, save_model


RESULTS_DIR = ROOT / "results"
CKPT_DIR    = RESULTS_DIR / "checkpoints"
METRICS_DIR = RESULTS_DIR / "metrics"
TOK_CACHE   = ROOT / "data" / "tokenizer_cache"


# ─────────────────────────────────────────────────────────────
# Training Config
# ─────────────────────────────────────────────────────────────

def get_default_config():
    return {
        "model_name":    "microsoft/deberta-v3-small",
        "max_length":    256,
        "batch_size":    32,

        # Reduced epochs
        "epochs":        4,

        "lr_backbone":   4e-5,
        "lr_head":       2e-4,
        "weight_decay":  0.01,
        "warmup_ratio":  0.1,
        "max_grad_norm": 1.0,
        "dropout":       0.1,
        "loss_type":     "bce",
        "pos_weight":    2.0,
        "patience":      3,
        "freeze_layers": 0,
        "seed":          42,

        # CPU optimizations
        "num_workers":   4,
        "pin_memory":    True,
    }


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cpu_threads():
    n_cores = os.cpu_count() or 4
    n_threads = max(1, n_cores - 1)
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(max(1, n_threads // 2))

    print(f"  CPU threads: {n_threads} / {n_cores} cores")
    return n_threads


def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  Using GPU: {torch.cuda.get_device_name(0)}")

    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("  Using MPS")

    else:
        dev = torch.device("cpu")
        n_threads = configure_cpu_threads()
        print(f"  Using CPU ({n_threads} threads)")

    return dev


def compute_pos_weight(windows):
    labels  = [w["label"] for w in windows]
    n_pos   = sum(labels)
    n_neg   = len(labels) - n_pos

    if n_pos == 0:
        return 1.0

    weight = n_neg / n_pos

    print(f"  Class balance → neg: {n_neg}, pos: {n_pos}, pos_weight: {weight:.2f}")

    return min(weight, 5.0)


# ─────────────────────────────────────────────────────────────
# Train / Eval
# ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    max_grad_norm,
    epoch,
):

    model.train()

    total_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch} [train]", leave=False)

    for batch in pbar:

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device).unsqueeze(1)

        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask, token_type_ids)
        logits  = outputs["logits"]

        if hasattr(loss_fn, "pos_weight") and loss_fn.pos_weight is not None:
            loss_fn.pos_weight = loss_fn.pos_weight.to(device)

        loss = loss_fn(logits, labels)

        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # LOWER THRESHOLD → HIGHER RECALL
        preds = (torch.sigmoid(logits) >= 0.4).long().cpu().squeeze().tolist()
        labs  = labels.long().cpu().squeeze().tolist()

        if isinstance(preds, int):
            preds = [preds]
            labs  = [labs]

        all_preds.extend(preds)
        all_labels.extend(labs)

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)

    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {"loss": avg_loss, "f1": f1}


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, split="val"):

    model.eval()

    total_loss = 0.0
    all_probs  = []
    all_labels = []

    for batch in tqdm(loader, desc=f"[{split}]", leave=False):

        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device).unsqueeze(1)

        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        if hasattr(loss_fn, "pos_weight") and loss_fn.pos_weight is not None:
            loss_fn.pos_weight = loss_fn.pos_weight.to(device)

        outputs = model(input_ids, attention_mask, token_type_ids)

        logits = outputs["logits"]

        loss = loss_fn(logits, labels)

        total_loss += loss.item()

        all_probs.extend(outputs["risk"].cpu().squeeze().tolist())
        all_labels.extend(labels.long().cpu().squeeze().tolist())

    avg_loss = total_loss / len(loader)

    preds_binary = [1 if p >= 0.4 else 0 for p in all_probs]

    metrics = {
        "loss": avg_loss,
        "f1_macro": f1_score(all_labels, preds_binary, average="macro", zero_division=0),
        "f1_pos": f1_score(all_labels, preds_binary, average="binary", zero_division=0),
    }

    try:
        metrics["auroc"] = roc_auc_score(all_labels, all_probs)
    except:
        metrics["auroc"] = 0.0

    try:
        metrics["auprc"] = average_precision_score(all_labels, all_probs)
    except:
        metrics["auprc"] = 0.0

    return metrics


# ─────────────────────────────────────────────────────────────
# Main Training
# ─────────────────────────────────────────────────────────────

def train(config, run_name="main"):

    set_seed(config["seed"])
    device = get_device()

    print("\n[1/5] Loading processed data...")

    train_windows = load_processed("train")
    dev_windows   = load_processed("dev")

    import random as _rnd

    def _stratified(windows, frac):
        pos = [w for w in windows if w["label"] == 1]
        neg = [w for w in windows if w["label"] == 0]
        _rnd.seed(42)
        return (_rnd.sample(pos, max(1, int(len(pos) * frac))) +
                _rnd.sample(neg, max(1, int(len(neg) * frac))))

    train_windows = _stratified(train_windows, 0.25)
    dev_windows   = _stratified(dev_windows,   0.25)

    print("\n[2/5] Loading tokenizer...")

    tokenizer = load_tokenizer(config["model_name"], save_path=TOK_CACHE)

    print("\n[3/5] Building dataloaders...")

    loaders = build_dataloaders(
        {"train": train_windows, "dev": dev_windows},
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        max_length=config["max_length"],
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    print("\n[4/5] Building model...")

    model = CommunicationRiskModel(
        model_name=config["model_name"],
        vocab_size=len(tokenizer),
        dropout=config["dropout"],
        freeze_layers=config["freeze_layers"],
    )

    model = model.to(device)

    params = model.count_parameters()

    print(f"  Total params: {params['total']:,} | Trainable: {params['trainable']:,}")

    auto_pos_weight = compute_pos_weight(train_windows)

    pw = config.get("pos_weight") or auto_pos_weight

    loss_fn = get_loss_fn(config["loss_type"], pos_weight=pw)

    backbone_params = [p for n, p in model.named_parameters() if "risk_head" not in n and p.requires_grad]
    head_params     = [p for n, p in model.named_parameters() if "risk_head" in n and p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": config["lr_backbone"]},
        {"params": head_params, "lr": config["lr_head"]},
    ], weight_decay=config["weight_decay"])

    total_steps  = len(loaders["train"]) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print("\n[5/5] Training...")

    ckpt_dir = CKPT_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config["epochs"] + 1):

        train_metrics = train_one_epoch(
            model, loaders["train"], optimizer, scheduler,
            loss_fn, device, config["max_grad_norm"], epoch
        )

        val_metrics = evaluate(model, loaders["dev"], loss_fn, device, "dev")

        print(
            f"Epoch {epoch}/{config['epochs']} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:

            best_val_loss = val_metrics["loss"]
            patience_counter = 0

            save_model(model, ckpt_dir / "best_model",
                       metadata={"epoch": epoch, "config": config})

            print("  ✓ Best model saved")

        else:

            patience_counter += 1

            if patience_counter >= config["patience"]:
                print("  Early stopping")
                break

    save_model(model, ckpt_dir / "last_model",
               metadata={"epoch": epoch, "config": config})

    print("\nTraining complete.")

    return ckpt_dir / "best_model"


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="microsoft/deberta-v3-small")
    parser.add_argument("--loss", default="bce",
                        choices=["bce", "focal", "bce_unweighted"])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-backbone", type=float, default=4e-5)
    parser.add_argument("--lr-head", type=float, default=2e-4)
    parser.add_argument("--freeze-layers", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--run-name", default="main")

    args = parser.parse_args()

    config = get_default_config()

    config.update({
        "model_name": args.model,
        "loss_type": args.loss,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr_backbone": args.lr_backbone,
        "lr_head": args.lr_head,
        "freeze_layers": args.freeze_layers,
        "num_workers": args.num_workers,
    })

    print("\n=== Training Configuration ===")

    for k, v in config.items():
        print(f"  {k}: {v}")

    best_path = train(config, run_name=args.run_name)

    print(f"\nDone! Best model: {best_path}")