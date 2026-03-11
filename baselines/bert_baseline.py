"""
baselines/bert_baseline.py
==========================
BERT-base / RoBERTa-base fine-tune baseline.
Uses flat input (no special tokens) for fair comparison.

Run:
    python baselines/bert_baseline.py --model bert-base-uncased
    python baselines/bert_baseline.py --model roberta-base
"""

import sys
import json
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_utils import load_processed

METRICS_DIR = ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class FlatWindowDataset(Dataset):
    """
    Flat (no special tokens) tokenized window dataset.
    All turn texts are concatenated with a separator.
    """

    def __init__(self, windows, tokenizer, max_length=256):
        self.windows = windows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._cache = {}

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        if idx not in self._cache:
            text = " [SEP] ".join(t["text"] for t in w["turns"])
            enc = self.tokenizer(text, max_length=self.max_length,
                                 padding="max_length", truncation=True,
                                 return_tensors="pt")
            self._cache[idx] = {k: v.squeeze(0) for k, v in enc.items()}
        item = dict(self._cache[idx])
        item["labels"] = torch.tensor(float(w["label"]), dtype=torch.float)
        return item


# ─── Model ────────────────────────────────────────────────────────────────────

class SimpleClassifier(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(hidden), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kw = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        out = self.encoder(**kw)
        cls = out.last_hidden_state[:, 0, :]
        return self.head(cls)


# ─── Training ─────────────────────────────────────────────────────────────────

def train_baseline(model_name: str, epochs: int = 5, batch_size: int = 8, lr: float = 2e-5):
    safe_name = model_name.replace("/", "_").replace("-", "_")
    print(f"\n{'='*55}")
    print(f"  BASELINE: {model_name}")
    print(f"{'='*55}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    train_w = load_processed("train")
    dev_w   = load_processed("dev")
    test_w  = load_processed("test")

    train_ds = FlatWindowDataset(train_w, tokenizer)
    dev_ds   = FlatWindowDataset(dev_w, tokenizer)
    test_ds  = FlatWindowDataset(test_w, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=16, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=16, shuffle=False)

    model = SimpleClassifier(model_name).to(device)
    n_pos = sum(w["label"] for w in train_w)
    n_neg = len(train_w) - n_pos
    pos_weight = torch.tensor([min(n_neg / max(n_pos, 1), 5.0)]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tti  = batch.get("token_type_ids")
            if tti is not None:
                tti = tti.to(device)
            labs = batch["labels"].to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits = model(ids, mask, tti)
            loss = loss_fn(logits, labs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in dev_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                tti  = batch.get("token_type_ids")
                if tti is not None:
                    tti = tti.to(device)
                labs = batch["labels"].to(device).unsqueeze(1)
                logits = model(ids, mask, tti)
                val_loss += loss_fn(logits, labs).item()

        avg_val = val_loss / len(dev_loader)
        avg_train = total_loss / len(train_loader)
        print(f"  Epoch {epoch}: train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Test evaluation
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tti  = batch.get("token_type_ids")
            if tti is not None:
                tti = tti.to(device)
            logits = model(ids, mask, tti)
            probs = torch.sigmoid(logits).cpu().squeeze().tolist()
            labs  = batch["labels"].long().cpu().tolist()
            if isinstance(probs, float):
                probs, labs = [probs], [labs]
            all_probs.extend(probs)
            all_labels.extend(labs)

    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    f1_macro = f1_score(all_labels, preds, average="macro", zero_division=0)
    f1_pos   = f1_score(all_labels, preds, average="binary", zero_division=0)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)
    except Exception:
        auroc = auprc = 0.0

    metrics = {
        "model": model_name,
        "f1_macro": f1_macro,
        "f1_positive": f1_pos,
        "auroc": auroc,
        "auprc": auprc,
    }
    print(f"\n  TEST → F1={f1_macro:.4f} | AUROC={auroc:.4f} | AUPRC={auprc:.4f}")

    out = METRICS_DIR / f"baseline_{safe_name}_metrics.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {out}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="bert-base-uncased",
                        choices=["bert-base-uncased", "roberta-base"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    train_baseline(args.model, args.epochs, args.batch_size)
