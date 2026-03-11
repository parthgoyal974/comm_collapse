"""
experiments/ablations.py
========================
Runs all 6 ablation conditions defined in the research plan.

Conditions:
  1. Full Model (DeBERTa-v3-small + all tokens + windowing + EMA)
  2. No role tokens
  3. No expectation tokens
  4. No special tokens at all (flat input)
  5. No sliding window (single-turn only — last turn)
  6. No EMA aggregation (raw window scores)

Each condition trains a fresh model for 5 epochs on the same data,
evaluates on the test set, and saves results.

Run:
    python experiments/ablations.py
    python experiments/ablations.py --fast  (2 epochs, for quick testing)
"""

import sys
import json
import argparse
import re
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_utils import load_processed
from tokenize_utils import (load_tokenizer, serialize_window, has_expectation,
                             ConversationWindowDataset, SPECIAL_TOKENS)
from model import CommunicationRiskModel, get_loss_fn
from temporal_agg import apply_ema_to_predictions, sweep_threshold

METRICS_DIR = ROOT / "results" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
TOK_CACHE = ROOT / "data" / "tokenizer_cache"


# ─── Ablation Input Serializers ───────────────────────────────────────────────

def serialize_no_role_tokens(turns, window_size=5):
    """Remove [SPEAKER_X] tokens. Keep turn index and expectation."""
    parts = []
    for i, turn in enumerate(turns[:window_size]):
        expect = "[EXPECT_RESPONSE]" if has_expectation(turn["text"]) else "[NO_EXPECT]"
        parts.append(f"[TURN_{i+1}] {expect} {turn['text'].strip()}")
    return " ".join(parts)


def serialize_no_expect_tokens(turns, window_size=5):
    """Remove [EXPECT_RESPONSE]/[NO_EXPECT]. Keep role and turn index."""
    parts = []
    for i, turn in enumerate(turns[:window_size]):
        speaker = f"[SPEAKER_{turn['speaker']}]" if turn['speaker'] in ('SYS', 'USR') else "[SPEAKER_SYS]"
        parts.append(f"[TURN_{i+1}] {speaker} {turn['text'].strip()}")
    return " ".join(parts)


def serialize_flat(turns, window_size=5):
    """No special tokens at all. Just plain text with separator."""
    return " | ".join(t["text"].strip() for t in turns[:window_size])


def serialize_single_turn(turns, window_size=5):
    """Single-turn: only the last turn in the window."""
    last = turns[min(len(turns)-1, window_size-1)]
    return last["text"].strip()


# ─── Ablation Dataset ─────────────────────────────────────────────────────────

class AblationDataset(Dataset):
    """Pre-tokenizes all samples at construction for CPU speed."""
    def __init__(self, windows, tokenizer, serializer_fn, max_length=256):
        self.windows      = windows
        self.serializer_fn = serializer_fn
        # Pre-tokenize everything upfront (same optimization as main train pipeline)
        self._encoded = []
        for w in windows:
            text = serializer_fn(w["turns"])
            enc  = tokenizer(text, max_length=max_length,
                             padding="max_length", truncation=True,
                             return_tensors="pt")
            self._encoded.append({k: v.squeeze(0) for k, v in enc.items()})

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        item = dict(self._encoded[idx])
        w    = self.windows[idx]
        item["labels"]       = torch.tensor(float(w["label"]), dtype=torch.float)
        item["sample_id"]    = w["sample_id"]
        item["conv_id"]      = w["conv_id"]
        item["window_start"] = w["window_start"]
        return item


# ─── Quick Train + Eval Loop ──────────────────────────────────────────────────

def quick_train_eval(
    condition_name: str,
    train_windows, dev_windows, test_windows,
    tokenizer: AutoTokenizer,
    serializer_fn,
    model_name: str = "microsoft/deberta-v3-small",
    epochs: int = 5,
    batch_size: int = 32,   # updated from 8 to match main training config
    alpha: float = 0.4,
    use_ema: bool = True,
) -> dict:
    print(f"\n  [{condition_name}]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def make_loader(windows, shuffle):
        ds = AblationDataset(windows, tokenizer, serializer_fn)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(train_windows, shuffle=True)
    dev_loader   = make_loader(dev_windows,   shuffle=False)
    test_loader  = make_loader(test_windows,  shuffle=False)

    model = CommunicationRiskModel(model_name=model_name, vocab_size=len(tokenizer))
    model = model.to(device)

    n_pos = sum(w["label"] for w in train_windows)
    n_neg = len(train_windows) - n_pos
    pw = min(n_neg / max(n_pos, 1), 5.0)
    loss_fn = get_loss_fn("bce", pos_weight=pw)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"    Epoch {epoch}/{epochs}", leave=False):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tti  = batch.get("token_type_ids")
            if tti is not None:
                tti = tti.to(device)
            labs = batch["labels"].to(device).unsqueeze(1)
            if hasattr(loss_fn, "pos_weight") and loss_fn.pos_weight is not None:
                loss_fn.pos_weight = loss_fn.pos_weight.to(device)
            optimizer.zero_grad()
            out = model(ids, mask, tti)
            loss = loss_fn(out["logits"], labs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # Validation loss
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
                if hasattr(loss_fn, "pos_weight") and loss_fn.pos_weight is not None:
                    loss_fn.pos_weight = loss_fn.pos_weight.to(device)
                out = model(ids, mask, tti)
                val_loss += loss_fn(out["logits"], labs).item()
        val_loss /= len(dev_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Test inference
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    preds_raw = []
    with torch.no_grad():
        for batch in test_loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tti  = batch.get("token_type_ids")
            if tti is not None:
                tti = tti.to(device)
            out = model(ids, mask, tti)
            scores = out["risk"].cpu().squeeze(-1).tolist()
            if isinstance(scores, float):
                scores = [scores]
            for i, score in enumerate(scores):
                batch_offset = 0
                w_idx = batch_offset + i
                preds_raw.append({
                    "sample_id":    batch["sample_id"][i] if isinstance(batch["sample_id"], list) else str(i),
                    "conv_id":      batch["conv_id"][i] if isinstance(batch["conv_id"], list) else str(i),
                    "window_start": int(batch["window_start"][i]),
                    "risk_score":   float(score),
                    "label":        int(batch["labels"][i].item()),
                    "soft_label":   float(batch["labels"][i].item()),
                })

    # Apply or skip EMA
    if use_ema:
        from temporal_agg import apply_ema_to_predictions
        preds_agg = apply_ema_to_predictions(preds_raw, alpha=alpha)
        final_scores = [p["ema_risk"] for p in preds_agg]
    else:
        final_scores = [p["risk_score"] for p in preds_raw]

    labels = [p["label"] for p in preds_raw]
    preds_bin = [1 if s >= 0.5 else 0 for s in final_scores]

    f1 = f1_score(labels, preds_bin, average="macro", zero_division=0)
    try:
        auroc = roc_auc_score(labels, final_scores)
        auprc = average_precision_score(labels, final_scores)
    except Exception:
        auroc = auprc = 0.0

    print(f"    → F1={f1:.4f} | AUROC={auroc:.4f} | AUPRC={auprc:.4f}")
    return {"condition": condition_name, "f1_macro": f1, "auroc": auroc, "auprc": auprc}


# ─── Run All Ablations ────────────────────────────────────────────────────────

def run_all_ablations(epochs: int = 5):
    print("\n" + "="*60)
    print("  ABLATION STUDY")
    print("="*60)

    train_windows = load_processed("train")
    dev_windows   = load_processed("dev")
    test_windows  = load_processed("test")

    tokenizer = load_tokenizer(save_path=TOK_CACHE)

    from tokenize_utils import serialize_window as full_serialize

    conditions = [
        # (name, serializer_fn, use_ema)
        ("Full Model",              full_serialize,              True),
        ("No Role Tokens",          serialize_no_role_tokens,    True),
        ("No Expect Tokens",        serialize_no_expect_tokens,  True),
        ("No Special Tokens",       serialize_flat,              True),
        ("Single Turn Only",        serialize_single_turn,       True),
        ("No EMA Aggregation",      full_serialize,              False),
    ]

    all_results = []
    for name, serializer, use_ema in conditions:
        result = quick_train_eval(
            condition_name=name,
            train_windows=train_windows,
            dev_windows=dev_windows,
            test_windows=test_windows,
            tokenizer=tokenizer,
            serializer_fn=serializer,
            epochs=epochs,
            use_ema=use_ema,
        )
        all_results.append(result)

    # Print summary table
    print("\n" + "="*60)
    print(f"  {'Condition':<30} {'F1 Macro':>10} {'AUROC':>10} {'AUPRC':>10}")
    print("  " + "-"*58)
    for r in all_results:
        marker = "◀ FULL MODEL" if "Full" in r["condition"] else ""
        print(f"  {r['condition']:<30} {r['f1_macro']:>10.4f} {r['auroc']:>10.4f} {r['auprc']:>10.4f}  {marker}")
    print("="*60)

    # Save
    out = METRICS_DIR / "ablation_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {out}")

    # Also save as dict for visualize.py
    ablation_dict = {r["condition"]: r for r in all_results}
    out2 = METRICS_DIR / "ablation_results_dict.json"
    with open(out2, "w") as f:
        json.dump(ablation_dict, f, indent=2)

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run 2 epochs (quick test)")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    epochs = 2 if args.fast else args.epochs
    run_all_ablations(epochs=epochs)
