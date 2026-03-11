"""
evaluate.py
===========

Complete evaluation suite:
  - Binary F1, AUROC, AUPRC
  - JS Divergence (DBDC official metric)
  - Early Detection Lead Time (novel metric)
  - False Alarm Rate
  - Threshold sweep + optimal tau selection
  - Per-conversation risk trajectory collection

Run:
    python src/evaluate.py --checkpoint results/checkpoints/main/best_model
    python src/evaluate.py --checkpoint results/checkpoints/main/best_model --tau 0.5
"""

import sys
import json
import argparse
import pickle
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)
from scipy.spatial.distance import jensenshannon

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_utils import load_processed
from tokenize_utils import load_tokenizer, ConversationWindowDataset
from model import load_model
from temporal_agg import apply_ema_to_predictions, sweep_threshold, compute_lead_time

RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
TOK_CACHE   = ROOT / "data" / "tokenizer_cache"


@torch.no_grad()
def run_inference(model, dataset, batch_size: int = 16, device=None) -> list[dict]:
    from torch.utils.data import DataLoader
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    model.to(device)

    results = []
    windows = dataset.windows

    for i, batch in enumerate(tqdm(loader, desc="Inference", leave=False)):
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        
        outputs = model(input_ids, attention_mask, token_type_ids)
        risk_scores = outputs["risk"].cpu().squeeze(-1).tolist()
        if isinstance(risk_scores, float):
            risk_scores = [risk_scores]

        batch_size_actual = input_ids.size(0)
        start_idx = i * batch_size
        for j in range(batch_size_actual):
            w = windows[start_idx + j]
            score = risk_scores[j] if isinstance(risk_scores, list) else risk_scores
            results.append({
                "sample_id":    w["sample_id"],
                "conv_id":      w["conv_id"],
                "window_start": w["window_start"],
                "window_end":   w.get("window_end", w["window_start"] + 5),
                "risk_score":   float(score),
                "label":        int(w["label"]),
                "soft_label":   float(w.get("soft_label", w["label"])),
                "source":       w.get("source", "unknown"),
            })

    return results


def compute_js_divergence(pred_probs: list[float], soft_labels: list[float]) -> float:
    eps = 1e-9
    preds = np.clip(pred_probs, eps, 1 - eps)
    truth = np.clip(soft_labels, eps, 1 - eps)
    
    p_dist = np.column_stack([preds, 1 - preds])
    q_dist = np.column_stack([truth, 1 - truth])
    
    js_vals = []
    for p, q in zip(p_dist, q_dist):
        js = jensenshannon(p, q, base=2)
        js_vals.append(js)
    
    return float(np.mean(js_vals))


def evaluate_full(
    predictions: list[dict],
    tau: float = None,
    alpha: float = 0.4,
    K: int = 2,
    run_name: str = "eval",
    save: bool = True,
) -> dict:

    preds_ema = apply_ema_to_predictions(predictions, alpha=alpha)
    
    raw_probs  = [p["risk_score"] for p in preds_ema]
    ema_probs  = [p["ema_risk"]   for p in preds_ema]
    labels     = [p["label"]      for p in preds_ema]
    soft_labels = [p["soft_label"] for p in preds_ema]

    if tau is None:
        sweep = sweep_threshold(predictions, alpha=alpha, K=K)
        tau = sweep["best_tau"]
    else:
        sweep = None

    preds_binary = [1 if p >= tau else 0 for p in ema_probs]

    f1_macro  = f1_score(labels, preds_binary, average="macro", zero_division=0)
    f1_pos    = f1_score(labels, preds_binary, average="binary", zero_division=0)
    f1_neg    = f1_score(labels, preds_binary, average="binary", pos_label=0, zero_division=0)

    cm = confusion_matrix(labels, preds_binary)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    try:
        auroc = roc_auc_score(labels, ema_probs)
    except ValueError:
        auroc = 0.0

    try:
        auprc = average_precision_score(labels, ema_probs)
    except ValueError:
        auprc = 0.0

    js_div = compute_js_divergence(ema_probs, soft_labels)

    lead_time_stats = compute_lead_time(preds_ema, alpha=alpha, tau=tau, K=K)

    metrics = {
        "tau": tau,
        "alpha": alpha,
        "K": K,
        "f1_macro":     f1_macro,
        "f1_positive":  f1_pos,
        "f1_negative":  f1_neg,
        "precision":    precision,
        "recall":       recall,
        "far":          far,
        "auroc":        auroc,
        "auprc":        auprc,
        "js_divergence": js_div,
        "mean_lead_time":   lead_time_stats["mean_lead_time"],
        "median_lead_time": lead_time_stats["median_lead_time"],
        "n_early_detections": lead_time_stats.get("n_early", 0),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_samples": len(labels),
        "n_positive": int(sum(labels)),
    }

    print_metrics(metrics)

    if save:
        METRICS_DIR.mkdir(parents=True, exist_ok=True)

        out = METRICS_DIR / f"{run_name}_metrics.json"
        with open(out, "w") as f:
            json.dump(metrics, f, indent=2)

        preds_path = METRICS_DIR / f"{run_name}_predictions.pkl"
        with open(preds_path, "wb") as f:
            pickle.dump(preds_ema, f)

        if sweep:
            sweep_path = METRICS_DIR / f"{run_name}_threshold_sweep.json"
            with open(sweep_path, "w") as f:
                json.dump(sweep, f, indent=2)

    return metrics


def print_metrics(m: dict):
    print("\n" + "="*50)
    print("  EVALUATION RESULTS")
    print("="*50)
    print(f"  Threshold (tau):    {m['tau']:.2f}")
    print(f"  Samples:            {m['n_samples']} ({m['n_positive']} positive)")
    print()
    print(f"  F1 Macro:           {m['f1_macro']:.4f}")
    print(f"  F1 Positive:        {m['f1_positive']:.4f}")
    print(f"  Precision:          {m['precision']:.4f}")
    print(f"  Recall:             {m['recall']:.4f}")
    print(f"  False Alarm Rate:   {m['far']:.4f}")
    print()
    print(f"  AUROC:              {m['auroc']:.4f}")
    print(f"  AUPRC:              {m['auprc']:.4f}")
    print(f"  JS Divergence:      {m['js_divergence']:.4f}")
    print()
    print(f"  Mean Lead Time:     {m['mean_lead_time']:.2f}")
    print(f"  Median Lead Time:   {m['median_lead_time']:.2f}")
    print(f"  Early Detections:   {m['n_early_detections']}")
    print()
    print(f"  TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}")
    print("="*50)


def collect_trajectories(predictions: list[dict], alpha: float = 0.4, n: int = 10) -> list[dict]:
    preds_ema = apply_ema_to_predictions(predictions, alpha=alpha)
    
    by_conv = defaultdict(list)
    for p in preds_ema:
        by_conv[p["conv_id"]].append(p)
    
    bd_convs = [cid for cid, ps in by_conv.items() if any(p["label"] == 1 for p in ps)]
    ok_convs = [cid for cid, ps in by_conv.items() if all(p["label"] == 0 for p in ps)]
    
    selected = bd_convs[:n//2] + ok_convs[:n//2]
    
    trajectories = []
    for cid in selected:
        ps = sorted(by_conv[cid], key=lambda x: x["window_start"])
        trajectories.append({
            "conv_id": cid,
            "has_breakdown": any(p["label"] == 1 for p in ps),
            "window_starts": [p["window_start"] for p in ps],
            "raw_scores":    [p["risk_score"] for p in ps],
            "ema_scores":    [p["ema_risk"] for p in ps],
            "labels":        [p["label"] for p in ps],
        })
    
    return trajectories


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--run-name", default="eval")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─────────────────────────────────────────────
    # Load tokenizer and trained model
    # ─────────────────────────────────────────────
    tokenizer = load_tokenizer(save_path=TOK_CACHE)
    model = load_model(Path(args.checkpoint), vocab_size=len(tokenizer), device=str(device))

    # ─────────────────────────────────────────────
    # LOAD TEST DATA (FIXED)
    # ─────────────────────────────────────────────
    windows = load_processed("test")

    print(f"\nLoaded {len(windows)} test windows for evaluation.")

    dataset = ConversationWindowDataset(windows, tokenizer)

    # ─────────────────────────────────────────────
    # Run inference
    # ─────────────────────────────────────────────
    predictions = run_inference(model, dataset, batch_size=16, device=device)

    # ─────────────────────────────────────────────
    # Compute metrics
    # ─────────────────────────────────────────────
    metrics = evaluate_full(
        predictions,
        tau=args.tau,
        alpha=args.alpha,
        run_name=args.run_name,
        save=True,
    )