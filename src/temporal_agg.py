"""
temporal_agg.py
===============
Temporal risk aggregation: applies Exponential Moving Average (EMA)
across overlapping windows of the same conversation.

Also handles:
  - Threshold sweep to find optimal tau on validation set
  - Alert triggering with persistence check (K consecutive windows)
  - Early Detection Lead Time computation

This is what converts a window-level score into an early warning system.
"""

import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ─── EMA Aggregation ──────────────────────────────────────────────────────────

def apply_ema(
    window_scores: list[float],
    alpha: float = 0.4
) -> list[float]:
    """
    Apply Exponential Moving Average to a list of window risk scores.
    
    EMA(t) = alpha * score(t) + (1 - alpha) * EMA(t-1)
    
    Args:
        window_scores: List of raw risk scores per window (same conversation)
        alpha: Smoothing factor. Higher = more weight to current window.
               Recommended: 0.4 (tunable)
    
    Returns:
        List of smoothed risk scores (same length as input)
    """
    if not window_scores:
        return []
    
    ema_scores = []
    ema = window_scores[0]  # Initialize with first score
    for score in window_scores:
        ema = alpha * score + (1 - alpha) * ema
        ema_scores.append(ema)
    
    return ema_scores


def apply_ema_to_predictions(
    predictions: list[dict],
    alpha: float = 0.4
) -> list[dict]:
    """
    Apply EMA aggregation to a list of prediction dicts, grouped by conv_id.
    
    Each prediction dict must have: {sample_id, conv_id, window_start, risk_score, label}
    Returns the same list with an additional 'ema_risk' field.
    """
    # Group by conversation
    by_conv = defaultdict(list)
    for pred in predictions:
        by_conv[pred["conv_id"]].append(pred)
    
    # Sort each conversation's windows by start position
    for conv_id in by_conv:
        by_conv[conv_id].sort(key=lambda x: x["window_start"])
    
    # Apply EMA per conversation
    results = []
    for conv_id, conv_preds in by_conv.items():
        raw_scores = [p["risk_score"] for p in conv_preds]
        ema_scores = apply_ema(raw_scores, alpha=alpha)
        for pred, ema in zip(conv_preds, ema_scores):
            results.append({**pred, "ema_risk": ema})
    
    return results


# ─── Alert Triggering ─────────────────────────────────────────────────────────

def trigger_alert(
    ema_scores: list[float],
    tau: float = 0.5,
    K: int = 2
) -> tuple[bool, int]:
    """
    Trigger an early warning alert if EMA risk exceeds tau for K consecutive windows.
    
    Returns:
        (alert_triggered: bool, alert_window_idx: int or -1)
    
    alert_window_idx: index of the window where the K-th consecutive threshold
                      crossing occurred (i.e., when the alert fires).
    """
    consecutive = 0
    for i, score in enumerate(ema_scores):
        if score > tau:
            consecutive += 1
            if consecutive >= K:
                return True, i
        else:
            consecutive = 0
    return False, -1


# ─── Early Detection Lead Time ────────────────────────────────────────────────

def compute_lead_time(
    predictions: list[dict],
    alpha: float = 0.4,
    tau: float = 0.5,
    K: int = 2,
    window_size: int = 5,
) -> dict:
    """
    For each conversation that actually has a breakdown, compute how many
    turns BEFORE the first actual breakdown turn the system fired an alert.
    
    Lead time > 0 means early detection.
    Lead time = 0 means detected at the breakdown.
    Lead time < 0 means detected after the breakdown (late).
    
    Returns dict with summary statistics.
    """
    # Group by conversation
    by_conv = defaultdict(list)
    for pred in predictions:
        by_conv[pred["conv_id"]].append(pred)
    
    lead_times = []
    missed = 0
    no_breakdown = 0
    
    for conv_id, conv_preds in by_conv.items():
        conv_preds.sort(key=lambda x: x["window_start"])
        
        # Find first actual breakdown turn
        first_bd_turn = None
        for pred in conv_preds:
            if pred.get("label", 0) == 1:
                first_bd_turn = pred["window_start"]  # turn at which breakdown window starts
                break
        
        if first_bd_turn is None:
            no_breakdown += 1
            continue
        
        # Apply EMA and check alert
        raw_scores = [p["risk_score"] for p in conv_preds]
        ema_scores = apply_ema(raw_scores, alpha=alpha)
        alert, alert_idx = trigger_alert(ema_scores, tau=tau, K=K)
        
        if not alert:
            missed += 1
            continue
        
        # Map alert window index to turn index
        alert_turn = conv_preds[alert_idx]["window_start"]
        lead_time = first_bd_turn - alert_turn  # positive = early detection
        lead_times.append(lead_time)
    
    if not lead_times:
        return {
            "mean_lead_time": 0.0,
            "median_lead_time": 0.0,
            "n_early": 0,
            "n_late": 0,
            "n_exact": 0,
            "n_missed": missed,
            "n_no_breakdown": no_breakdown,
        }
    
    lt = np.array(lead_times)
    return {
        "mean_lead_time": float(np.mean(lt)),
        "median_lead_time": float(np.median(lt)),
        "std_lead_time": float(np.std(lt)),
        "n_early": int(np.sum(lt > 0)),
        "n_late": int(np.sum(lt < 0)),
        "n_exact": int(np.sum(lt == 0)),
        "n_missed": missed,
        "n_no_breakdown": no_breakdown,
        "lead_time_distribution": lt.tolist(),
    }


# ─── Threshold Sweep ──────────────────────────────────────────────────────────

def sweep_threshold(
    predictions: list[dict],
    alpha: float = 0.4,
    tau_range: Optional[list] = None,
    K: int = 2,
    metric: str = "f1",
) -> dict:
    """
    Sweep tau threshold on validation predictions to find optimal operating point.
    
    Args:
        predictions: List of {conv_id, window_start, risk_score, label} dicts
        alpha: EMA smoothing factor
        tau_range: List of thresholds to try (default: 0.3 to 0.8 in 0.05 steps)
        K: Persistence window for alert
        metric: "f1" or "f1_pos" — metric to maximize
    
    Returns:
        dict with best_tau, best_metric, full sweep results
    """
    from sklearn.metrics import f1_score, roc_auc_score

    if tau_range is None:
        tau_range = [round(t, 2) for t in np.arange(0.25, 0.85, 0.05)]

    # Apply EMA once
    preds_with_ema = apply_ema_to_predictions(predictions, alpha=alpha)
    
    # Collect (ema_risk, label) per window for threshold sweep
    ema_scores = [p["ema_risk"] for p in preds_with_ema]
    true_labels = [p["label"] for p in preds_with_ema]

    sweep_results = []
    best_tau = 0.5
    best_metric_val = -1

    for tau in tau_range:
        preds_binary = [1 if s >= tau else 0 for s in ema_scores]
        f1_macro = f1_score(true_labels, preds_binary, average="macro", zero_division=0)
        f1_pos   = f1_score(true_labels, preds_binary, average="binary", zero_division=0)
        tp = sum(1 for p, l in zip(preds_binary, true_labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds_binary, true_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(preds_binary, true_labels) if p == 0 and l == 1)
        tn = sum(1 for p, l in zip(preds_binary, true_labels) if p == 0 and l == 0)
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        result = {
            "tau": tau, "f1_macro": f1_macro, "f1_pos": f1_pos,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "far": far
        }
        sweep_results.append(result)

        metric_val = f1_macro if metric == "f1" else f1_pos
        if metric_val > best_metric_val:
            best_metric_val = metric_val
            best_tau = tau

    return {
        "best_tau": best_tau,
        "best_metric": best_metric_val,
        "sweep": sweep_results,
    }


# ─── Demo / Test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Demo conversation with a gradual breakdown
    print("=== EMA Demo ===")
    raw_scores = [0.1, 0.15, 0.2, 0.3, 0.45, 0.6, 0.7, 0.75, 0.8]
    ema_scores = apply_ema(raw_scores, alpha=0.4)
    print("Raw scores: ", [f"{s:.2f}" for s in raw_scores])
    print("EMA scores: ", [f"{s:.2f}" for s in ema_scores])
    
    alert, idx = trigger_alert(ema_scores, tau=0.5, K=2)
    print(f"\nAlert triggered: {alert} at window {idx}")
    if alert:
        print(f"  (Breakdown detected at window {idx} before final turn {len(raw_scores)-1})")
        print(f"  Lead time: {len(raw_scores) - 1 - idx} windows early")
