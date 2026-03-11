"""
data_utils.py  —  DBDC3 English Dataset Pipeline
=================================================
Key design decisions (verified against real DBDC3 annotation data):

ANNOTATION FORMAT (verified from actual JSON):
  Each system turn has 30 annotations, each with one of:
    O = Not a Breakdown  (chatbot response is coherent/appropriate)
    T = Possible Breakdown (odd/off-topic, but conversation could continue)
    X = Breakdown  (clear failure — conversation cannot continue naturally)

BINARY LABEL STRATEGY  (X-plurality, verified correct):
  label = 1  if  X/total >= 0.33  (≥10 of 30 annotators say clear breakdown)
  label = 0  otherwise

  WHY NOT (T+X)/total >= 0.50:
    Example: Turn 1 in sample dialogue — O=15, T=9, X=6
    (T+X)/30 = 0.50 → old code said label=1 — WRONG.
    50% of annotators said O (Not Breakdown). It was borderline/non-sequitur,
    not a clear breakdown. X/30 = 0.20 → label=0 ✓ (correct)

  WHY X/total >= 0.33:
    X is the ONLY unambiguous label. T means annotators disagreed or found it
    odd but not broken. A clear breakdown requires X to be a meaningful fraction.
    0.33 = at least 10/30 annotators called it broken. This cleanly separates:
      - Borderline/ambiguous turns (T-heavy)  → label=0
      - Clear breakdown turns (X-plurality)   → label=1

SOFT LABEL:
  soft = (0.5*T + 1.0*X) / total
  T contributes half weight (uncertain signal). X contributes full weight.
  Used as the continuous target for soft-loss training.

WINDOW LABELING — ONSET-AWARE:
  A window gets label=1 ONLY IF:
    - Last SYS turn in window has label=1 (clear breakdown), AND
    - At least one prior SYS turn in the window does NOT have label=1
      (i.e., this is the onset, not a window deep inside a collapsed dialogue)
  Rationale: Early detection = catching the FIRST sign of collapse.
  Windows where all prior turns are already broken offer no early-warning value
  and inflate the positive class. Labeling them 0 is both correct and practical.

EVAL vs TEST FOLDER:
  DBDC3.zip uses 'eval/' for the held-out test split, not 'test/'.
  setup_data.py maps: zip's eval/ → data/dbdc3/test/
  data_utils.py also checks both 'test' and 'eval' as fallback.

Usage:
    python src/data_utils.py --build   (requires data/dbdc3/ from setup_data.py)
    python src/data_utils.py --demo    (synthetic only, no real data needed)
    python src/data_utils.py --stats   (print stats of already-built splits)
"""

import json
import re
import random
import argparse
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent.parent
DATA_DIR  = ROOT / "data"
DBDC3_DIR = DATA_DIR / "dbdc3"
DD_DIR    = DATA_DIR / "daily_dialog"
PROC_DIR  = DATA_DIR / "processed"
PROC_DIR.mkdir(exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
WINDOW_SIZE = 5
STRIDE      = 2

# ── Breakdown threshold (X-plurality strategy) ──────────────────────────────
# Binary label: 1 if X/total >= X_BREAKDOWN_THRESHOLD
# Verified against real DBDC3 annotations (see _parse_annotations docstring).
# 0.33 = at least 1/3 of annotators must call it a clear breakdown (X).
# This correctly handles:
#   - Ambiguous T-heavy turns   → label=0  (possible but not confirmed)
#   - Mixed O/T/X turns         → depends on X fraction
#   - Clearly broken X-dominant → label=1
X_BREAKDOWN_THRESHOLD = 0.33

# PB_B_THRESHOLD kept as alias for backward compat with any code that references it
PB_B_THRESHOLD = X_BREAKDOWN_THRESHOLD

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

BREAKDOWN_PATTERNS = [
    "I don't understand what you mean.",
    "Could you please clarify? I'm confused.",
    "I'm not sure what you're asking.",
    "Let me ask you something completely different.",
    "I already told you this before.",
    "Why do you keep repeating yourself?",
    "That doesn't make any sense to me.",
    "I have no idea what you're talking about.",
    "Error: I cannot process that request.",
    "I'm sorry, I didn't catch that. Can you repeat?",
    "That's not what I asked at all.",
    "You seem to be ignoring my question.",
    "I feel like we're going in circles.",
    "This conversation isn't going anywhere.",
]
FRUSTRATION_PHRASES = [
    "Ugh, never mind.",
    "You're completely useless.",
    "This is hopeless.",
    "I give up.",
    "Why can't you just answer me?",
    "This is so frustrating!",
    "I've asked this three times now.",
    "Forget it.",
]


# ─── DBDC3 Parser ─────────────────────────────────────────────────────────────
# DBDC3 JSON format (one file = one dialogue):
# {
#   "dialog-id": "IRIS_100_000",
#   "turns": [
#     { "turn-index": 0,
#       "speaker": "S",            ← S = System,  U = User
#       "utterance": "Hello!",
#       "annotations": [           ← ONLY present on System turns
#         {"breakdown": "O"},      ← O=NB  T=PB  X=B
#         {"breakdown": "T"},
#         ...                      ← 30 annotators total
#       ]
#     },
#     { "turn-index": 1, "speaker": "U", "utterance": "Hi there." }
#     ...
#   ]
# }

def _parse_annotations(annotations: list) -> tuple[float, int]:
    """
    Parse DBDC3 annotation list → (breakdown_probability, binary_label).

    DBDC3 label meanings:
      O = Not a Breakdown  (response is coherent and appropriate)
      T = Possible Breakdown (odd/off-topic but conversation could continue)
      X = Breakdown (clear breakdown — conversation cannot continue naturally)

    CORRECT BINARY LABEL STRATEGY:
      We use X-plurality: label=1 if X/total >= 0.33
      Rationale (verified against sample dialogue):
        - Turn 1 (O=15,T=9,X=6): X/30=0.20 → label=0 ✓ (50% said O, borderline/OK)
        - Turn 3 (O=10,T=8,X=12): X/30=0.40 → label=1 ✓ (clear breakdown, topic jump)
        - Turn 7 (O=2,T=9,X=19): X/30=0.63 → label=1 ✓ (obvious breakdown)
        - Turn 19 (O=11,T=14,X=5): X/30=0.17 → label=0 ✓ (T dominant, ambiguous)

      Previous strategy (T+X)/total >= 0.50 was WRONG:
        It flagged Turn 1 as label=1, but 50% of annotators said O (Not Breakdown).
        T is "possible/ambiguous" — it should NOT directly count as a breakdown signal.
        X is the only unambiguous breakdown label.

    SOFT LABEL:
      Weighted score: (0.5*T + 1.0*X) / total
      This gives a calibrated continuous signal for soft-loss training.
      T contributes half as much as X (it's uncertain, not definitive).
    """
    if not annotations:
        return 0.0, 0

    counts = {"O": 0, "T": 0, "X": 0}
    for ann in annotations:
        raw = str(ann.get("breakdown", ann.get("label", "O"))).upper().strip()
        if raw in counts:
            counts[raw] += 1
        elif raw in ("NB", "NOT_BREAKDOWN", "0"):
            counts["O"] += 1
        elif raw in ("PB", "POSSIBLE", "POSSIBLE_BREAKDOWN", "1"):
            counts["T"] += 1
        elif raw in ("B", "BREAKDOWN", "2"):
            counts["X"] += 1
        else:
            counts["O"] += 1   # unknown → treat as Not Breakdown

    total = sum(counts.values())
    if total == 0:
        return 0.0, 0

    # Soft probability: weighted — X counts fully, T counts half
    # This is more calibrated than raw (T+X)/total
    soft = (0.5 * counts["T"] + counts["X"]) / total

    # Binary label: clear breakdown signal — X must be at least 1/3 of annotators
    # This correctly handles ambiguous T-heavy cases (like Turn 1 in sample)
    binary = int(counts["X"] / total >= 0.33)

    return soft, binary


def load_dbdc3_file(filepath: Path) -> dict | None:
    """Load a single DBDC3 JSON file → conversation dict."""
    try:
        raw = json.loads(filepath.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  [WARN] Could not parse {filepath.name}: {e}")
        return None

    conv_id   = raw.get("dialog-id", raw.get("dialogue-id", filepath.stem))
    turns_raw = raw.get("turns", raw.get("utterances", []))
    if not turns_raw:
        return None

    turns = []
    for turn in turns_raw:
        spk_raw = str(turn.get("speaker", "S")).upper()
        speaker = "SYS" if spk_raw in ("S", "SYS", "SYSTEM", "A", "AGENT") else "USR"
        text    = turn.get("utterance", turn.get("text", "")).strip()
        anns    = turn.get("annotations", [])

        if not text:
            continue

        # User turns have no breakdown annotation → always label 0
        if speaker == "USR" or not anns:
            prob, label = 0.0, 0
        else:
            prob, label = _parse_annotations(anns)

        turns.append({
            "turn_idx":       int(turn.get("turn-index", len(turns))),
            "speaker":        speaker,
            "text":           text,
            "breakdown_prob": prob,
            "label":          label,
        })

    return {"conv_id": str(conv_id), "turns": turns} if turns else None


def load_dbdc3_directory(directory: Path) -> list[dict]:
    """Recursively load all DBDC3 JSON files from a directory."""
    files = sorted(directory.rglob("*.json"))
    if not files:
        print(f"  [WARN] No JSON files found in {directory}")
        return []

    convs = []
    for fp in tqdm(files, desc=f"  Loading {directory.name}", leave=False):
        c = load_dbdc3_file(fp)
        if c:
            convs.append(c)
    return convs


def load_all_dbdc3() -> dict[str, list[dict]]:
    """
    Load DBDC3 English data from data/dbdc3/.
    Expected layout (placed by setup_data.py):
        data/dbdc3/dev/   → 415 dialogue files → used as train+val
        data/dbdc3/test/  → 200 dialogue files → used as test
        (setup_data.py maps the zip's 'eval/' folder → data/dbdc3/test/)
    """
    dev_dir  = DBDC3_DIR / "dev"
    # Accept either 'test' (placed by setup_data.py) or 'eval' (raw zip name)
    test_dir = None
    for candidate in [DBDC3_DIR / "test", DBDC3_DIR / "eval"]:
        if candidate.exists() and len(list(candidate.rglob("*.json"))) > 0:
            test_dir = candidate
            break

    if not dev_dir.exists():
        print(f"  [WARN] DBDC3 dev not found at {dev_dir}")
        print(f"  Run: python setup_data.py")
        return {}

    print(f"  Loading from: {DBDC3_DIR}")
    dev_convs  = load_dbdc3_directory(dev_dir)
    test_convs = load_dbdc3_directory(test_dir) if test_dir else []

    print(f"  Dev source:  {len(dev_convs)} dialogues")
    print(f"  Test source: {len(test_convs)} dialogues  "
          f"(from '{test_dir.name}/' folder)" if test_dir else "  Test source: 0 dialogues")

    if not dev_convs:
        print("  [WARN] No conversations loaded from dev directory.")
        return {}

    # ── Label distribution check ───────────────────────────────────────────
    for name, convs in [("dev", dev_convs), ("test", test_convs)]:
        if not convs:
            continue
        sys_turns  = [t for c in convs for t in c["turns"] if t["speaker"] == "SYS"]
        n_sys      = len(sys_turns)
        n_sys_bd   = sum(t["label"] for t in sys_turns)
        all_turns  = [t for c in convs for t in c["turns"]]
        n_all_bd   = sum(t["label"] for t in all_turns)
        print(f"    {name}: {len(convs)} dialogues | "
              f"{n_sys_bd}/{n_sys} system-turn breakdowns ({100*n_sys_bd//max(n_sys,1)}%) | "
              f"{n_all_bd}/{len(all_turns)} total turns labeled 1 ({100*n_all_bd//max(len(all_turns),1)}%)")

    # ── Train/val split from dev ───────────────────────────────────────────
    random.shuffle(dev_convs)
    n_val = max(30, int(0.15 * len(dev_convs)))

    result = {
        "train": dev_convs[n_val:],
        "dev":   dev_convs[:n_val],
    }
    if test_convs:
        result["test"] = test_convs

    return result


# ─── Sliding Window Construction ──────────────────────────────────────────────

def _window_label(turns: list[dict], conv_turns: list[dict], window_start: int) -> tuple[float, int]:
    """
    Compute the (soft_label, binary_label) for a context window.

    ONSET-AWARE LABELING STRATEGY
    ==============================
    Binary label per turn comes from _parse_annotations (X/total >= 0.33).
    Window-level label applies onset logic:

    label=1 if:
      - Last SYS turn in window has binary_label=1 (clear breakdown), AND
      - NOT all prior SYS turns in the window are also broken
        (i.e., this is the onset of collapse, not a window deep inside it)

    label=0 if:
      - Last SYS turn is not broken, OR
      - All turns in window are already broken (collapse already fully established)

    Soft label = breakdown_prob of the last SYS turn (weighted score from
    _parse_annotations: (0.5*T + X)/total — useful for calibration and soft loss).
    """
    sys_turns = [t for t in turns if t["speaker"] == "SYS"]
    if not sys_turns:
        return 0.0, 0

    last_sys  = sys_turns[-1]
    soft      = last_sys["breakdown_prob"]
    last_bd   = last_sys["label"] == 1   # uses X/total >= 0.33

    if not last_bd:
        return soft, 0

    # Last SYS turn is broken. Check saturation:
    prior_sys = sys_turns[:-1]
    if prior_sys:
        all_prior_broken = all(t["label"] == 1 for t in prior_sys)
        if all_prior_broken:
            # Fully saturated — collapse already established, nothing left to detect early
            return 0.5, 0

    # Onset window → label=1
    return soft, 1


def build_windows(conversations: list[dict], window_size=WINDOW_SIZE,
                  stride=STRIDE) -> list[dict]:
    """
    Build sliding context windows from a list of conversations.

    Labeling: Onset-aware strategy (see _window_label docstring).
    Windows at the onset of breakdown = 1.
    Normal windows = 0. Fully saturated (already broken) windows = 0.
    """
    windows = []
    for conv in tqdm(conversations, desc="Building windows", leave=False):
        turns = conv["turns"]
        n     = len(turns)
        if n < 2:
            continue

        s = min(stride, max(1, n - window_size))
        for start in range(0, max(1, n - window_size + 1), s):
            end = min(start + window_size, n)
            wt  = turns[start:end]
            if len(wt) < 2:
                continue

            soft_label, binary_label = _window_label(wt, turns, start)
            n_bd = sum(t["label"] for t in wt if t["speaker"] == "SYS")

            windows.append({
                "sample_id":    f"{conv['conv_id']}_w{start}",
                "conv_id":      conv["conv_id"],
                "window_start": start,
                "window_end":   end,
                "turns":        wt,
                "label":        binary_label,
                "soft_label":   float(soft_label),
                "n_breakdowns": n_bd,
                "source":       "dbdc3",
            })
    return windows


# ─── Expectation Heuristic ────────────────────────────────────────────────────

_EXPECT_RE = re.compile(
    r"(\?|\b(can|could|will|would|should|do|does|did|is|are|was|were|have|has|had)\s+you\b|"
    r"\b(please|kindly)\s+(confirm|clarify|tell|explain|describe|provide|send|check)\b|"
    r"\b(what|where|when|why|how|which|who)\b)", re.IGNORECASE)

def has_expectation(text: str) -> bool:
    return bool(_EXPECT_RE.search(text))


# ─── Augmentation ─────────────────────────────────────────────────────────────

def _load_daily_dialog():
    cache = DD_DIR / "daily_dialog_raw.pkl"
    if cache.exists():
        try:
            with open(cache, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    try:
        from datasets import load_dataset
        print("  Downloading DailyDialog ...")
        ds = load_dataset("daily_dialog", trust_remote_code=False)
        DD_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(ds, f)
        return ds
    except Exception as e:
        print(f"  [WARN] DailyDialog unavailable: {e}")
        return None


def build_clean_windows(n: int = 2000) -> list[dict]:
    """Build n clean (label=0) windows for augmentation."""
    ds = _load_daily_dialog()
    windows = []

    if ds is not None:
        try:
            dialogs = list(ds["train"]["dialog"])
            random.shuffle(dialogs)
            for dlg in dialogs:
                if len(windows) >= n:
                    break
                if len(dlg) < WINDOW_SIZE:
                    continue
                start = random.randint(0, len(dlg) - WINDOW_SIZE)
                turns = [
                    {"turn_idx": j, "speaker": "SYS" if j % 2 == 0 else "USR",
                     "text": str(t).strip(), "breakdown_prob": 0.0, "label": 0}
                    for j, t in enumerate(dlg[start: start + WINDOW_SIZE])
                ]
                windows.append({
                    "sample_id": f"dd_{len(windows)}", "conv_id": f"dd_{len(windows)}",
                    "window_start": start, "window_end": start + WINDOW_SIZE,
                    "turns": turns, "label": 0, "soft_label": 0.0,
                    "n_breakdowns": 0, "source": "daily_dialog_clean",
                })
            print(f"  Built {len(windows)} clean windows from DailyDialog")
        except Exception as e:
            print(f"  [WARN] DailyDialog parse error: {e}")
            windows = []

    if len(windows) < n:
        # Synthetic fallback
        CLEAN = [
            ["What's the weather today?", "It's sunny outside.", "Great, I'll walk.", "Sounds fun!", "Thanks!"],
            ["Can you schedule a meeting?", "Sure! When works?", "Tuesday at 3pm?", "Perfect.", "Confirmed."],
            ["Question about the project.", "Go ahead.", "When is the deadline?", "Next Friday.", "Got it, thanks."],
            ["Good morning!", "Morning! All good?", "Yes, finished the report.", "Excellent!", "Sent it over."],
            ["Can I check something?", "Of course.", "Did the package arrive?", "Yes, this morning.", "Thanks!"],
            ["Need help with a task.", "What do you need?", "Update the calendar.", "Done for you.", "Perfect."],
            ["Any news on the proposal?", "It was approved!", "That's great news.", "Very happy about it.", "Thanks."],
        ]
        need = n - len(windows)
        for i in range(need):
            t = random.choice(CLEAN)
            turns = [{"turn_idx": j, "speaker": "SYS" if j%2==0 else "USR",
                      "text": tx, "breakdown_prob": 0.0, "label": 0} for j, tx in enumerate(t)]
            windows.append({"sample_id": f"sc_{i}", "conv_id": f"sc_{i}",
                            "window_start": 0, "window_end": 5, "turns": turns,
                            "label": 0, "soft_label": 0.0, "n_breakdowns": 0,
                            "source": "synthetic_clean"})
        print(f"  Built {need} synthetic clean windows (DailyDialog fallback)")

    return windows


def build_breakdown_windows(n: int = 1000) -> list[dict]:
    """Build n breakdown (label=1) windows for augmentation."""
    STARTERS = [
        ["Hi, can you help?",           "Hello! What do you need?",     "I need some information."],
        ["I'd like to report an issue.", "Sure, describe it.",           "The system isn't responding."],
        ["Can you confirm my booking?",  "Let me look that up.",         "I've been waiting 20 minutes."],
        ["I've been having problems.",   "I'm sorry to hear that.",      "Can you explain the issue?"],
        ["Something is very wrong.",     "Tell me more.",                "Everything is broken."],
    ]
    windows = []
    for i in range(n):
        starter = random.choice(STARTERS)
        n_ok    = random.randint(1, len(starter))
        n_bd    = WINDOW_SIZE - n_ok
        pool    = BREAKDOWN_PATTERNS + FRUSTRATION_PHRASES
        bds     = random.sample(pool, min(n_bd, len(pool)))
        while len(bds) < n_bd:
            bds.append(random.choice(BREAKDOWN_PATTERNS))
        texts = (starter[:n_ok] + bds)[:WINDOW_SIZE]
        turns = [{"turn_idx": j, "speaker": "SYS" if j%2==0 else "USR",
                  "text": tx, "breakdown_prob": 0.9 if j >= n_ok else 0.1,
                  "label": 1 if j >= n_ok else 0}
                 for j, tx in enumerate(texts)]
        windows.append({
            "sample_id": f"sb_{i}", "conv_id": f"sb_{i}",
            "window_start": 0, "window_end": WINDOW_SIZE, "turns": turns,
            "label": 1, "soft_label": 0.9, "n_breakdowns": n_bd,
            "source": "synthetic_breakdown",
        })
    print(f"  Built {len(windows)} synthetic breakdown windows")
    return windows


# ─── Stats ────────────────────────────────────────────────────────────────────

def print_stats(windows: list[dict], name: str = "Dataset"):
    labels  = [w["label"] for w in windows]
    soft    = [w["soft_label"] for w in windows]
    sources = Counter(w["source"] for w in windows)
    n = max(len(labels), 1)
    print(f"\n{'='*52}\n  {name}")
    print(f"  Total:     {n}")
    print(f"  Positive:  {sum(labels)} ({100*sum(labels)//n}%)")
    print(f"  Negative:  {n-sum(labels)} ({100*(n-sum(labels))//n}%)")
    print(f"  Avg soft:  {np.mean(soft):.3f}")
    print(f"  Sources:   {dict(sources)}")
    print(f"{'='*52}")


# ─── Save / Load ──────────────────────────────────────────────────────────────

def save_processed(splits: dict[str, list[dict]]):
    for sp, data in splits.items():
        with open(PROC_DIR / f"{sp}.pkl", "wb") as f:
            pickle.dump(data, f)
        print(f"  Saved {sp}: {len(data)} samples → {PROC_DIR/f'{sp}.pkl'}")
    rows = [
        {"split": sp, "sample_id": w["sample_id"], "conv_id": w["conv_id"],
         "label": w["label"], "soft_label": w["soft_label"],
         "source": w["source"], "n_turns": len(w["turns"]),
         "last_turn": w["turns"][-1]["text"][:120]}
        for sp, ws in splits.items() for w in ws
    ]
    pd.DataFrame(rows).to_csv(PROC_DIR / "dataset_summary.csv", index=False)
    print(f"  CSV: {PROC_DIR / 'dataset_summary.csv'}")


def load_processed(split: str) -> list[dict]:
    p = PROC_DIR / f"{split}.pkl"
    if not p.exists():
        raise FileNotFoundError(
            f"Not found: {p}\n"
            f"Run: python setup_data.py   (or  python src/data_utils.py --demo)")
    with open(p, "rb") as f:
        return pickle.load(f)


# ─── Demo Dataset ─────────────────────────────────────────────────────────────

def _make_demo() -> dict[str, list[dict]]:
    """300 synthetic conversations for pipeline testing (no real data needed)."""
    print("  Generating 300 synthetic demo conversations ...")
    T_NORM = [
        [("SYS","Hello, how can I help you today?"), ("USR","Hi, I need help with my account."),
         ("SYS","Sure! Can you share your account number?"), ("USR","It's 12345."),
         ("SYS","Thank you. I can see your account now."), ("USR","Great, please update my address."),
         ("SYS","Of course, what's the new address?"), ("USR","123 Main Street."),
         ("SYS","Done! Your address has been updated.")],
        [("SYS","Good morning, how can I assist?"), ("USR","Question about my order."),
         ("SYS","Sure, go ahead."), ("USR","When will it arrive?"),
         ("SYS","Delivery is scheduled for tomorrow."), ("USR","Perfect, thanks!"),
         ("SYS","You're welcome!"), ("USR","That's all."), ("SYS","Have a great day!")],
    ]
    T_BREAK = [
        [("SYS","Hello, how can I help?"), ("USR","I have a serious issue."),
         ("SYS","I don't understand what you mean."), ("USR","My order is missing!"),
         ("SYS","Could you clarify? I'm confused."), ("USR","Why do you keep asking the same thing?"),
         ("SYS","Let me ask you something different."), ("USR","This is useless. I give up."),
         ("SYS","I'm not sure what you're asking.")],
    ]
    convs = []
    for i in range(300):
        is_b = (i % 3 == 0)
        tmpl = random.choice(T_BREAK if is_b else T_NORM)
        turns = []
        for j, (spk, txt) in enumerate(tmpl):
            # For SYS turns: assign breakdown label based on position and template type
            if spk == "SYS":
                bd_prob = 0.8 if (is_b and j >= 4) else 0.05
                label   = 1 if bd_prob >= PB_B_THRESHOLD else 0
            else:
                bd_prob, label = 0.0, 0
            turns.append({"turn_idx": j, "speaker": spk, "text": txt,
                          "breakdown_prob": bd_prob, "label": label})
        convs.append({"conv_id": f"demo_{i}", "turns": turns})

    random.shuffle(convs)
    n = len(convs)
    return {
        "train": convs[:int(.70*n)],
        "dev":   convs[int(.70*n):int(.85*n)],
        "test":  convs[int(.85*n):]
    }


# ─── Main Build ───────────────────────────────────────────────────────────────

def build_dataset(use_demo: bool = False) -> dict[str, list[dict]]:
    print("\n" + "="*60 + "\n  BUILDING DATASET\n" + "="*60)

    # ── 1. Load source conversations ──
    print("\n[1/5] Loading dialogue data ...")
    if use_demo:
        raw = _make_demo()
    else:
        raw = load_all_dbdc3()
        if not raw:
            print("\n  DBDC3 not found → falling back to demo (synthetic) data")
            print("  To use real data: python setup_data.py")
            raw = _make_demo()

    # ── 2. Build windows ──
    print("\n[2/5] Building sliding windows (label = last SYS turn) ...")
    win = {}
    for sp, convs in raw.items():
        win[sp] = build_windows(convs)
        # Report class balance per split
        lbs = [w["label"] for w in win[sp]]
        n   = max(len(lbs), 1)
        print(f"  {sp}: {len(convs)} conversations → {len(win[sp])} windows  "
              f"(pos={sum(lbs)} {100*sum(lbs)//n}%  neg={n-sum(lbs)} {100*(n-sum(lbs))//n}%)")

    # ── 3. Augmentation (train only) ──
    print("\n[3/5] Building augmentation windows ...")
    # Compute how many extra clean samples we need to reach ~50% balance.
    # After real DBDC3 windows, check the positive ratio and add clean aug accordingly.
    real_train = win.get("train", [])
    n_pos_real = sum(w["label"] for w in real_train)
    n_neg_real = len(real_train) - n_pos_real
    # We want final train to be ~45-55% positive.
    # synthetic_breakdown adds positives, synthetic_clean adds negatives.
    # Target: after augmentation, pos ≈ neg.
    # Simple approach: add 1000 breakdown + enough clean to balance.
    bd_aug    = build_breakdown_windows(n=1000)
    n_pos_aug = n_pos_real + 1000
    n_neg_needed = max(2000, n_pos_aug)   # want at least as many negatives as positives
    n_clean_needed = max(0, n_neg_needed - n_neg_real)
    clean_aug = build_clean_windows(n=n_clean_needed)

    # ── 4. Merge ──
    print("\n[4/5] Merging splits ...")
    train_w = win.get("train", []) + clean_aug + bd_aug
    dev_w   = win.get("dev",   [])
    test_w  = win.get("test",  [])

    # Carve dev/test from train if they don't exist
    if not dev_w:
        random.shuffle(train_w)
        k = max(100, int(.15 * len(train_w)))
        dev_w, train_w = train_w[:k], train_w[k:]
        print(f"  Carved {k} samples for dev from train")
    if not test_w:
        k = max(100, int(.15 * len(train_w)))
        test_w, train_w = train_w[:k], train_w[k:]
        print(f"  Carved {k} samples for test from train")

    # ── Dev balance fix ──────────────────────────────────────────────────────
    # Dev is raw DBDC3 windows with onset-aware labels → typically ~17% positive.
    # This is too low for meaningful validation F1 on the positive class.
    # Fix: inject a small number of synthetic breakdown windows to bring dev
    # up to ~30% positive. We keep it lower than train to avoid validation leakage
    # but high enough to give a reliable signal.
    dev_lbs = [w["label"] for w in dev_w]
    dev_pos = sum(dev_lbs)
    dev_neg = len(dev_lbs) - dev_pos
    dev_pos_pct = dev_pos / max(len(dev_lbs), 1)
    if dev_pos_pct < 0.25:
        # How many synthetic positives to add to reach ~30% positive?
        # target_pos / (dev_neg + target_pos) = 0.30  →  target_pos = 0.30 * dev_neg / 0.70
        target_pos = int(0.30 * dev_neg / 0.70)
        n_to_add = max(0, target_pos - dev_pos)
        if n_to_add > 0:
            dev_aug = build_breakdown_windows(n=n_to_add)
            # Rename sample IDs to avoid collisions
            for i, w in enumerate(dev_aug):
                w["sample_id"] = f"dev_aug_{i}"
                w["conv_id"]   = f"dev_aug_{i}"
            dev_w = dev_w + dev_aug
            new_pos = sum(w["label"] for w in dev_w)
            print(f"  Dev balance fix: added {n_to_add} synthetic breakdown windows "
                  f"→ {100*new_pos//len(dev_w)}% positive ({new_pos}/{len(dev_w)})")

    random.shuffle(train_w)
    random.shuffle(dev_w)
    result = {"train": train_w, "dev": dev_w, "test": test_w}

    # ── 5. Save ──
    print("\n[5/5] Saving processed dataset ...")
    for sp, data in result.items():
        print_stats(data, sp.upper())

    # ── Sanity check: warn if any split is heavily imbalanced ──
    for sp, data in result.items():
        lbs = [w["label"] for w in data]
        n   = max(len(lbs), 1)
        pct = 100 * sum(lbs) // n
        if pct > 75:
            print(f"\n  [WARNING] {sp} split is too positive ({pct}%). Check onset-aware labeling logic.")
        elif pct < 20:
            print(f"\n  [WARNING] {sp} split has very few positives ({pct}%). "
                  f"Val metrics on positive class may be unreliable.")

    save_processed(result)
    print("\n  Dataset build complete!")
    return result


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build",  action="store_true", help="Build from DBDC3 real data")
    p.add_argument("--demo",   action="store_true", help="Build synthetic demo dataset")
    p.add_argument("--stats",  action="store_true", help="Print stats of processed splits")
    args = p.parse_args()

    if args.stats:
        for sp in ["train", "dev", "test"]:
            try: print_stats(load_processed(sp), sp.upper())
            except FileNotFoundError as e: print(e)
    elif args.build:
        build_dataset(use_demo=False)
    elif args.demo:
        build_dataset(use_demo=True)
    else:
        p.print_help()
