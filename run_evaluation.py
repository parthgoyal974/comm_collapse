"""
run_evaluation.py
=================

Runs evaluation and visualization for both:
1) best_model checkpoint
2) last_model checkpoint

Outputs:
results/metrics/
results/figures/
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

BEST_CKPT = "results/checkpoints/main/best_model"
LAST_CKPT = "results/checkpoints/main/last_model"


def run(cmd, desc):
    print("\n" + "=" * 60)
    print(f"STEP: {desc}")
    print(f"CMD : {cmd}")
    print("=" * 60)

    result = subprocess.run(cmd, shell=True, cwd=ROOT)

    if result.returncode != 0:
        print(f"\n[ERROR] Step failed: {desc}")
        sys.exit(1)

    print(f"[DONE] {desc}")


def main():

    # ── Evaluate BEST model ─────────────────────────────
    run(
        f"python src/evaluate.py --checkpoint {BEST_CKPT} --run-name best_eval",
        "Evaluate BEST checkpoint"
    )

    # ── Visualize BEST model ────────────────────────────
    run(
        f"python src/visualize.py --checkpoint {BEST_CKPT} --run-name best_eval",
        "Generate figures for BEST checkpoint"
    )

    # ── Evaluate LAST model ─────────────────────────────
    run(
        f"python src/evaluate.py --checkpoint {LAST_CKPT} --run-name last_eval",
        "Evaluate LAST checkpoint"
    )

    # ── Visualize LAST model ────────────────────────────
    run(
        f"python src/visualize.py --checkpoint {LAST_CKPT} --run-name last_eval",
        "Generate figures for LAST checkpoint"
    )

    print("\n" + "=" * 60)
    print("ALL EVALUATIONS COMPLETE")
    print(f"Metrics folder : {ROOT / 'results' / 'metrics'}")
    print(f"Figures folder : {ROOT / 'results' / 'figures'}")
    print("=" * 60)


if __name__ == "__main__":
    main()