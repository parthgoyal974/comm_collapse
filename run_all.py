"""
run_all.py
==========
Master script to reproduce all results end-to-end.

Usage:
    python run_all.py                  # Full pipeline (~2-3 hrs on CPU)
    python run_all.py --fast           # Smoke test: 2 epochs, skip baselines (~30 min)
    python run_all.py --skip-baselines # Skip TF-IDF and BERT baselines
    python run_all.py --demo           # Use synthetic data (no DBDC3 needed)
"""

import subprocess
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent


def run(cmd, desc):
    print(f"\n{'='*60}")
    print(f"  STEP: {desc}")
    print(f"  CMD:  {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True, cwd=ROOT)
    if result.returncode != 0:
        print(f"\n  [ERROR] Step failed: {desc}")
        sys.exit(1)
    print(f"  [DONE] {desc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",            action="store_true",
                        help="Use synthetic data (no DBDC3 download)")
    parser.add_argument("--skip-baselines",  action="store_true",
                        help="Skip TF-IDF and BERT baselines")
    parser.add_argument("--fast",            action="store_true",
                        help="Smoke test: 2 epochs, skip baselines")
    args = parser.parse_args()

    if args.fast:
        args.skip_baselines = True

    main_epochs = 1 if args.fast else 4
    run_name    = "smoke" if args.fast else "main"
    ckpt_path   = f"results/checkpoints/{run_name}/best_model"
    eval_name   = f"{run_name}_eval"

    # ── Step 1: Build dataset ──────────────────────────────────────────────────
    if args.demo:
        run("python src/data_utils.py --demo", "Build synthetic demo dataset")
    else:
        run("python setup_data.py", "Download DBDC3 + build dataset")

    # ── Step 2: TF-IDF baseline ────────────────────────────────────────────────
    if not args.skip_baselines:
        run("python baselines/tfidf_lr.py", "TF-IDF + Logistic Regression baseline")

    # ── Step 3: BERT/RoBERTa baselines ────────────────────────────────────────
    if not args.skip_baselines:
        bert_epochs = min(main_epochs, 3)
        run(f"python baselines/bert_baseline.py --model bert-base-uncased --epochs {bert_epochs}",
            "BERT-base baseline")
        run(f"python baselines/bert_baseline.py --model roberta-base --epochs {bert_epochs}",
            "RoBERTa-base baseline")
    else:
        print("\n  [SKIP] Baselines (--skip-baselines or --fast)")

    # ── Step 4: Train main model ───────────────────────────────────────────────
    compile_flag = "--no-compile" if args.fast else ""
    run(
        f"python src/train.py --epochs {main_epochs} --run-name {run_name} {compile_flag}".strip(),
        f"Train DeBERTa-v3-small ({main_epochs} epoch{'s' if main_epochs > 1 else ''})"
    )

    # ── Step 5: Evaluate on test set ──────────────────────────────────────────
    run(
        f"python src/evaluate.py --checkpoint {ckpt_path} --run-name {eval_name}",
        "Evaluate on test set"
    )

    # ── Step 6: Generate all figures ──────────────────────────────────────────
    run(
        f"python src/visualize.py --checkpoint {ckpt_path} --run-name {eval_name}",
        "Generate all figures (incl. published baselines comparison)"
    )

    print("\n" + "="*60)
    print("  ALL STEPS COMPLETE!")
    print(f"  Figures:  {ROOT / 'results' / 'figures'}")
    print(f"  Metrics:  {ROOT / 'results' / 'metrics'}")
    print(f"  Model:    {ROOT / ckpt_path}")
    print("="*60)


if __name__ == "__main__":
    main()
