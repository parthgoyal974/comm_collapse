"""
setup_data.py
=============
One-click data setup. Run this ONCE before anything else.

Usage:
    python setup_data.py                  # full setup (recommended)
    python setup_data.py --demo           # synthetic data only, no downloads
    python setup_data.py --skip-augment   # skip DailyDialog
    python setup_data.py --check          # show what is already present
"""

import sys
import shutil
import zipfile
import argparse
import pickle
import urllib.request
from pathlib import Path

ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "data"
DBDC3_DIR = DATA_DIR / "dbdc3"
DD_DIR    = DATA_DIR / "daily_dialog"
PROC_DIR  = DATA_DIR / "processed"

DBDC3_URL = "https://dbd-challenge.github.io/dbdc3/data/DBDC3.zip"
DBDC3_ZIP = DATA_DIR / "DBDC3.zip"


# ─── Helpers ──────────────────────────────────────────────────────────────────

def header(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")
def ok(m):     print(f"  ✓  {m}")
def warn(m):   print(f"  ⚠  {m}")
def info(m):   print(f"  →  {m}")
def fail(m):   print(f"  ✗  {m}")


def _count_json(d: Path) -> int:
    return len(list(d.rglob("*.json"))) if d.exists() else 0


def download_with_progress(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _hook(count, block, total):
        if total > 0:
            pct  = min(100, count * block * 100 // total)
            bar  = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r  [{bar}] {pct:3d}%  ({total/1_048_576:.1f} MB)", end="", flush=True)

    print(f"  URL: {url}")
    urllib.request.urlretrieve(url, dest, reporthook=_hook)
    print()
    ok(f"Downloaded → {dest}")


# ─── DBDC3 Download + Extract ─────────────────────────────────────────────────

def _dbdc3_ready() -> bool:
    """True if dev folder already has enough JSON files."""
    return _count_json(DBDC3_DIR / "dev") >= 50


def download_dbdc3() -> bool:
    header("STEP 1 / 4 — Downloading DBDC3 English Dataset")
    if _dbdc3_ready():
        ok(f"DBDC3 already present ({_count_json(DBDC3_DIR / 'dev')} dev files) — skipping")
        return True

    if not DBDC3_ZIP.exists():
        info("Downloading DBDC3.zip (~4 MB) ...")
        try:
            download_with_progress(DBDC3_URL, DBDC3_ZIP)
        except Exception as e:
            fail(f"Download failed: {e}")
            print()
            print("  Manual instructions:")
            print(f"    1. Open in browser: {DBDC3_URL}")
            print(f"    2. Save to:         {DBDC3_ZIP}")
            print(f"    3. Re-run:          python setup_data.py")
            return False
    else:
        ok(f"DBDC3.zip already on disk: {DBDC3_ZIP}")

    return _extract_dbdc3()


def _extract_dbdc3() -> bool:
    header("STEP 2 / 4 — Extracting & Organising DBDC3")

    tmp = DATA_DIR / "_dbdc3_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    info("Extracting zip ...")
    try:
        with zipfile.ZipFile(DBDC3_ZIP, "r") as zf:
            zf.extractall(tmp)
        ok("Extraction complete")
    except zipfile.BadZipFile as e:
        fail(f"Corrupt zip: {e}  — deleting it so next run re-downloads")
        DBDC3_ZIP.unlink(missing_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
        return False

    # ── Debug: show what folders are in the zip ────────────────────────────
    info("Scanning zip contents ...")
    for p in sorted(tmp.rglob("*"))[:40]:
        if p.is_dir():
            print(f"    [DIR]  {p.relative_to(tmp)}")

    # ── Find matching dev + eval/test pair ─────────────────────────────────
    # DBDC3.zip actual structure:
    #   DBDC3/dbdc3_revised/en/dev/    ← training/dev data (415 files)
    #   DBDC3/dbdc3_revised/en/eval/   ← evaluation/test data (200 files)
    #   DBDC3/dbdc3/en/dev/
    #   DBDC3/dbdc3/en/eval/
    #
    # NOTE: The folder is called 'eval', NOT 'test'. This was the root cause
    # of the 0 test files bug. We check for both names.
    # We prefer dbdc3_revised subtree over dbdc3.

    dev_dir  = None
    eval_dir = None   # This will become our test split

    # Sort candidates: prefer 'revised', then deepest path (most specific)
    all_dev_candidates = sorted(
        tmp.rglob("dev"),
        key=lambda p: ("revised" not in str(p), -len(p.parts))
    )

    for d in all_dev_candidates:
        if not d.is_dir():
            continue
        # Only accept English data
        path_str = str(d).lower()
        if "/jp/" in path_str or "\\jp\\" in path_str:
            continue
        if _count_json(d) < 50:
            continue

        # Look for eval or test sibling in the same parent
        for eval_name in ("eval", "test"):
            candidate_eval = d.parent / eval_name
            if candidate_eval.exists() and _count_json(candidate_eval) >= 10:
                dev_dir  = d
                eval_dir = candidate_eval
                info(f"Found dev:  {d.relative_to(tmp)}  ({_count_json(d)} files)")
                info(f"Found eval: {candidate_eval.relative_to(tmp)}  ({_count_json(candidate_eval)} files)")
                break
        if dev_dir:
            break

    # Fallback: any dev with enough files (no matching eval)
    if dev_dir is None:
        for d in all_dev_candidates:
            if d.is_dir() and _count_json(d) >= 50:
                dev_dir = d
                warn("No matching eval/test folder found — will carve test from dev")
                info(f"Using dev: {d.relative_to(tmp)}  ({_count_json(d)} files)")
                break

    if dev_dir is None:
        fail("Could not find a valid dev/ folder inside the zip.")
        info(f"Inspect the contents manually: {tmp}")
        return False

    # ── Copy into data/dbdc3/ ──────────────────────────────────────────────
    DBDC3_DIR.mkdir(parents=True, exist_ok=True)
    _safe_copy(dev_dir,  DBDC3_DIR / "dev")
    if eval_dir and _count_json(eval_dir) > 0:
        _safe_copy(eval_dir, DBDC3_DIR / "test")  # we always call it 'test' internally

    # ── Cleanup ───────────────────────────────────────────────────────────
    shutil.rmtree(tmp, ignore_errors=True)
    DBDC3_ZIP.unlink(missing_ok=True)
    ok("Temporary files cleaned up")

    n_dev  = _count_json(DBDC3_DIR / "dev")
    n_test = _count_json(DBDC3_DIR / "test")
    ok(f"data/dbdc3/dev/  — {n_dev} dialogue files")
    ok(f"data/dbdc3/test/ — {n_test} dialogue files")

    if n_dev < 100:
        warn(f"Expected ~415 dev files, got {n_dev}. Data may be incomplete.")
    if n_test == 0:
        warn("No test files — test split will be carved from dev (15%).")

    return True


def _safe_copy(src: Path, dst: Path):
    """Copy src directory tree to dst, merging if dst exists."""
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    ok(f"Placed: {dst}  ({_count_json(dst)} files)")


# ─── DailyDialog ──────────────────────────────────────────────────────────────

def download_daily_dialog() -> bool:
    header("STEP 3 / 4 — DailyDialog Augmentation Data")
    cache = DD_DIR / "daily_dialog_raw.pkl"
    if cache.exists():
        ok("DailyDialog already cached — skipping")
        return True

    info("Trying HuggingFace datasets library ...")
    try:
        from datasets import load_dataset
        # Try parquet-based version (no loading script needed)
        ds = load_dataset("daily_dialog", trust_remote_code=False)
        DD_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(ds, f)
        ok(f"DailyDialog cached ({len(ds['train'])} train dialogues)")
        return True
    except Exception as e:
        warn(f"DailyDialog unavailable: {e}")
        info("Synthetic clean samples will be used instead — training is unaffected.")
        return False


# ─── Build Processed Dataset ──────────────────────────────────────────────────

def build_processed_dataset(use_demo: bool = False) -> dict | None:
    header("STEP 4 / 4 — Building Processed Dataset")
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from data_utils import build_dataset
        return build_dataset(use_demo=use_demo)
    except Exception as e:
        fail(f"Dataset build failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ─── Check Status ─────────────────────────────────────────────────────────────

def check_status():
    header("DATA STATUS CHECK")
    n_dev  = _count_json(DBDC3_DIR / "dev")
    n_test = _count_json(DBDC3_DIR / "test")

    (ok if n_dev >= 100 else warn)(f"DBDC3 dev:  {n_dev} files  (need ~415)")
    (ok if n_test >= 50  else warn)(f"DBDC3 test: {n_test} files  (need ~200)")

    dd = DD_DIR / "daily_dialog_raw.pkl"
    (ok if dd.exists() else warn)(f"DailyDialog: {'cached' if dd.exists() else 'not cached (synthetic fallback used)'}")

    for sp in ["train", "dev", "test"]:
        p = PROC_DIR / f"{sp}.pkl"
        if p.exists():
            with open(p, "rb") as f:
                d = pickle.load(f)
            lbs = [w["label"] for w in d]
            ok(f"Processed {sp}: {len(d)} samples  "
               f"pos={sum(lbs)} ({100*sum(lbs)//max(len(lbs),1)}%)  "
               f"neg={len(lbs)-sum(lbs)} ({100*(len(lbs)-sum(lbs))//max(len(lbs),1)}%)")
        else:
            warn(f"Processed {sp}: not built  → run: python setup_data.py")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo",         action="store_true")
    parser.add_argument("--skip-augment", action="store_true")
    parser.add_argument("--check",        action="store_true")
    args = parser.parse_args()

    print("\n" + "█" * 60)
    print("  COMMUNICATION COLLAPSE — DATA SETUP")
    print("  Parth Goyal | 23BCE0411")
    print("█" * 60)

    if args.check:
        check_status()
        return

    if args.demo:
        info("Demo mode — synthetic data only")
        splits = build_processed_dataset(use_demo=True)
    else:
        ok_dbdc3  = download_dbdc3()
        if not ok_dbdc3:
            warn("DBDC3 download failed — falling back to demo (synthetic) data")
        if not args.skip_augment:
            download_daily_dialog()
        splits = build_processed_dataset(use_demo=False)

    if splits:
        header("SETUP COMPLETE — READY TO TRAIN")
        for sp, data in splits.items():
            lbs = [w["label"] for w in data]
            n   = max(len(lbs), 1)
            print(f"  {sp:<8} {len(data):>5} samples   "
                  f"pos={sum(lbs):>4} ({100*sum(lbs)//n:>2}%)   "
                  f"neg={n-sum(lbs):>4} ({100*(n-sum(lbs))//n:>2}%)")
        print()
        print("  Next step:  python src/train.py")
        print()
    else:
        header("SETUP INCOMPLETE")
        print("  Re-run with: python setup_data.py --demo")


if __name__ == "__main__":
    main()
