"""
tokenize_utils.py
=================
Handles all tokenization logic:
  - Special token definitions + registration
  - Serialization of windows to structured input strings
  - Expectation heuristic tagging
  - HuggingFace Dataset creation from processed windows
"""

import re
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# ─── Special Tokens ───────────────────────────────────────────────────────────

SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "[TURN_1]", "[TURN_2]", "[TURN_3]", "[TURN_4]", "[TURN_5]",
        "[TURN_6]", "[TURN_7]", "[TURN_8]", "[TURN_9]", "[TURN_10]",
        "[SPEAKER_SYS]",
        "[SPEAKER_USR]",
        "[EXPECT_RESPONSE]",
        "[NO_EXPECT]",
    ]
}

MAX_LENGTH = 256
MODEL_NAME = "microsoft/deberta-v3-small"

# ─── Expectation Heuristic ────────────────────────────────────────────────────

_EXPECT_RE = re.compile(
    r"(\?|"
    r"\b(can|could|will|would|should|do|does|did|is|are|was|were|have|has|had)\s+you\b|"
    r"\b(please|kindly)\s+(confirm|clarify|tell|explain|describe|provide|send|check|update|verify)\b|"
    r"\b(what|where|when|why|how|which|who)\b|"
    r"\b(need|require|want|expect|waiting\s+for)\b)",
    re.IGNORECASE
)

def has_expectation(text: str) -> bool:
    return bool(_EXPECT_RE.search(text))


# ─── Window Serialization ─────────────────────────────────────────────────────

def serialize_window(turns: list[dict], window_size: int = 5) -> str:
    """
    Convert a list of turn dicts into a single structured input string.
    
    Format per turn:
        [TURN_n] [SPEAKER_X] [EXPECT_FLAG] <utterance text>

    Example:
        [TURN_1] [SPEAKER_SYS] [EXPECT_RESPONSE] Can you confirm your account number?
        [TURN_2] [SPEAKER_USR] [NO_EXPECT] It is 12345.
        [TURN_3] [SPEAKER_SYS] [EXPECT_RESPONSE] Thank you. Is there anything else?
    """
    parts = []
    for i, turn in enumerate(turns[:window_size]):
        turn_token = f"[TURN_{i+1}]"
        speaker_token = f"[SPEAKER_{turn['speaker']}]" if turn['speaker'] in ('SYS', 'USR') else "[SPEAKER_SYS]"
        expect_token = "[EXPECT_RESPONSE]" if has_expectation(turn["text"]) else "[NO_EXPECT]"
        text = turn["text"].strip()
        parts.append(f"{turn_token} {speaker_token} {expect_token} {text}")

    return " ".join(parts)


# ─── Tokenizer Setup ──────────────────────────────────────────────────────────

def load_tokenizer(model_name: str = MODEL_NAME, save_path: Optional[Path] = None) -> AutoTokenizer:
    """
    Load DeBERTa-v3-small tokenizer and register special tokens.

    IMPORTANT: AutoTokenizer.from_pretrained fails for DeBERTa-v3 on newer
    transformers versions because it tries to convert the SentencePiece model
    to a fast tokenizer via tiktoken (which may not be installed). The fix is
    to load DebertaV2Tokenizer directly — this is the correct slow tokenizer
    class for all deberta-v3-* models and bypasses the AutoTokenizer routing
    that triggers the broken fast-tokenizer conversion path.

    use_fast=False alone is NOT sufficient — transformers ignores it for this
    model class in recent versions and still attempts the fast conversion.
    """
    # Import the specific slow tokenizer class directly
    from transformers import DebertaV2Tokenizer

    if save_path and (save_path / "tokenizer_config.json").exists():
        print(f"  Loading tokenizer from cache: {save_path}")
        tokenizer = DebertaV2Tokenizer.from_pretrained(str(save_path))
    else:
        print(f"  Loading tokenizer: {model_name}")
        tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
        n_added = tokenizer.add_special_tokens(SPECIAL_TOKENS)
        print(f"  Added {n_added} special tokens. Vocab size: {len(tokenizer)}")

        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            tokenizer.save_pretrained(str(save_path))
            print(f"  Tokenizer saved to {save_path}")

    return tokenizer


def tokenize_window(
    turns: list[dict],
    tokenizer: AutoTokenizer,
    max_length: int = MAX_LENGTH
) -> dict:
    """
    Serialize a window and tokenize it.
    Returns a dict with input_ids, attention_mask, token_type_ids.
    """
    text = serialize_window(turns)
    encoded = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {k: v.squeeze(0) for k, v in encoded.items()}


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class ConversationWindowDataset(Dataset):
    """
    PyTorch Dataset wrapping processed conversation windows.

    PRE-TOKENIZATION (CPU optimization):
    All samples are tokenized once at construction time and stored in memory.
    With 32GB RAM and ~4200 samples at seq_len=256, this uses ~200MB —
    negligible. Eliminates all tokenization overhead during training loops,
    which on CPU was a significant bottleneck (tokenizer is pure Python).
    """

    def __init__(
        self,
        windows: list[dict],
        tokenizer: AutoTokenizer,
        max_length: int = MAX_LENGTH,
        use_soft_labels: bool = False,
    ):
        self.windows         = windows
        self.tokenizer       = tokenizer
        self.max_length      = max_length
        self.use_soft_labels = use_soft_labels

        # Pre-tokenize everything upfront
        print(f"    Pre-tokenizing {len(windows)} samples...", end=" ", flush=True)
        self._encoded = [
            tokenize_window(w["turns"], tokenizer, max_length)
            for w in windows
        ]
        print("done")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        item      = dict(self._encoded[idx])
        window    = self.windows[idx]
        label_val = window["soft_label"] if self.use_soft_labels else float(window["label"])
        item["labels"]    = torch.tensor(label_val, dtype=torch.float)
        item["sample_id"] = window["sample_id"]
        return item

    def get_labels(self) -> list[int]:
        return [w["label"] for w in self.windows]


def build_dataloaders(
    splits: dict,
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
    max_length: int = MAX_LENGTH,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """
    Build DataLoaders for all splits.

    num_workers: parallel prefetch workers. On Windows, must be 0 inside
                 scripts not guarded by if __name__ == '__main__'.
                 train.py's CLI is guarded, so 4 works fine there.
                 Set to 0 if you see DataLoader worker errors.

    pin_memory: speeds up CPU→GPU transfer. No-op on pure CPU but harmless.
                Since we pre-tokenize, the main benefit here is batch assembly.
    """
    from torch.utils.data import DataLoader

    # With pre-tokenization, num_workers for I/O prefetch is less critical,
    # but still helps batch collation on large datasets.
    # On CPU-only machines, set num_workers=0 to avoid multiprocessing overhead.
    effective_workers = num_workers if torch.cuda.is_available() else 0
    if num_workers > 0 and not torch.cuda.is_available():
        print(f"    Note: num_workers reduced to 0 for CPU training "
              f"(pre-tokenization makes workers unnecessary)")

    loaders = {}
    for split, windows in splits.items():
        dataset = ConversationWindowDataset(windows, tokenizer, max_length)
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=effective_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            # Persistent workers keep processes alive between epochs (GPU only)
            persistent_workers=(effective_workers > 0),
        )
        print(f"  {split}: {len(dataset)} samples → {len(loaders[split])} batches "
              f"(bs={batch_size})")

    return loaders


# ─── Quick Sanity Check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_turns = [
        {"speaker": "SYS", "text": "Hello, how can I help you today?"},
        {"speaker": "USR", "text": "I have a problem with my order."},
        {"speaker": "SYS", "text": "Could you please describe the issue?"},
        {"speaker": "USR", "text": "It never arrived. I'm very frustrated."},
        {"speaker": "SYS", "text": "I'm sorry, I don't understand what you mean."},
    ]

    print("=== Serialized Window ===")
    print(serialize_window(sample_turns))
    print()

    print("=== Expectation Tags ===")
    for t in sample_turns:
        print(f"  [{has_expectation(t['text'])}] {t['text']}")
