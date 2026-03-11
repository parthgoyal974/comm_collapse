"""
model.py
========
CommunicationRiskModel: DeBERTa-v3-small + Risk Estimation Head
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from pathlib import Path


class CommunicationRiskModel(nn.Module):

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-small",
        vocab_size: int = None,
        dropout: float = 0.1,
        freeze_layers: int = 0,
    ):
        super().__init__()

        self.model_name = model_name

        config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=config)

        if vocab_size is not None and vocab_size != config.vocab_size:
            old_vocab = config.vocab_size
            self.encoder.resize_token_embeddings(vocab_size)
            print(f"  Resized embeddings: {old_vocab} → {vocab_size}")

        hidden_size = config.hidden_size

        self.risk_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        return_cls: bool = False,
    ):

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)

        cls_emb = outputs.last_hidden_state[:, 0, :]

        logits = self.risk_head(cls_emb)

        risk = torch.sigmoid(logits)

        result = {
            "logits": logits,
            "risk": risk,
        }

        if return_cls:
            result["cls_emb"] = cls_emb

        return result

    def _freeze_layers(self, n: int):

        layers = None

        if hasattr(self.encoder, "encoder") and hasattr(self.encoder.encoder, "layer"):
            layers = self.encoder.encoder.layer

        if layers is not None:
            for i, layer in enumerate(layers):
                if i < n:
                    for param in layer.parameters():
                        param.requires_grad = False

            print(f"  Froze bottom {n} transformer layers")

    def count_parameters(self):

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }


class FocalLoss(nn.Module):

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()

        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        probs = torch.sigmoid(logits)

        p_t = probs * targets + (1 - probs) * (1 - targets)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()

        if self.reduction == "sum":
            return loss.sum()

        return loss


def get_loss_fn(loss_type: str = "bce", pos_weight: float = 2.0):

    if loss_type == "bce":

        pw = torch.tensor([pos_weight], dtype=torch.float32)

        return nn.BCEWithLogitsLoss(pos_weight=pw)

    elif loss_type == "focal":

        return FocalLoss(gamma=2.0, alpha=0.25)

    else:

        return nn.BCEWithLogitsLoss()


def save_model(model: CommunicationRiskModel, path: Path, metadata: dict = None):

    path = Path(path)

    path.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model.model_name,
            "metadata": metadata or {},
        },
        path / "model.pt",
    )

    print(f"  Model saved to {path / 'model.pt'}")


def load_model(
    path: Path,
    vocab_size: int = None,
    device: str = "cpu",
) -> CommunicationRiskModel:

    path = Path(path)

    ckpt = torch.load(path / "model.pt", map_location=device)

    model = CommunicationRiskModel(
        model_name=ckpt.get("model_name", "microsoft/deberta-v3-small"),
        vocab_size=vocab_size,
    )

    model.load_state_dict(ckpt["model_state_dict"])

    model.eval()

    print(f"  Loaded model from {path / 'model.pt'}")

    return model

