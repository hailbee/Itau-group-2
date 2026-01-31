# siamese.py
from __future__ import annotations

from typing import Optional, Literal

import torch
import torch.nn as nn


ActName = Literal["relu", "gelu", "tanh"]


def _make_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    *,
    activation: ActName = "relu",
    dropout: float = 0.0,
) -> nn.Sequential:
    if activation == "relu":
        act = nn.ReLU()
    elif activation == "gelu":
        act = nn.GELU()
    elif activation == "tanh":
        act = nn.Tanh()
    else:
        raise ValueError(f"Unknown activation: {activation}")

    layers = [nn.Linear(in_dim, hidden_dim), act]
    if dropout and dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class SiameseEmbeddingModel(nn.Module):
    """
    Flexible Siamese model for your new setup.

    Supports:
      - separate fraud-text and real-text projection heads (or shared)
      - projecting image/teacher embeddings to out_dim (or identity if teacher_dim=None)

    Typical usage patterns:

    1) FINAL FINAL evaluation target (text->image-like space):
        z_fraud = normalize(encode_fraud_text(fraud_txt))
        z_real  = normalize(encode_real_text(real_txt))
        score = cosine(z_fraud, z_real)

    2) Text->Image (teacher) alignment:
        z_txt = normalize(encode_text(...))
        z_img = normalize(encode_teacher(image_emb))
        score = cosine(z_txt, z_img)

    NOTE: This module does NOT normalize outputs; do that in trainer/evaluator.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        out_dim: int,
        teacher_dim: Optional[int] = None,
        *,
        share_text_heads: bool = False,
        share_teacher_heads: bool = True,
        activation: ActName = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.teacher_dim = None if teacher_dim is None else int(teacher_dim)
        self.share_text_heads = bool(share_text_heads)
        self.share_teacher_heads = bool(share_teacher_heads)

        # ---- text heads ----
        if self.share_text_heads:
            self.text_head = _make_mlp(
                self.embedding_dim,
                self.hidden_dim,
                self.out_dim,
                activation=activation,
                dropout=dropout,
            )
            self.fraud_text_head = self.text_head
            self.real_text_head = self.text_head
        else:
            self.fraud_text_head = _make_mlp(
                self.embedding_dim,
                self.hidden_dim,
                self.out_dim,
                activation=activation,
                dropout=dropout,
            )
            self.real_text_head = _make_mlp(
                self.embedding_dim,
                self.hidden_dim,
                self.out_dim,
                activation=activation,
                dropout=dropout,
            )
            self.text_head = None  # not used when separate

        # ---- teacher/image heads ----
        # If teacher_dim is None, we assume teacher embeddings are already in out_dim
        # and do an identity mapping.
        if self.teacher_dim is None:
            self.teacher_proj = None
            self.fraud_teacher_proj = None
            self.real_teacher_proj = None
        else:
            if self.share_teacher_heads:
                self.teacher_proj = _make_mlp(
                    self.teacher_dim,
                    self.hidden_dim,
                    self.out_dim,
                    activation=activation,
                    dropout=dropout,
                )
                self.fraud_teacher_proj = self.teacher_proj
                self.real_teacher_proj = self.teacher_proj
            else:
                self.teacher_proj = None  # not used when separate
                self.fraud_teacher_proj = _make_mlp(
                    self.teacher_dim,
                    self.hidden_dim,
                    self.out_dim,
                    activation=activation,
                    dropout=dropout,
                )
                self.real_teacher_proj = _make_mlp(
                    self.teacher_dim,
                    self.hidden_dim,
                    self.out_dim,
                    activation=activation,
                    dropout=dropout,
                )

    # -------------------------
    # Text encoders
    # -------------------------
    def encode_fraud_text(self, x: torch.Tensor) -> torch.Tensor:
        return self.fraud_text_head(x)

    def encode_real_text(self, x: torch.Tensor) -> torch.Tensor:
        return self.real_text_head(x)

    def encode_text(
        self,
        x: torch.Tensor,
        *,
        side: Literal["fraud", "real"] = "fraud",
    ) -> torch.Tensor:
        if side == "fraud":
            return self.encode_fraud_text(x)
        elif side == "real":
            return self.encode_real_text(x)
        else:
            raise ValueError(f"side must be 'fraud' or 'real', got {side!r}")

    # -------------------------
    # Teacher / image encoders
    # -------------------------
    def encode_fraud_teacher(self, t: torch.Tensor) -> torch.Tensor:
        if self.teacher_dim is None:
            return t
        return self.fraud_teacher_proj(t)

    def encode_real_teacher(self, t: torch.Tensor) -> torch.Tensor:
        if self.teacher_dim is None:
            return t
        return self.real_teacher_proj(t)

    def encode_teacher(
        self,
        t: torch.Tensor,
        *,
        side: Optional[Literal["fraud", "real"]] = None,
    ) -> torch.Tensor:
        """
        Backwards-compatible default:
          - if side is None and we have a shared teacher head, use it
          - otherwise require side to pick fraud vs real teacher projection
        """
        if self.teacher_dim is None:
            return t

        if self.share_teacher_heads:
            return self.fraud_teacher_proj(t)  # same as real_teacher_proj
        if side is None:
            raise ValueError(
                "encode_teacher(side=None) is ambiguous because share_teacher_heads=False. "
                "Call encode_teacher(t, side='fraud'|'real')."
            )
        if side == "fraud":
            return self.encode_fraud_teacher(t)
        elif side == "real":
            return self.encode_real_teacher(t)
        else:
            raise ValueError(f"side must be 'fraud' or 'real', got {side!r}")

    # Alias if you prefer "image" naming
    def encode_image(
        self,
        img: torch.Tensor,
        *,
        side: Optional[Literal["fraud", "real"]] = None,
    ) -> torch.Tensor:
        return self.encode_teacher(img, side=side)

    # -------------------------
    # Forward (for text-text scoring/evaluation)
    # -------------------------
    def forward(self, fraud_txt: torch.Tensor, real_txt: torch.Tensor):
        """
        Returns:
          z_fraud_txt, z_real_txt  (NOT normalized)

        This matches your Evaluator2 pattern:
          z_f, z_r = model(fraud_txt, real_txt)
        """
        z_f = self.encode_fraud_text(fraud_txt)
        z_r = self.encode_real_text(real_txt)
        return z_f, z_r
