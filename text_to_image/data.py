from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


LabelCol = Union[int, str]


def _coerce_numeric_label(series: pd.Series, *, name: str = "label") -> np.ndarray:
    """Convert a label Series to float32 numeric array.

    Accepts ints/floats/bools/"0"/"1"/"0.0"/"1.0".
    Raises a clear error if non-numeric values exist.
    """
    s = series
    if s.dtype == bool:
        s = s.astype(np.int64)

    if s.dtype == object:
        s2 = pd.to_numeric(s, errors="coerce")
        if s2.isna().any():
            bad = s[s2.isna()].unique()[:10]
            raise TypeError(
                f"{name} column contains non-numeric values; examples: {bad}. "
                f"Fix the source parquet/csv so label is 0/1."
            )
        s = s2

    arr = s.to_numpy()
    return arr.astype(np.float32, copy=False)


def _get_label_series(df: pd.DataFrame, label_col: LabelCol) -> pd.Series:
    """label_col can be an int (iloc) or str (named column)."""
    if isinstance(label_col, int):
        return df.iloc[:, int(label_col)]
    if isinstance(label_col, str):
        if label_col not in df.columns:
            raise KeyError(f"label_col='{label_col}' not found in df.columns")
        return df[label_col]
    raise TypeError(f"label_col must be int or str, got {type(label_col)}")


def _has_prefix(df: pd.DataFrame, prefix: str) -> bool:
    return any(isinstance(c, str) and c.startswith(prefix) for c in df.columns)


def _sorted_prefixed_cols(df: pd.DataFrame, prefix: str) -> List[str]:
    """Return columns starting with `prefix`, sorted by integer suffix where possible."""
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if not cols:
        raise KeyError(f"No columns found with prefix '{prefix}'")

    def key(c: str):
        suf = c[len(prefix):]
        try:
            return (0, int(suf))
        except Exception:
            return (1, suf)

    return sorted(cols, key=key)


def _prefix_to_numpy(df: pd.DataFrame, prefix: str, *, name: str) -> np.ndarray:
    cols = _sorted_prefixed_cols(df, prefix)
    mat = df[cols].to_numpy(dtype=np.float32, copy=False)
    if mat.ndim != 2:
        raise TypeError(f"{name} prefix '{prefix}' did not produce a 2D matrix.")
    return mat


@dataclass(frozen=True)
class PairPrefixes:
    left: str
    right: str


class EmbeddingPairDataset(Dataset):
    """Prefix-based pair dataset for Siamese training/export: returns (x1, x2, y).

    Expected columns:
      - label
      - {x1_prefix}0..{x1_prefix}D-1
      - {x2_prefix}0..{x2_prefix}D-1

    Defaults match your exported golden parquet:
      - fraud_raw_*
      - real_raw_*
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        x1_prefix: str = "fraud_raw_",
        x2_prefix: str = "real_raw_",
        label_col: LabelCol = "label",
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)

        if not _has_prefix(self.df, x1_prefix):
            raise KeyError(f"EmbeddingPairDataset: missing columns with prefix '{x1_prefix}'")
        if not _has_prefix(self.df, x2_prefix):
            raise KeyError(f"EmbeddingPairDataset: missing columns with prefix '{x2_prefix}'")

        self.x1 = _prefix_to_numpy(self.df, x1_prefix, name="x1")
        self.x2 = _prefix_to_numpy(self.df, x2_prefix, name="x2")

        y_series = _get_label_series(self.df, label_col)
        self.y = _coerce_numeric_label(y_series, name="label")

        if len(self.x1) != len(self.x2) or len(self.x1) != len(self.y):
            raise ValueError("EmbeddingPairDataset: length mismatch among x1/x2/y.")
        if self.x1.shape[1] != self.x2.shape[1]:
            raise ValueError(
                f"EmbeddingPairDataset: dim mismatch x1_dim={self.x1.shape[1]} vs x2_dim={self.x2.shape[1]}"
            )

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int):
        x1 = torch.from_numpy(self.x1[idx])
        x2 = torch.from_numpy(self.x2[idx])
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x1, x2, y


class Text2TeacherDistillDataset(Dataset):
    """Prefix-based distillation dataset.

    Returns:
      (fraud_txt, real_txt, fraud_teacher, real_teacher, label)

    Intended for your "Text -> Golden teacher" task.

    Expected columns:
      - label
      - fraud_txt_* , real_txt_*
      - fraud_aligned_* , real_aligned_*   (teacher by default)

    You may override prefixes if you used different names.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        fraud_txt_prefix: str = "fraud_txt_",
        real_txt_prefix: str = "real_txt_",
        fraud_teacher_prefix: str = "fraud_aligned_",
        real_teacher_prefix: str = "real_aligned_",
        label_col: LabelCol = "label",
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)

        for pfx, nm in [
            (fraud_txt_prefix, "fraud_txt"),
            (real_txt_prefix, "real_txt"),
            (fraud_teacher_prefix, "fraud_teacher"),
            (real_teacher_prefix, "real_teacher"),
        ]:
            if not _has_prefix(self.df, pfx):
                raise KeyError(f"Text2TeacherDistillDataset: missing {nm} columns with prefix '{pfx}'")

        self.fraud_txt = _prefix_to_numpy(self.df, fraud_txt_prefix, name="fraud_txt")
        self.real_txt = _prefix_to_numpy(self.df, real_txt_prefix, name="real_txt")
        self.fraud_teacher = _prefix_to_numpy(self.df, fraud_teacher_prefix, name="fraud_teacher")
        self.real_teacher = _prefix_to_numpy(self.df, real_teacher_prefix, name="real_teacher")

        y_series = _get_label_series(self.df, label_col)
        self.labels = _coerce_numeric_label(y_series, name="label")

        n = len(self.labels)
        for name, arr in [
            ("fraud_txt", self.fraud_txt),
            ("real_txt", self.real_txt),
            ("fraud_teacher", self.fraud_teacher),
            ("real_teacher", self.real_teacher),
        ]:
            if len(arr) != n:
                raise ValueError(f"Text2TeacherDistillDataset: length mismatch for {name} vs labels.")

        if self.fraud_txt.shape[1] != self.real_txt.shape[1]:
            raise ValueError(
                f"Text2TeacherDistillDataset: txt dim mismatch fraud_txt_dim={self.fraud_txt.shape[1]} "
                f"vs real_txt_dim={self.real_txt.shape[1]}"
            )
        if self.fraud_teacher.shape[1] != self.real_teacher.shape[1]:
            raise ValueError(
                f"Text2TeacherDistillDataset: teacher dim mismatch fraud_teacher_dim={self.fraud_teacher.shape[1]} "
                f"vs real_teacher_dim={self.real_teacher.shape[1]}"
            )

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, idx: int):
        fraud_txt = torch.from_numpy(self.fraud_txt[idx])
        real_txt = torch.from_numpy(self.real_txt[idx])
        fraud_teacher = torch.from_numpy(self.fraud_teacher[idx])
        real_teacher = torch.from_numpy(self.real_teacher[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return fraud_txt, real_txt, fraud_teacher, real_teacher, y
