from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

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
    """Return columns starting with `prefix`, sorted by integer suffix.

    Requires columns like:
      prefix + "0", prefix + "1", ... prefix + "D-1"
    """
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix)]
    if not cols:
        raise KeyError(f"No columns found with prefix '{prefix}'")

    pairs = []
    for c in cols:
        suf = c[len(prefix):]
        try:
            i = int(suf)
        except Exception:
            continue
        pairs.append((i, c))

    if not pairs:
        raise KeyError(
            f"Found columns with prefix '{prefix}', but none had integer suffixes. "
            f"Examples: {cols[:10]}"
        )

    pairs.sort(key=lambda t: t[0])
    return [c for _, c in pairs]


def _prefix_to_numpy(df: pd.DataFrame, prefix: str, *, name: str) -> np.ndarray:
    cols = _sorted_prefixed_cols(df, prefix)

    sub = df[cols]
    try:
        mat = sub.to_numpy(dtype=np.float32, copy=False)
    except Exception:
        # Fallback: coerce each column explicitly
        sub2 = sub.apply(pd.to_numeric, errors="coerce")
        if sub2.isna().any().any():
            bad_cols = sub2.columns[sub2.isna().any()].tolist()[:10]
            raise TypeError(
                f"{name}: non-numeric values encountered in columns with prefix '{prefix}'. "
                f"Example bad columns: {bad_cols}. "
                f"Your parquet likely contains object dtype in embedding columns."
            )
        mat = sub2.to_numpy(dtype=np.float32, copy=False)

    if mat.ndim != 2:
        raise TypeError(f"{name} prefix '{prefix}' did not produce a 2D matrix.")
    return mat


@dataclass(frozen=True)
class PairPrefixes:
    left: str
    right: str


class EmbeddingPairDataset(Dataset):
    """Prefix-based pair dataset: returns (x1, x2, y).

    Expected columns:
      - label
      - {x1_prefix}0..{x1_prefix}D-1
      - {x2_prefix}0..{x2_prefix}D-1

    NOTE: requires x1 and x2 to have the same dimension.
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

    Defaults match your merged parquet:
      - fraud_txt_emb_* , real_txt_emb_*
      - fraud_aligned_* , real_aligned_*   (teacher)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        fraud_txt_prefix: str = "fraud_txt_emb_",
        real_txt_prefix: str = "real_txt_emb_",
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


class TextPairDataset(Dataset):
    """Text-text pair dataset for evaluation (and/or training) that returns (fraud_txt, real_txt, label).

    This is handy for your updated evaluator2 where you:
      - translate fraud_txt -> image-like
      - translate real_txt  -> image-like
      - compute cosine(z_fraud, z_real)
      - compute ROC AUC vs label
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        fraud_txt_prefix: str = "fraud_txt_emb_",
        real_txt_prefix: str = "real_txt_emb_",
        label_col: LabelCol = "label",
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)

        if not _has_prefix(self.df, fraud_txt_prefix):
            raise KeyError(f"TextPairDataset: missing fraud_txt columns with prefix '{fraud_txt_prefix}'")
        if not _has_prefix(self.df, real_txt_prefix):
            raise KeyError(f"TextPairDataset: missing real_txt columns with prefix '{real_txt_prefix}'")

        self.fraud_txt = _prefix_to_numpy(self.df, fraud_txt_prefix, name="fraud_txt")
        self.real_txt = _prefix_to_numpy(self.df, real_txt_prefix, name="real_txt")

        y_series = _get_label_series(self.df, label_col)
        self.labels = _coerce_numeric_label(y_series, name="label")

        if len(self.fraud_txt) != len(self.real_txt) or len(self.fraud_txt) != len(self.labels):
            raise ValueError("TextPairDataset: length mismatch among fraud_txt/real_txt/labels.")
        if self.fraud_txt.shape[1] != self.real_txt.shape[1]:
            raise ValueError(
                f"TextPairDataset: dim mismatch fraud_txt_dim={self.fraud_txt.shape[1]} vs real_txt_dim={self.real_txt.shape[1]}"
            )

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, idx: int):
        fraud_txt = torch.from_numpy(self.fraud_txt[idx])
        real_txt = torch.from_numpy(self.real_txt[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return fraud_txt, real_txt, y


class TextImageBinaryPairDataset(Dataset):
    """Dataset for your new 4-pairs-per-row text->image parquet.

    Expected columns (defaults match the builder script you used):
      - left_txt_emb_0..D-1
      - right_img_emb_0..K-1
      - label (0/1)
      - (optional) pair_kind (string)
      - (optional) orig_row_id (int)

    Returns by default:
      (left_txt, right_img, label)

    Optionally returns metadata:
      (left_txt, right_img, label, pair_kind_id, orig_row_id)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        left_txt_prefix: str = "left_txt_emb_",
        right_img_prefix: str = "right_img_emb_",
        label_col: LabelCol = "label",
        pair_kind_col: Optional[str] = "pair_kind",
        orig_row_id_col: Optional[str] = "orig_row_id",
        return_pair_kind: bool = False,
        return_orig_row_id: bool = False,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)

        if not _has_prefix(self.df, left_txt_prefix):
            raise KeyError(f"TextImageBinaryPairDataset: missing left_txt columns with prefix '{left_txt_prefix}'")
        if not _has_prefix(self.df, right_img_prefix):
            raise KeyError(f"TextImageBinaryPairDataset: missing right_img columns with prefix '{right_img_prefix}'")

        self.left_txt = _prefix_to_numpy(self.df, left_txt_prefix, name="left_txt")
        self.right_img = _prefix_to_numpy(self.df, right_img_prefix, name="right_img")

        y_series = _get_label_series(self.df, label_col)
        self.labels = _coerce_numeric_label(y_series, name="label")

        n = len(self.labels)
        if len(self.left_txt) != n or len(self.right_img) != n:
            raise ValueError("TextImageBinaryPairDataset: length mismatch among left_txt/right_img/labels.")

        self.return_pair_kind = bool(return_pair_kind)
        self.return_orig_row_id = bool(return_orig_row_id)

        # Optional metadata
        self.pair_kind_id: Optional[np.ndarray] = None
        self.pair_kind_map: Optional[Dict[str, int]] = None
        if self.return_pair_kind:
            if not pair_kind_col or pair_kind_col not in self.df.columns:
                raise KeyError(
                    f"return_pair_kind=True but pair_kind_col='{pair_kind_col}' not found in dataframe."
                )
            kinds = self.df[pair_kind_col].astype(str).to_numpy()
            uniq = pd.unique(kinds).tolist()
            self.pair_kind_map = {k: i for i, k in enumerate(uniq)}
            self.pair_kind_id = np.array([self.pair_kind_map[k] for k in kinds], dtype=np.int64)

        self.orig_row_id: Optional[np.ndarray] = None
        if self.return_orig_row_id:
            if not orig_row_id_col or orig_row_id_col not in self.df.columns:
                raise KeyError(
                    f"return_orig_row_id=True but orig_row_id_col='{orig_row_id_col}' not found in dataframe."
                )
            self.orig_row_id = self.df[orig_row_id_col].to_numpy(dtype=np.int64, copy=False)

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, idx: int):
        left_txt = torch.from_numpy(self.left_txt[idx])
        right_img = torch.from_numpy(self.right_img[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)

        if not self.return_pair_kind and not self.return_orig_row_id:
            return left_txt, right_img, y

        out = [left_txt, right_img, y]
        if self.return_pair_kind:
            assert self.pair_kind_id is not None
            out.append(torch.tensor(self.pair_kind_id[idx], dtype=torch.long))
        if self.return_orig_row_id:
            assert self.orig_row_id is not None
            out.append(torch.tensor(self.orig_row_id[idx], dtype=torch.long))
        return tuple(out)
