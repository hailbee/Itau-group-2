from __future__ import annotations

from typing import List, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


LabelCol = Union[int, str]


# -------------------------------------------------
# helpers
# -------------------------------------------------
def _coerce_numeric_label(series: pd.Series, *, name: str = "label") -> np.ndarray:
    """Convert a label Series to float32 numeric array."""
    if series.dtype == bool:
        series = series.astype(np.int64)

    if series.dtype == object:
        s2 = pd.to_numeric(series, errors="coerce")
        if s2.isna().any():
            bad = series[s2.isna()].unique()[:10]
            raise TypeError(
                f"{name} column contains non-numeric values; examples: {bad}. "
                f"Fix the source parquet/csv so label is numeric."
            )
        series = s2

    return series.to_numpy(dtype=np.float32, copy=False)


def _get_label_series(df: pd.DataFrame, label_col: LabelCol) -> pd.Series:
    """label_col can be an int (iloc) or str (column name)."""
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
    """Return columns starting with prefix, sorted by integer suffix."""
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
            f"Columns with prefix '{prefix}' found, but none have integer suffixes."
        )

    pairs.sort(key=lambda t: t[0])
    return [c for _, c in pairs]


def _prefix_to_numpy(df: pd.DataFrame, prefix: str, *, name: str) -> np.ndarray:
    cols = _sorted_prefixed_cols(df, prefix)
    sub = df[cols]

    try:
        mat = sub.to_numpy(dtype=np.float32, copy=False)
    except Exception:
        sub2 = sub.apply(pd.to_numeric, errors="coerce")
        if sub2.isna().any().any():
            bad_cols = sub2.columns[sub2.isna().any()].tolist()[:10]
            raise TypeError(
                f"{name}: non-numeric values in embedding columns. "
                f"Bad columns: {bad_cols}"
            )
        mat = sub2.to_numpy(dtype=np.float32, copy=False)

    if mat.ndim != 2:
        raise TypeError(f"{name} prefix '{prefix}' did not produce a 2D matrix.")

    return mat


# -------------------------------------------------
# DATASET (4-EMBEDDING FORMAT)
# -------------------------------------------------
class TextTeacherPairDataset(Dataset):
    """
    Dataset where each row contains 4 embeddings:

      fraud_txt_emb_*
      real_txt_emb_*
      fraud_img_emb_*
      real_img_emb_*
      label

    Returns:
      (fraud_txt, real_txt, fraud_img, real_img, label)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        fraud_txt_prefix: str = "fraud_txt_emb_",
        real_txt_prefix: str = "real_txt_emb_",
        fraud_img_prefix: str = "fraud_img_emb_",
        real_img_prefix: str = "real_img_emb_",
        label_col: LabelCol = "label",
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)

        for prefix in [fraud_txt_prefix, real_txt_prefix, fraud_img_prefix, real_img_prefix]:
            if not _has_prefix(self.df, prefix):
                raise KeyError(f"Missing embedding columns with prefix '{prefix}'")

        self.fraud_txt = _prefix_to_numpy(self.df, fraud_txt_prefix, name="fraud_txt")
        self.real_txt = _prefix_to_numpy(self.df, real_txt_prefix, name="real_txt")
        self.fraud_img = _prefix_to_numpy(self.df, fraud_img_prefix, name="fraud_img")
        self.real_img = _prefix_to_numpy(self.df, real_img_prefix, name="real_img")

        y_series = _get_label_series(self.df, label_col)
        self.labels = _coerce_numeric_label(y_series, name="label")

        n = len(self.labels)
        if not (len(self.fraud_txt) == len(self.real_txt) == len(self.fraud_img) == len(self.real_img) == n):
            raise ValueError("Length mismatch among fraud_txt, real_txt, fraud_img, real_img, labels")

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, idx: int):
        fraud_txt = torch.from_numpy(self.fraud_txt[idx])
        real_txt = torch.from_numpy(self.real_txt[idx])
        fraud_img = torch.from_numpy(self.fraud_img[idx])
        real_img = torch.from_numpy(self.real_img[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)

        return fraud_txt, real_txt, fraud_img, real_img, y
