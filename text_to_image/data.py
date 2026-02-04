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
# DATASET (NEW FORMAT ONLY)
# -------------------------------------------------
class TextTeacherPairDataset(Dataset):
    """
    Positive-only text → image/teacher embedding dataset.

    Expected columns:
      - label
      - {txt_prefix}0..D-1
      - {img_prefix}0..D-1

    Returns:
      (txt, img, label)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        txt_prefix: str = "left_txt_emb_",
        img_prefix: str = "right_img_emb_",
        label_col: LabelCol = "label",
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)

        if not _has_prefix(self.df, txt_prefix):
            raise KeyError(f"Missing text embedding columns with prefix '{txt_prefix}'")
        if not _has_prefix(self.df, img_prefix):
            raise KeyError(f"Missing image embedding columns with prefix '{img_prefix}'")

        self.txt = _prefix_to_numpy(self.df, txt_prefix, name="txt")
        self.img = _prefix_to_numpy(self.df, img_prefix, name="img")

        y_series = _get_label_series(self.df, label_col)
        self.labels = _coerce_numeric_label(y_series, name="label")

        if len(self.txt) != len(self.img) or len(self.txt) != len(self.labels):
            raise ValueError("Length mismatch among txt, img, and label")

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, idx: int):
        txt = torch.from_numpy(self.txt[idx])
        img = torch.from_numpy(self.img[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return txt, img, y
