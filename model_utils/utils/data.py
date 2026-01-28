from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


LabelCol = Union[int, str]


def _coerce_numeric_label(series: pd.Series, *, name: str = "label") -> np.ndarray:
    """
    Convert a label Series to float32 numeric array.
    Accepts ints/floats/bools/"0"/"1"/"0.0"/"1.0".
    Raises a clear error if non-numeric values exist.
    """
    s = series

    if s.dtype == bool:
        s = s.astype(np.int64)

    # object -> try numeric conversion
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


def _get_label_series(df: pd.DataFrame, label_col: LabelCol, fallback_label_idx: int = 2) -> pd.Series:
    """
    label_col can be:
      - int: df.iloc[:, label_col]
      - str: df[label_col] if exists, else df.iloc[:, fallback_label_idx]
    """
    if isinstance(label_col, int):
        return df.iloc[:, int(label_col)]
    if isinstance(label_col, str):
        if label_col in df.columns:
            return df[label_col]
        return df.iloc[:, int(fallback_label_idx)]
    raise TypeError(f"label_col must be int or str, got {type(label_col)}")


def _slice_to_numpy(df: pd.DataFrame, sl: slice, *, name: str) -> np.ndarray:
    """
    Extract a slice of columns and return float32 numpy array.

    Works for:
      - wide numeric columns (expected)
      - single list/array column (vector-in-cell) if sl selects 1 col
    """
    sub = df.iloc[:, sl]

    # Case A: wide numeric matrix
    try:
        mat = sub.to_numpy(dtype=np.float32, copy=False)
        if mat.ndim != 2:
            raise ValueError
        return mat
    except Exception:
        pass

    # Case B: single object/list column
    if sub.shape[1] == 1:
        col = sub.columns[0]
        vals = df[col].to_numpy()
        try:
            mat = np.stack([np.asarray(v, dtype=np.float32) for v in vals], axis=0)
            return mat
        except Exception as e:
            raise TypeError(
                f"Column '{col}' for {name} looks like object/list data but couldn't be stacked. "
                f"Example type: {type(vals[0])}. Error: {e}"
            )

    raise TypeError(
        f"{name} slice {sl.start}:{sl.stop} could not be converted to float32 matrix. "
        f"Your parquet likely has object dtypes in embedding columns."
    )


class EmbeddingPairDataset(Dataset):
    """
    Old pair dataset (Siamese training/export): returns (x1, x2, y)

    Default assumes parquet layout:
      col 0: fraudulent_name
      col 1: real_name
      col 2: label
      col 3..770: fraud image embedding (768)
      col 771..1538: real image embedding (768)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        x1_slice: slice = slice(3, 771),
        x2_slice: slice = slice(771, 1539),
        label_col: LabelCol = "label",
        fallback_label_idx: int = 2,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)

        self.x1 = _slice_to_numpy(self.df, x1_slice, name="x1")
        self.x2 = _slice_to_numpy(self.df, x2_slice, name="x2")

        y_series = _get_label_series(self.df, label_col, fallback_label_idx=fallback_label_idx)
        self.y = _coerce_numeric_label(y_series, name="label")

        if len(self.x1) != len(self.x2) or len(self.x1) != len(self.y):
            raise ValueError("EmbeddingPairDataset: length mismatch among x1/x2/y.")

    def __len__(self) -> int:
        return int(len(self.y))

    def __getitem__(self, idx: int):
        x1 = torch.from_numpy(self.x1[idx])
        x2 = torch.from_numpy(self.x2[idx])
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x1, x2, y


class Text2ImgDistillDataset(Dataset):
    """
    Text -> Image distillation dataset (NO brand_id).

    Returns 5 items:
      (fraud_txt, real_txt, fraud_img_teacher, real_img_teacher, label)

    Default assumes combined parquet layout:
      0: fraudulent_name
      1: real_name
      2: label
      3..770: fraud image teacher (768)
      771..1538: real image teacher (768)
      1539..2306: fraud text input (768)
      2307..3074: real text input (768)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        # Accept the "new" names:
        fraud_img_slice: slice = slice(3, 771),
        real_img_slice: slice = slice(771, 1539),
        fraud_txt_slice: slice = slice(1539, 2307),
        real_txt_slice: slice = slice(2307, 3075),
        # Also accept older aliases if some code still uses them:
        fake_img_slice: slice | None = None,
        label_col: LabelCol = "label",
        fallback_label_idx: int = 2,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)

        # Back-compat: some code might pass fake_img_slice instead of fraud_img_slice
        if fake_img_slice is not None:
            fraud_img_slice = fake_img_slice

        self.fraud_img = _slice_to_numpy(self.df, fraud_img_slice, name="fraud_img_teacher")
        self.real_img  = _slice_to_numpy(self.df, real_img_slice,  name="real_img_teacher")
        self.fraud_txt = _slice_to_numpy(self.df, fraud_txt_slice, name="fraud_txt")
        self.real_txt  = _slice_to_numpy(self.df, real_txt_slice,  name="real_txt")

        y_series = _get_label_series(self.df, label_col, fallback_label_idx=fallback_label_idx)
        self.labels = _coerce_numeric_label(y_series, name="label")

        n = len(self.labels)
        for name, arr in [
            ("fraud_img_teacher", self.fraud_img),
            ("real_img_teacher", self.real_img),
            ("fraud_txt", self.fraud_txt),
            ("real_txt", self.real_txt),
        ]:
            if len(arr) != n:
                raise ValueError(f"Text2ImgDistillDataset: length mismatch for {name} vs labels.")

    def __len__(self) -> int:
        return int(len(self.labels))

    def __getitem__(self, idx: int):
        fraud_txt = torch.from_numpy(self.fraud_txt[idx])
        real_txt  = torch.from_numpy(self.real_txt[idx])
        fraud_img_teacher = torch.from_numpy(self.fraud_img[idx])
        real_img_teacher  = torch.from_numpy(self.real_img[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return fraud_txt, real_txt, fraud_img_teacher, real_img_teacher, y
