#!/usr/bin/env python3
"""
Create SigLIP glyph-image embeddings for name pairs.

Input:
  - CSV or Parquet file with at least columns: `fraudulent_name`, `real_name`.

Output:
  - CSV or Parquet file containing the original columns plus:
      fraud_emb_0..fraud_emb_{D-1}
      real_emb_0..real_emb_{D-1}

Example:
  python extra_notebooks/create_embeddings_run.py \
    --input ../Ref/train_pairs_ref.parquet \
    --output ../Downloads/train_pairs_with_siglip_embeddings.parquet \
    --batch-size 128
"""

from __future__ import annotations

import argparse
import os
import re
import tempfile
import unicodedata
import uuid
from functools import lru_cache
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager
from tqdm.auto import tqdm
from transformers import AutoProcessor, SiglipVisionModel

import time


# -----------------------------
# Small status heartbeat
# -----------------------------
def heartbeat(last, every=30):
    now = time.time()
    if now - last >= every:
        print(f"[HEARTBEAT] still working at {time.strftime('%H:%M:%S')}", flush=True)
        return now
    return last


# -----------------------------
# Font caching
# -----------------------------
@lru_cache(maxsize=1)
def _get_font_path() -> str:
    return font_manager.findfont("DejaVu Sans", fallback_to_default=True)


@lru_cache(maxsize=2048)
def _get_font(font_size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(_get_font_path(), int(font_size))
    except Exception:
        return ImageFont.load_default()


# -----------------------------
# Measurement
# -----------------------------
def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int, int, int]:
    """
    Returns (w, h, x0, y0) where x0,y0 are bbox origin offsets (can be negative).
    """
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    return (x1 - x0), (y1 - y0), x0, y0


def _max_fitting_font_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    allowed_w: int,
    allowed_h: int,
    base_font_size: int,
    min_font_size: int,
    max_font_cap: int,
) -> Tuple[int, int, int, int, int]:
    """
    Find the LARGEST integer font size that fits within (allowed_w, allowed_h).
    One line only. No truncation. No clipping.

    Returns (fs, w, h, x0, y0) measured at that fs.
    """

    min_fs = max(1, int(min_font_size))
    cap_fs = max(min_fs, int(max_font_cap))
    base_fs = max(min_fs, min(cap_fs, int(base_font_size)))

    # Estimate a near-optimal size from a single measurement at base_fs
    font0 = _get_font(base_fs)
    w0, h0, _, _ = _measure_text(draw, text, font0)

    # Degenerate bbox: just return min_fs
    if w0 <= 0 or h0 <= 0:
        font = _get_font(min_fs)
        w, h, x0, y0 = _measure_text(draw, text, font)
        return min_fs, w, h, x0, y0

    # Allow scale > 1 (this was a key mistake before)
    scale = min(allowed_w / w0, allowed_h / h0)
    est = int(base_fs * scale)
    est = max(min_fs, min(cap_fs, est))

    def fits(fs: int) -> bool:
        font = _get_font(fs)
        w, h, _, _ = _measure_text(draw, text, font)
        return (w <= allowed_w) and (h <= allowed_h)

    # Ensure est fits (rounding can make it overflow)
    fs = est
    while fs > min_fs and not fits(fs):
        fs -= 1

    # If even min_fs doesn't fit, we still return min_fs (it will clip; unavoidable with this constraint)
    if fs == min_fs and not fits(fs):
        font = _get_font(min_fs)
        w, h, x0, y0 = _measure_text(draw, text, font)
        return min_fs, w, h, x0, y0

    # Find an upper bound by doubling until it overflows or hits cap
    lo = fs
    hi = min(cap_fs, max(lo + 1, lo))
    while hi < cap_fs and fits(hi):
        lo = hi
        hi = min(cap_fs, hi * 2)

    # Now binary search (lo fits, hi may not fit or hi==cap_fs)
    if fits(hi):
        lo = hi  # hi fits and is cap_fs

    left = lo
    right = hi if not fits(hi) else lo

    if right == left:
        font = _get_font(left)
        w, h, x0, y0 = _measure_text(draw, text, font)
        return left, w, h, x0, y0

    # Ensure right is the first non-fitting size (or cap)
    if fits(right):
        # Shouldn't happen, but guard
        font = _get_font(right)
        w, h, x0, y0 = _measure_text(draw, text, font)
        return right, w, h, x0, y0

    low = left
    high = right  # high does NOT fit
    while low + 1 < high:
        mid = (low + high) // 2
        if fits(mid):
            low = mid
        else:
            high = mid

    font = _get_font(low)
    w, h, x0, y0 = _measure_text(draw, text, font)
    return low, w, h, x0, y0


# -----------------------------
# Rendering (224x224 final)
# -----------------------------
def generate_glyph_image(
    text: str,
    image_size: Tuple[int, int] = (224, 224),
    pad: int = 0,                    # FIXED: default low padding
    min_font_size: int = 1,          # allow tiny fonts to avoid clipping
    base_font_size: int = 120,       # starting point for the estimate
    max_font_cap: int = 1024,        # search cap; binary search keeps this fast
    hd_enabled: bool = True,         # FIXED: HD fallback default ON
    hd_threshold: int = 8,           # trigger 2× when the best font at 224 is <= this
    hd_scale: int = 2,               # quickest useful HD
) -> Image.Image:
    """
    Returns exactly (W,H) image (default 224x224), one line, no truncation, no clipping if possible.

    HD fallback: if the best-fitting font at native resolution is very small, render at 2× and downsample.
    """
    text = unicodedata.normalize("NFC", str(text))

    W, H = int(image_size[0]), int(image_size[1])
    pad = int(pad)

    # ----- First decide if we should use HD fallback (based on native best font size) -----
    tmp = Image.new("RGB", (W, H), (0, 0, 0))
    tmp_draw = ImageDraw.Draw(tmp)
    allowed_w = max(1, W - 2 * pad)
    allowed_h = max(1, H - 2 * pad)

    fs_native, _, _, _, _ = _max_fitting_font_size(
        draw=tmp_draw,
        text=text,
        allowed_w=allowed_w,
        allowed_h=allowed_h,
        base_font_size=base_font_size,
        min_font_size=min_font_size,
        max_font_cap=max_font_cap,
    )

    use_hd = bool(hd_enabled and (hd_scale >= 2) and (fs_native <= int(hd_threshold)))

    if not use_hd:
        img = Image.new("RGB", (W, H), (0, 0, 0))
        draw = ImageDraw.Draw(img)

        fs, w, h, x0, y0 = _max_fitting_font_size(
            draw=draw,
            text=text,
            allowed_w=allowed_w,
            allowed_h=allowed_h,
            base_font_size=base_font_size,
            min_font_size=min_font_size,
            max_font_cap=max_font_cap,
        )

        x = (W - w) // 2 - x0
        y = (H - h) // 2 - y0
        draw.text((x, y), text, font=_get_font(fs), fill=(255, 255, 255))
        return img

    # ----- HD fallback: render at 2× (or hd_scale×), then downsample to 224 -----
    S = int(hd_scale)
    WR, HR = W * S, H * S
    pad_r = pad * S
    allowed_wr = max(1, WR - 2 * pad_r)
    allowed_hr = max(1, HR - 2 * pad_r)

    img_r = Image.new("RGB", (WR, HR), (0, 0, 0))
    draw_r = ImageDraw.Draw(img_r)

    fs_r, w_r, h_r, x0_r, y0_r = _max_fitting_font_size(
        draw=draw_r,
        text=text,
        allowed_w=allowed_wr,
        allowed_h=allowed_hr,
        base_font_size=base_font_size * S,
        min_font_size=max(1, min_font_size * S),
        max_font_cap=max_font_cap * S,
    )

    x_r = (WR - w_r) // 2 - x0_r
    y_r = (HR - h_r) // 2 - y0_r
    draw_r.text((x_r, y_r), text, font=_get_font(fs_r), fill=(255, 255, 255))

    # fastest decent downsample
    return img_r.resize((W, H), resample=Image.Resampling.BILINEAR)


# -----------------------------
# I/O helpers
# -----------------------------
def normalize_name(x: object, strip_com: bool) -> str:
    s = unicodedata.normalize("NFC", str(x))
    s = s.lstrip("-").strip()
    if strip_com:
        s = re.sub(r"\.com$", "", s, flags=re.IGNORECASE)
    return s


def load_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def save_table(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if path.lower().endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


# -----------------------------
# Embedding
# -----------------------------
@torch.no_grad()
def embed_unique_names(
    uniq_names: List[str],
    model: SiglipVisionModel,
    processor: AutoProcessor,
    device: torch.device,
    batch_size: int,
    pad: int,
    min_font_size: int,
    base_font_size: int,
    max_font_cap: int,
    hd_enabled: bool,
    hd_threshold: int,
    hd_scale: int,
    memmap_path: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    n = len(uniq_names)
    if n == 0:
        raise ValueError("No names to embed (unique name list is empty).")

    use_amp = device.type == "cuda"
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

    # First batch: infer D
    first_chunk = uniq_names[: min(batch_size, n)]
    first_imgs = [
        generate_glyph_image(
            name,
            image_size=(224, 224),
            pad=pad,
            min_font_size=min_font_size,
            base_font_size=base_font_size,
            max_font_cap=max_font_cap,
            hd_enabled=hd_enabled,
            hd_threshold=hd_threshold,
            hd_scale=hd_scale,
        )
        for name in first_chunk
    ]
    first_batch = processor(images=first_imgs, return_tensors="pt")
    first_batch = {k: v.to(device, non_blocking=True) for k, v in first_batch.items()}

    with autocast_ctx:
        out0 = model(**first_batch)
        embs0 = out0.pooler_output

    embs0 = F.normalize(embs0, dim=-1, eps=1e-8).float().detach().cpu().numpy()
    dim = int(embs0.shape[1])

    bytes_needed = n * dim * 4
    use_memmap = memmap_path is not None or bytes_needed >= 1_500_000_000

    backing_path = ""
    if use_memmap:
        backing_path = memmap_path or os.path.join(tempfile.gettempdir(), f"siglip_embs_{uuid.uuid4().hex}.mmap")
        emb_mat = np.memmap(backing_path, mode="w+", dtype=np.float32, shape=(n, dim))
    else:
        emb_mat = np.empty((n, dim), dtype=np.float32)

    emb_mat[: len(first_chunk)] = embs0

    start_idx = len(first_chunk)
    if start_idx < n:
        last_hb = time.time()
        for start in tqdm(range(start_idx, n, batch_size), desc="Embedding batches"):
            last_hb = heartbeat(last_hb, every=30)

            chunk = uniq_names[start : start + batch_size]
            imgs = [
                generate_glyph_image(
                    name,
                    image_size=(224, 224),
                    pad=pad,
                    min_font_size=min_font_size,
                    base_font_size=base_font_size,
                    max_font_cap=max_font_cap,
                    hd_enabled=hd_enabled,
                    hd_threshold=hd_threshold,
                    hd_scale=hd_scale,
                )
                for name in chunk
            ]

            batch = processor(images=imgs, return_tensors="pt")
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with autocast_ctx:
                out = model(**batch)
                embs_t = out.pooler_output

            embs = F.normalize(embs_t, dim=-1, eps=1e-8).float().cpu().numpy()
            emb_mat[start : start + len(chunk)] = embs

    return emb_mat, backing_path


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Create SigLIP glyph embeddings and merge into a pairs file")
    parser.add_argument("--input", required=True, help="Input CSV/Parquet containing name pairs")
    parser.add_argument("--output", required=True, help="Output Parquet/CSV path")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size")
    parser.add_argument("--model", default="google/siglip-base-patch16-224", help="HuggingFace model name")
    parser.add_argument("--device", default=None, help='Override device: "cuda", "cpu", or "mps"')
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on number of rows processed (debug)")

    # Rendering knobs (defaults fixed per your request)
    parser.add_argument("--pad", type=int, default=0, help="Padding around text inside 224 canvas (DEFAULT 0)")
    parser.add_argument("--min-font-size", type=int, default=1, help="Minimum font size (DEFAULT 1)")
    parser.add_argument("--base-font-size", type=int, default=120, help="Base font size for estimating scale (DEFAULT 120)")
    parser.add_argument("--max-font-cap", type=int, default=1024, help="Max font size search cap (DEFAULT 1024)")

    # HD fallback (DEFAULT ON)
    parser.add_argument("--no-hd", action="store_true", help="Disable HD fallback")
    parser.add_argument("--hd-threshold", type=int, default=8, help="Use HD fallback if best native font <= this (DEFAULT 8)")
    parser.add_argument("--hd-scale", type=int, default=2, help="HD render scale factor (DEFAULT 2)")

    # .com stripping
    parser.add_argument("--strip-com", action="store_true", help='Force stripping trailing ".com" per row')
    parser.add_argument("--no-strip-com", action="store_true", help="Disable auto-stripping of .com")

    parser.add_argument("--memmap-path", default=None, help="Optional path to store the embedding matrix as a memmap file")
    args = parser.parse_args()

    df = load_table(args.input)

    if "fraudulent_name" not in df.columns or "real_name" not in df.columns:
        raise ValueError("Input must include columns: fraudulent_name, real_name")

    if args.max_rows is not None:
        df = df.head(int(args.max_rows))

    # Decide whether to strip .com
    if args.no_strip_com:
        strip_com = False
        print("[INFO] .com stripping disabled via --no-strip-com")
    elif args.strip_com:
        strip_com = True
        print('[INFO] Forcing trailing ".com" stripping via --strip-com')
    else:
        sample_n = min(len(df), 10_000)
        if sample_n > 0:
            fraud_sample = df["fraudulent_name"].head(sample_n).astype(str).str.strip()
            real_sample = df["real_name"].head(sample_n).astype(str).str.strip()
            strip_com = bool(
                fraud_sample.str.contains(r"\.com$", case=False, regex=True, na=False).any()
                or real_sample.str.contains(r"\.com$", case=False, regex=True, na=False).any()
            )
        else:
            strip_com = False

        if strip_com:
            print('[INFO] Auto-detected trailing ".com" in sample; stripping will be applied.')
        else:
            print('[INFO] No trailing ".com" detected in sample; stripping will NOT be applied.')

    df = df.copy()
    df["fraudulent_name"] = df["fraudulent_name"].map(lambda x: normalize_name(x, strip_com))
    df["real_name"] = df["real_name"].map(lambda x: normalize_name(x, strip_com))

    # Device selection
    if args.device:
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "You requested CUDA but torch.cuda.is_available() is False.\n"
                f"Your torch build is: {torch.__version__}\n"
                "Install a CUDA-enabled PyTorch build and run on a GPU node."
            )
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print(f"[INFO] Using device: {device}")

    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = SiglipVisionModel.from_pretrained(args.model, torch_dtype=torch_dtype)

    try:
        processor = AutoProcessor.from_pretrained(args.model, use_fast=True)
    except TypeError:
        processor = AutoProcessor.from_pretrained(args.model)

    model.to(device)
    model.eval()

    all_names = pd.concat([df["fraudulent_name"], df["real_name"]], ignore_index=True)
    uniq_names = pd.unique(all_names.astype(str)).tolist()
    print(f"[INFO] Unique names to embed: {len(uniq_names):,}")

    hd_enabled = not args.no_hd

    emb_mat, backing_path = embed_unique_names(
        uniq_names=uniq_names,
        model=model,
        processor=processor,
        device=device,
        batch_size=int(args.batch_size),
        pad=int(args.pad),
        min_font_size=int(args.min_font_size),
        base_font_size=int(args.base_font_size),
        max_font_cap=int(args.max_font_cap),
        hd_enabled=hd_enabled,
        hd_threshold=int(args.hd_threshold),
        hd_scale=int(args.hd_scale),
        memmap_path=args.memmap_path,
    )
    if backing_path:
        print(f"[INFO] Embedding matrix stored as memmap at: {backing_path}")

    dim = int(emb_mat.shape[1])

    name_to_idx = pd.Series(np.arange(len(uniq_names), dtype=np.int64), index=pd.Index(uniq_names, dtype="object"))
    fraud_idx = df["fraudulent_name"].map(name_to_idx).to_numpy(dtype=np.int64)
    real_idx = df["real_name"].map(name_to_idx).to_numpy(dtype=np.int64)

    fraud_embs = np.asarray(emb_mat[fraud_idx], dtype=np.float32)
    real_embs = np.asarray(emb_mat[real_idx], dtype=np.float32)

    fraud_cols = [f"fraud_emb_{i}" for i in range(dim)]
    real_cols = [f"real_emb_{i}" for i in range(dim)]

    out_df = pd.concat(
        [
            df.reset_index(drop=True),
            pd.DataFrame(fraud_embs, columns=fraud_cols),
            pd.DataFrame(real_embs, columns=real_cols),
        ],
        axis=1,
    )

    save_table(out_df, args.output)
    print(f"[INFO] Wrote: {args.output}")


if __name__ == "__main__":
    main()