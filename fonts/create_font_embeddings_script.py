#!/usr/bin/env python3
"""Create SigLIP glyph-image embeddings for name pairs.

Input:
  - CSV or Parquet file with at least columns: `fraudulent_name`, `real_name`.

Output:
  - CSV or Parquet file containing the original columns plus cosine similarity
  - NPZ file of embeddings

Key fixes vs the earlier script:
  - No longer pre-renders/stores ALL glyph images in RAM (renders per batch).
  - Caches the font (huge speedup).
  - Robust optional/auto ".com" stripping (row-wise, end-only, case-insensitive).
  - Uses model outputs on the chosen device; stores embeddings in float32 on CPU.
  - Handles CUDA/MPS/CPU device selection more safely.

Example:
  python /Users/valeriexin/IAP_2026_urop/Itau-group-2/fonts/create_font_embeddings_script.py
    --input /Users/valeriexin/IAP_2026_urop/Itau-group-2/data/processed/test_pairs_10k.parquet
    --output testing.parquet
    --output-embeddings testing_embed.npz 
    --font "DejaVu Sans"
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
from sklearn.metrics import roc_curve, roc_auc_score
import sys

import time

# Task ID and total tasks passed from SLURM
my_task_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])

# Define fonts and corresponding output names
fonts = ["Tahoma", "Comic Sans MS","Caveat", "Helvetica", "Kristi", "calibri", "Pacifico", "Nanum Brush Script", "Source Code Pro", "Roboto"]
output_parquets = ["tahoma_validate_pairs_ref_10k.parquet", 
                   "comicsans_validate_pairs_ref_10k.parquet",
                   "caveat_validate_pairs_ref_10k.parquet",
                   "helvetica_validate_pairs_ref_10k.parquet",
                   "kristi_validate_pairs_ref_10k.parquet",
                   "calibri_validate_pairs_ref_10k.parquet",
                   "pacifico_validate_pairs_ref_10k.parquet",
                   "nanum_validate_pairs_ref_10k.parquet",
                   "sourcecodepro_validate_pairs_ref_10k.parquet",
                   "roboto_validate_pairs_ref_10k.parquet"]
output_embeddings = ["tahoma_embed_validate_pairs_ref_10k.npz", 
                   "comicsans_embed_validate_pairs_ref_10k.npz",
                   "caveat_embed_validate_pairs_ref_10k.npz",
                   "helvetica_embed_validate_pairs_ref_10k.npz",
                   "kristi_embed_validate_pairs_ref_10k.npz",
                   "calibri_embed_validate_pairs_ref_10k.npz",
                   "pacifico_embed_validate_pairs_ref_10k.npz",
                   "nanum_embed_validate_pairs_ref_10k.npz",
                   "sourcecodepro_embed_validate_pairs_ref_10k.npz",
                   "roboto_embed_validate_pairs_ref_10k.npz"]

# Assign the font/output for this task
font = fonts[my_task_id]
output_parquet = output_parquets[my_task_id]
output_npz = output_embeddings[my_task_id]

def upload_fonts(path):
    font_dirs = [path]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

def heartbeat(last, every=30):
    now = time.time()
    if now - last >= every:
        print(f"[HEARTBEAT] still working at {time.strftime('%H:%M:%S')}", flush=True)
        return now
    return last


def _get_unicode_font(font_size: int = 14, font: str = 'DejaVu Sans') -> ImageFont.FreeTypeFont:
    """Return a font that supports a wide range of Unicode (cached)."""
    try:
        path = font_manager.findfont(font, fallback_to_default=True)
        return ImageFont.truetype(path, font_size)
    except Exception:
        return ImageFont.load_default()


def generate_glyph_image(
    text: str,
    image_size: Tuple[int, int] = (224, 224),
    font_size: int = 14,
    font: str = 'DejaVu Sans'
) -> Image.Image:
    """Convert text into a centered glyph image."""
    text = unicodedata.normalize("NFC", str(text))
    image = Image.new("RGB", image_size, color=(0, 0, 0))
    draw = ImageDraw.Draw(image)
    font = _get_unicode_font(font_size=font_size, font = font)

    # Pillow compatibility
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except Exception:
        text_width, text_height = draw.textsize(text, font=font)

    x = (image_size[0] - text_width) // 2
    y = (image_size[1] - text_height) // 2
    draw.text((x, y), text, font=font, fill=(255, 255, 255))
    return image


def normalize_name(x: object, strip_com: bool) -> str:
    """Normalize names consistently for embedding + joining."""
    s = unicodedata.normalize("NFC", str(x))
    s = s.lstrip("-").strip()
    if strip_com:
        s = re.sub(r"\.com$", "", s, flags=re.IGNORECASE)
    return s


def load_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    return pd.read_parquet(path)


def save_table(df: pd.DataFrame, path: str, scores: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df["cosine_sim"] = scores
    if path.lower().endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
        print(f"Saved dataframe w/ cosine similarities to {path}")


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


@torch.no_grad()
def embed_unique_names(
    uniq_names: List[str],
    model: SiglipVisionModel,
    processor: AutoProcessor,
    device: torch.device,
    batch_size: int,
    font_size: int,
    font: str,
    memmap_path: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    """Embed unique names -> (N, D) float32 array (or memmap), returned on CPU."""
    n = len(uniq_names)
    if n == 0:
        raise ValueError("No names to embed (unique name list is empty).")

    use_amp = device.type == "cuda"
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()

    # First batch: infer D
    first_chunk = uniq_names[: min(batch_size, n)]
    first_imgs = [generate_glyph_image(name, font_size=font_size, font= font) for name in first_chunk]
    first_batch = processor(images=first_imgs, return_tensors="pt")
    first_batch = {k: v.to(device, non_blocking=True) for k, v in first_batch.items()}

    with autocast_ctx:
        out0 = model(**first_batch)
        embs0 = out0.pooler_output

    embs0 = F.normalize(embs0, dim=-1, eps=1e-8).float().detach().cpu().numpy()
    dim = int(embs0.shape[1])

    bytes_needed = n * dim * 4
    use_memmap = memmap_path is not None or bytes_needed >= 1_500_000_000  # ~1.5GB

    backing_path = ""
    if use_memmap:
        if memmap_path is None:
            backing_path = os.path.join(tempfile.gettempdir(), f"siglip_embs_{uuid.uuid4().hex}.mmap")
        else:
            backing_path = memmap_path
        emb_mat = np.memmap(backing_path, mode="w+", dtype=np.float32, shape=(n, dim))
    else:
        emb_mat = np.empty((n, dim), dtype=np.float32)

    emb_mat[: len(first_chunk)] = embs0

    start_idx = len(first_chunk)
    if start_idx < n:
        last_hb = time.time()
        for start in tqdm(range(start_idx, n, batch_size), desc="Embedding batches"):
        
            chunk = uniq_names[start : start + batch_size]
        
            imgs = [generate_glyph_image(name, font_size=font_size) for name in chunk]
        
            batch = processor(images=imgs, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
        
            out = model(**batch)

            embs = F.normalize(out.pooler_output, dim=-1).cpu().numpy()
            emb_mat[start : start + len(chunk)] = embs

    return emb_mat, backing_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create SigLIP glyph embeddings and merge into a pairs file")
    parser.add_argument("--input", required=True, help="Input CSV/Parquet containing name pairs")
    # parser.add_argument("--output", required=True, help="Output Parquet/CSV path")
    # parser.add_argument("--output-embeddings", required=True, help="Output NPZ path")
    # parser.add_argument("--font", default="DejaVu Sans", help="Font Type")
    parser.add_argument("--batch-size", type=int, default=128, help="Embedding batch size")
    parser.add_argument("--model", default="google/siglip-base-patch16-224", help="HuggingFace model name")
    parser.add_argument("--device", default=None, help='Override device: "cuda", "cpu", or "mps"')
    parser.add_argument("--font-size", type=int, default=14, help="Font size for glyph rendering")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional cap on number of rows processed (debug)")
    parser.add_argument("--strip-com", action="store_true", help='Force stripping trailing ".com" per row')
    parser.add_argument("--no-strip-com", action="store_true", help="Disable auto-stripping of .com")
    parser.add_argument("--memmap-path", default=None, help="Optional path to store the embedding matrix as a memmap file")
    args = parser.parse_args()
    
    # Upload fonts to font manager
    font_dirs = ["/Users/valeriexin/IAP_2026_urop/font_files"] 
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

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

    # Normalize names
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

    emb_mat, backing_path = embed_unique_names(
        uniq_names=uniq_names,
        model=model,
        processor=processor,
        device=device,
        batch_size=int(args.batch_size),
        font_size=int(args.font_size),
        font=font,
        memmap_path=args.memmap_path,
    )
    if backing_path:
        print(f"[INFO] Embedding matrix stored as memmap at: {backing_path}")

    dim = int(emb_mat.shape[1])

    # Map name -> index
    name_to_idx = pd.Series(np.arange(len(uniq_names), dtype=np.int64), index=pd.Index(uniq_names, dtype="object"))

    fraud_idx = df["fraudulent_name"].map(name_to_idx).to_numpy(dtype=np.int64)
    real_idx = df["real_name"].map(name_to_idx).to_numpy(dtype=np.int64)

    # Gather embeddings
    fraud_embs = np.asarray(emb_mat[fraud_idx], dtype=np.float32)
    real_embs = np.asarray(emb_mat[real_idx], dtype=np.float32)
    
    cosine_sim = np.sum(fraud_embs * real_embs, axis=-1)
    save_table(df, output_parquet, cosine_sim)
    print(f"[INFO] Wrote new table to {output_parquet}")
    
    if "label" in df.columns:
        y_true = df["label"].to_numpy(dtype=np.int32)

        fpr, tpr, thresholds = roc_curve(y_true, cosine_sim)
        youden_j = tpr - fpr
        auc = roc_auc_score(y_true, cosine_sim)
        optimal_threshold = float(thresholds[np.argmax(youden_j)])
        print(f"[INFO] Optimal cosine threshold: {optimal_threshold}")
        print(f"[INFO] ROC-AUC Score: {auc}")

    else:
        print("[INFO] No label column found; skipping threshold computation")
    
    np.savez_compressed(
        output_npz,
        names=np.array(uniq_names, dtype=object),
        embeddings=np.asarray(emb_mat, dtype=np.float32),
    )
    print(f"Embeddings saved to {output_npz}")

if __name__ == "__main__":
    main()
