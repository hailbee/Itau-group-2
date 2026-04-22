from __future__ import annotations

import heapq
import json
import os
import re
import time
import unicodedata
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from main import (
    REPO_ROOT,
    SOURCE_ALIAS_COLUMNS,
    STRING_FEATURES,
    choose_device,
    feature_source_for,
    levenshtein_distance,
    load_model_bundle,
    load_projector,
    partial_ratio,
    resolve_model_path,
    token_set_ratio,
)


DEFAULT_DOMAIN_DATASET = REPO_ROOT / "data" / "benign_domains.csv"
DEFAULT_MODEL_PATH = REPO_ROOT / "saved_models" / "total_5f_img_model.joblib"
DEFAULT_SIGLIP_MODEL = os.getenv("SIGLIP_MODEL_NAME", "google/siglip-base-patch16-224")
DEFAULT_CHUNK_SIZE = int(os.getenv("MATCHER_CHUNK_SIZE", "256"))
DEFAULT_PROJECTOR_DIR = REPO_ROOT / "projectors"
DEFAULT_HF_CACHE_DIR = REPO_ROOT / ".cache" / "huggingface"
DEFAULT_PRECOMPUTED_STORE_DIR = REPO_ROOT / "precomputed" / "benign_total5f_img"
LEGACY_PRECOMPUTED_STORE_DIR = REPO_ROOT / "precomputed"
LEGACY_MODEL_STORE_DIRS: dict[str, tuple[str, ...]] = {
    "total_5f_img_model.joblib": ("benign_total5f",),
}

os.environ.setdefault("HF_HOME", str(DEFAULT_HF_CACHE_DIR))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DEFAULT_HF_CACHE_DIR / "hub"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

FONT_FILES: dict[str, Path] = {
    "deja": REPO_ROOT / "assets" / "fonts" / "deja.ttf",
    "unifont": REPO_ROOT / "assets" / "fonts" / "unifont.ttf",
    "gentium": REPO_ROOT / "assets" / "fonts" / "gentium.ttf",
    "libre": REPO_ROOT / "assets" / "fonts" / "libre.ttf",
    "exo2": REPO_ROOT / "assets" / "fonts" / "exo2.ttf",
    "doulos": REPO_ROOT / "assets" / "fonts" / "doulos.ttf",
    "cousine": REPO_ROOT / "assets" / "fonts" / "cousine.ttf",
}

PROJECTOR_PATHS: dict[str, Path] = {
    "text": DEFAULT_PROJECTOR_DIR / "text_projector.pt",
    "deja": DEFAULT_PROJECTOR_DIR / "deja_projector.pt",
    "unifont": DEFAULT_PROJECTOR_DIR / "unifont_projector.pt",
    "gentium": DEFAULT_PROJECTOR_DIR / "gentium_projector.pt",
    "libre": DEFAULT_PROJECTOR_DIR / "libre_projector.pt",
    "exo2": DEFAULT_PROJECTOR_DIR / "exo2_projector.pt",
    "doulos": DEFAULT_PROJECTOR_DIR / "doulos_projector.pt",
    "cousine": DEFAULT_PROJECTOR_DIR / "cousine_projector.pt",
}


def default_precomputed_store_dir(model_path: Path | None = None) -> Path:
    resolved_model = Path(model_path).resolve() if model_path is not None else DEFAULT_MODEL_PATH.resolve()
    if resolved_model.name == "total_5f_img_model.joblib":
        return DEFAULT_PRECOMPUTED_STORE_DIR.resolve()
    if resolved_model.name == "total_5f_model.joblib":
        return (REPO_ROOT / "precomputed" / "benign_total5f").resolve()

    stem = resolved_model.stem
    if stem.endswith("_model"):
        stem = stem[: -len("_model")]
    return (REPO_ROOT / "precomputed" / f"benign_{stem}").resolve()


def candidate_precomputed_store_dirs(model_path: Path | None = None) -> list[Path]:
    override = os.getenv("PRECOMPUTED_STORE_DIR")
    if override:
        return [Path(override).resolve()]

    resolved_model = Path(model_path).resolve() if model_path is not None else DEFAULT_MODEL_PATH.resolve()
    candidates = [default_precomputed_store_dir(resolved_model)]
    for legacy_dir_name in LEGACY_MODEL_STORE_DIRS.get(resolved_model.name, ()):
        candidates.append((REPO_ROOT / "precomputed" / legacy_dir_name).resolve())
    candidates.append(LEGACY_PRECOMPUTED_STORE_DIR.resolve())

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        deduped.append(candidate)
        seen.add(key)
    return deduped


def resolve_precomputed_store_dir(model_path: Path | None = None) -> Path:
    override = os.getenv("PRECOMPUTED_STORE_DIR")
    if override:
        return Path(override).resolve()
    return default_precomputed_store_dir(model_path)


@dataclass
class FontCosineColumn:
    key: str
    label: str


@dataclass
class SearchHit:
    domain: str
    mean_font_cosine: float
    font_cosines: dict[str, float]
    exact_match: bool = False
    normalized_domain: str | None = None


@dataclass
class SearchReport:
    query: str
    normalized_query: str
    scanned_rows: int
    total_rows_target: int
    total_threshold_hits: int
    duration_seconds: float
    feature_mode: str
    warnings: list[str]
    font_columns: list[FontCosineColumn]
    matches: list[SearchHit]
    top_candidates: list[SearchHit]


@dataclass
class PairwiseComparisonReport:
    left_query: str
    right_query: str
    left_host: str
    right_host: str
    left_normalized: str
    right_normalized: str
    threshold: float
    mean_font_cosine: float
    font_columns: list[FontCosineColumn]
    font_cosines: dict[str, float]
    is_spoof: bool
    model_prediction: bool | None
    model_probability: float | None


class SearchCancelled(RuntimeError):
    def __init__(self, progress: dict[str, Any] | None = None):
        super().__init__("Search cancelled by user.")
        self.progress = progress or {}


@lru_cache(maxsize=4096)
def _load_font(font_path: str, font_size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, int(font_size))


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    return (x1 - x0), (y1 - y0), x0, y0


def _max_fitting_font_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font_path: Path,
    allowed_w: int,
    allowed_h: int,
    base_font_size: int,
    min_font_size: int,
    max_font_cap: int,
) -> tuple[int, int, int, int, int]:
    min_fs = max(1, int(min_font_size))
    cap_fs = max(min_fs, int(max_font_cap))
    base_fs = max(min_fs, min(cap_fs, int(base_font_size)))

    font0 = _load_font(str(font_path), base_fs)
    w0, h0, _, _ = _measure_text(draw, text, font0)
    if w0 <= 0 or h0 <= 0:
        font = _load_font(str(font_path), min_fs)
        w, h, x0, y0 = _measure_text(draw, text, font)
        return min_fs, w, h, x0, y0

    scale = min(allowed_w / max(w0, 1), allowed_h / max(h0, 1))
    est = max(min_fs, min(cap_fs, int(base_fs * scale)))

    def fits(font_size: int) -> bool:
        font = _load_font(str(font_path), font_size)
        w, h, _, _ = _measure_text(draw, text, font)
        return w <= allowed_w and h <= allowed_h

    font_size = est
    while font_size > min_fs and not fits(font_size):
        font_size -= 1

    if font_size == min_fs and not fits(font_size):
        font = _load_font(str(font_path), min_fs)
        w, h, x0, y0 = _measure_text(draw, text, font)
        return min_fs, w, h, x0, y0

    lo = font_size
    hi = min(cap_fs, lo + 1)
    while hi < cap_fs and fits(hi):
        lo = hi
        hi = min(cap_fs, hi * 2)

    if fits(hi):
        font = _load_font(str(font_path), hi)
        w, h, x0, y0 = _measure_text(draw, text, font)
        return hi, w, h, x0, y0

    low, high = lo, hi
    while low + 1 < high:
        mid = (low + high) // 2
        if fits(mid):
            low = mid
        else:
            high = mid

    font = _load_font(str(font_path), low)
    w, h, x0, y0 = _measure_text(draw, text, font)
    return low, w, h, x0, y0


def generate_glyph_image(
    text: str,
    font_path: Path,
    image_size: tuple[int, int] = (224, 224),
    pad: int = 4,
    min_font_size: int = 1,
    base_font_size: int = 120,
    max_font_cap: int = 1024,
    hd_enabled: bool = True,
    hd_threshold: int = 10,
    hd_min_scale: int = 2,
    hd_max_scale: int = 8,
    hd_target_font_px: int = 28,
) -> Image.Image:
    text = unicodedata.normalize("NFC", str(text))
    width, height = int(image_size[0]), int(image_size[1])
    pad = int(pad)

    probe = Image.new("RGB", (width, height), (0, 0, 0))
    probe_draw = ImageDraw.Draw(probe)
    allowed_w = max(1, width - 2 * pad)
    allowed_h = max(1, height - 2 * pad)

    native_font_size, _, _, _, _ = _max_fitting_font_size(
        draw=probe_draw,
        text=text,
        font_path=font_path,
        allowed_w=allowed_w,
        allowed_h=allowed_h,
        base_font_size=base_font_size,
        min_font_size=min_font_size,
        max_font_cap=max_font_cap,
    )

    use_hd = bool(hd_enabled and native_font_size <= int(hd_threshold) and hd_max_scale >= 2)
    if not use_hd:
        img = Image.new("RGB", (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        font_size, text_w, text_h, x0, y0 = _max_fitting_font_size(
            draw=draw,
            text=text,
            font_path=font_path,
            allowed_w=allowed_w,
            allowed_h=allowed_h,
            base_font_size=base_font_size,
            min_font_size=min_font_size,
            max_font_cap=max_font_cap,
        )
        x = (width - text_w) // 2 - x0
        y = (height - text_h) // 2 - y0
        draw.text((x, y), text, font=_load_font(str(font_path), font_size), fill=(255, 255, 255))
        return img

    target = max(1, int(hd_target_font_px))
    needed_scale = int(np.ceil(target / max(1, int(native_font_size))))
    scale = int(min(int(hd_max_scale), max(int(hd_min_scale), max(2, needed_scale))))

    width_hd, height_hd = width * scale, height * scale
    pad_hd = pad * scale
    img_hd = Image.new("RGB", (width_hd, height_hd), (0, 0, 0))
    draw_hd = ImageDraw.Draw(img_hd)
    font_size_hd, text_w_hd, text_h_hd, x0_hd, y0_hd = _max_fitting_font_size(
        draw=draw_hd,
        text=text,
        font_path=font_path,
        allowed_w=max(1, width_hd - 2 * pad_hd),
        allowed_h=max(1, height_hd - 2 * pad_hd),
        base_font_size=base_font_size * scale,
        min_font_size=max(1, min_font_size * scale),
        max_font_cap=max_font_cap * scale,
    )
    x_hd = (width_hd - text_w_hd) // 2 - x0_hd
    y_hd = (height_hd - text_h_hd) // 2 - y0_hd
    draw_hd.text((x_hd, y_hd), text, font=_load_font(str(font_path), font_size_hd), fill=(255, 255, 255))
    return img_hd.resize((width, height), resample=Image.Resampling.LANCZOS)

def canonicalize_domain_host(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    text = unicodedata.normalize("NFC", str(value)).strip().lower()
    if not text:
        return ""

    scheme_match = re.match(r"^[a-z][a-z0-9+.-]*://", text)
    has_explicit_authority = bool(scheme_match) or text.startswith("//")
    if scheme_match:
        text = text[scheme_match.end() :]
    elif text.startswith("//"):
        text = text[2:]

    # Keep only host portion
    host_end = len(text)
    for delimiter in ("/", "?", "#"):
        index = text.find(delimiter)
        if index != -1:
            host_end = min(host_end, index)
    authority = text[:host_end]

    # Treat `@` as URL user info only when the input clearly looked like a URL.
    if (has_explicit_authority or host_end < len(text)) and "@" in authority:
        authority = authority.rsplit("@", 1)[-1]

    text = authority

    # Remove port, like :8080
    text = text.split(":", 1)[0]

    # Remove trailing dot and common www prefix
    text = text.rstrip(".")
    text = re.sub(r"^www\.", "", text)

    return text


def normalize_domain_string(value: object) -> str:
    text = canonicalize_domain_host(value)

    # For datasets like google.com, google.ie, google.xyz
    # return the part before the first dot
    return text.split(".", 1)[0]

def _push_topk(heap: list[tuple[float, int, SearchHit]], score: float, counter: int, hit: SearchHit, limit: int) -> None:
    item = (float(score), int(counter), hit)
    if len(heap) < limit:
        heapq.heappush(heap, item)
        return
    if item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


class SiglipBackbone:
    def __init__(self, model_name: str, device: torch.device):
        from transformers import AutoProcessor, SiglipTextModel, SiglipVisionModel

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        cache_dir = DEFAULT_HF_CACHE_DIR
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, use_fast=True, cache_dir=str(cache_dir))
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=str(cache_dir))
        self.vision_model = self._load_component(
            SiglipVisionModel,
            model_name=model_name,
            device=device,
            dtype=dtype,
            cache_dir=cache_dir,
        )
        self.text_model = self._load_component(
            SiglipTextModel,
            model_name=model_name,
            device=device,
            dtype=dtype,
            cache_dir=cache_dir,
        )
        self.vision_model.eval()
        self.text_model.eval()

    @staticmethod
    def _load_component(
        model_cls: type,
        *,
        model_name: str,
        device: torch.device,
        dtype: torch.dtype,
        cache_dir: Path,
    ):
        load_kwargs = {"cache_dir": str(cache_dir)}
        if device.type == "cuda":
            try:
                return model_cls.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    **load_kwargs,
                ).to(device)
            except TypeError as exc:
                if "torch_dtype" not in str(exc):
                    raise

        return model_cls.from_pretrained(
            model_name,
            **load_kwargs,
        ).to(device=device, dtype=dtype)

    @torch.inference_mode()
    def encode_texts(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        results: list[np.ndarray] = []
        use_amp = self.device.type == "cuda"
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
        for start in range(0, len(texts), max(1, batch_size)):
            batch_texts = list(texts[start : start + max(1, batch_size)])
            inputs = self.processor(text=batch_texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with autocast_ctx:
                outputs = self.text_model(**inputs)
                embeddings = outputs.pooler_output
            embeddings = F.normalize(embeddings, dim=-1, eps=1e-8).float().cpu().numpy()
            results.append(embeddings)
        return np.vstack(results).astype(np.float32, copy=False)

    @torch.inference_mode()
    def encode_glyphs(self, texts: Sequence[str], font_path: Path, batch_size: int) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        results: list[np.ndarray] = []
        use_amp = self.device.type == "cuda"
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
        for start in range(0, len(texts), max(1, batch_size)):
            batch_texts = list(texts[start : start + max(1, batch_size)])
            images = [generate_glyph_image(text, font_path=font_path) for text in batch_texts]
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with autocast_ctx:
                outputs = self.vision_model(**inputs)
                embeddings = outputs.pooler_output
            embeddings = F.normalize(embeddings, dim=-1, eps=1e-8).float().cpu().numpy()
            results.append(embeddings)
        return np.vstack(results).astype(np.float32, copy=False)


class PrecomputedFeatureStore:
    def __init__(self, store_dir: Path):
        self.store_dir = store_dir.resolve()
        metadata_path = self.store_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing precomputed store metadata at {metadata_path}")

        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.row_count = int(self.metadata["rows"])
        self.feature_names = list(self.metadata.get("feature_names", []))
        self.domains_path = (self.store_dir / self.metadata["domains_file"]).resolve()
        self.sources = dict(self.metadata["sources"])
        self.embeddings = {
            source_key: np.load(self._source_file_path(source_info), mmap_mode="r")
            for source_key, source_info in self.sources.items()
        }

    def _source_file_path(self, source_info: dict[str, Any]) -> Path:
        source_dir = source_info.get("dir")
        file_name = source_info["file"]
        if source_dir:
            return (self.store_dir / str(source_dir) / str(file_name)).resolve()
        return (self.store_dir / str(file_name)).resolve()

    def supports(
        self,
        *,
        model_name: str,
        feature_names: Sequence[str],
        required_sources: Sequence[str],
    ) -> bool:
        if str(self.metadata.get("model_name", "")) != str(model_name):
            return False
        if self.feature_names and list(self.feature_names) != list(feature_names):
            return False
        return all(source_key in self.embeddings for source_key in required_sources)

    def iter_chunks(
        self,
        *,
        chunk_size: int,
        max_rows: int | None = None,
    ) -> Iterable[tuple[pd.DataFrame, dict[str, np.ndarray]]]:
        limit = self.row_count if max_rows is None else min(self.row_count, int(max_rows))
        start = 0
        for chunk in pd.read_csv(self.domains_path, chunksize=max(1, int(chunk_size))):
            if start >= limit:
                break

            remaining = limit - start
            if len(chunk) > remaining:
                chunk = chunk.head(remaining)

            end = start + len(chunk)
            source_slices = {
                source_key: embedding_matrix[start:end]
                for source_key, embedding_matrix in self.embeddings.items()
            }
            yield chunk, source_slices
            start = end


class DomainMatcher:
    def __init__(
        self,
        model_path: Path | None = None,
        dataset_path: Path | None = None,
        siglip_model_name: str = DEFAULT_SIGLIP_MODEL,
        projector_paths: dict[str, Path] | None = None,
    ):
        self.model_path = resolve_model_path(
            str((model_path or DEFAULT_MODEL_PATH).resolve()),
            REPO_ROOT / "saved_models",
        )
        self.bundle = load_model_bundle(self.model_path)
        self.estimator = self.bundle.estimator
        self.dataset_path = (dataset_path or DEFAULT_DOMAIN_DATASET).resolve()
        self.device = choose_device(os.getenv("MATCHER_DEVICE"))
        self.siglip_model_name = siglip_model_name
        self.backbone: SiglipBackbone | None = None
        self.projectors = self._load_projectors(projector_paths or PROJECTOR_PATHS)
        self.required_sources = [
            feature_source_for(self.bundle, name)
            for name in self.bundle.feature_names
            if name not in STRING_FEATURES
        ]
        self.required_sources = list(dict.fromkeys(self.required_sources))
        self.font_feature_names = [
            name
            for name in self.bundle.feature_names
            if name.startswith("cosine_") and name != "text_cosine"
        ]
        self.positive_label = self.bundle.metadata.get("positive_label", 1)
        self.precomputed_store_notices: list[str] = []
        self.precomputed_store = self._load_precomputed_store()
        self._dataset_row_count_cache: int | None = None
        missing_query_projectors = [
            source_key for source_key in self.required_sources if self.projectors.get(source_key) is None
        ]
        if self.precomputed_store is not None and missing_query_projectors:
            missing_text = ", ".join(missing_query_projectors)
            raise RuntimeError(
                "A precomputed store is available, but query-side projector checkpoints are missing for: "
                f"{missing_text}"
            )

        missing_font_sources = [
            source for source in self.required_sources if source in FONT_FILES and not FONT_FILES[source].exists()
        ]
        if missing_font_sources:
            missing_text = ", ".join(missing_font_sources)
            raise FileNotFoundError(f"Missing font files for: {missing_text}")
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Missing benign-domain dataset at {self.dataset_path}")

    def _backbone(self) -> SiglipBackbone:
        if self.backbone is None:
            self.backbone = SiglipBackbone(self.siglip_model_name, self.device)
        return self.backbone

    def _load_projectors(self, projector_paths: dict[str, Path]) -> dict[str, tuple[Any, int] | None]:
        loaded: dict[str, tuple[Any, int] | None] = {}
        for source_key, path in projector_paths.items():
            env_key = f"{source_key.upper()}_PROJECTOR_PATH"
            override = os.getenv(env_key)
            resolved = Path(override).resolve() if override else path.resolve()
            loaded[source_key] = load_projector(resolved, self.device) if resolved.exists() else None
        return loaded

    def _load_precomputed_store(self) -> PrecomputedFeatureStore | None:
        for store_dir in candidate_precomputed_store_dirs(self.model_path):
            metadata_path = store_dir / "metadata.json"
            if not metadata_path.exists():
                continue

            store = PrecomputedFeatureStore(store_dir)
            if store.supports(
                model_name=self.model_path.name,
                feature_names=self.bundle.feature_names,
                required_sources=self.required_sources,
            ):
                return store

            store_model_name = str(store.metadata.get("model_name", "unknown model"))
            self.precomputed_store_notices.append(
                f"Ignoring precomputed store at {store.store_dir} because it was built for {store_model_name}, "
                f"not {self.model_path.name}."
            )

        return None

    def _feature_mode(self) -> str:
        if not self.required_sources:
            return "string_metrics_only"
        if self.precomputed_store is not None:
            return "precomputed_projected"
        missing = [source for source in self.required_sources if self.projectors.get(source) is None]
        if not missing:
            return "projected_runtime"
        return f"raw_cosine_fallback ({', '.join(missing)})"

    def _warnings(self, max_rows: int | None) -> list[str]:
        warnings: list[str] = []
        warnings.extend(self.precomputed_store_notices)
        if self.precomputed_store is not None and self.required_sources:
            warnings.append(
                f"Using precomputed projected embeddings from {self.precomputed_store.store_dir}."
            )
        elif self.precomputed_store is not None:
            warnings.append(
                f"Using cached candidate domains from {self.precomputed_store.store_dir}."
            )
        missing = [source for source in self.required_sources if self.projectors.get(source) is None]
        if missing and self.precomputed_store is None:
            warnings.append(
                "Projector files are missing for "
                + ", ".join(missing)
                + ". The app is using raw cosine similarity as an approximation for those learned features."
            )
        if max_rows is not None:
            warnings.append(f"Only the first {max_rows:,} benign domains were scanned for this request.")
        warnings.append(
            "A full scan across all benign domains is computationally heavy because each request recomputes text "
            "metrics for the query against many candidate domains."
        )
        return warnings

    @staticmethod
    def _font_label(feature_name: str) -> str:
        return feature_name.replace("cosine_", "").replace("_", " ").title()

    def _dataset_row_count(self) -> int:
        if self.precomputed_store is not None:
            return int(self.precomputed_store.row_count)
        if self._dataset_row_count_cache is None:
            with self.dataset_path.open("r", encoding="utf-8", newline="") as handle:
                self._dataset_row_count_cache = max(0, sum(1 for _ in handle) - 1)
        return self._dataset_row_count_cache

    def _target_row_count(self, max_rows: int | None) -> int:
        total_rows = self._dataset_row_count()
        if max_rows is None:
            return total_rows
        return min(total_rows, int(max_rows))

    def _project_if_available(self, source_key: str, embeddings: np.ndarray) -> np.ndarray:
        projector_entry = self.projectors.get(source_key)
        if projector_entry is None:
            return embeddings
        projector, input_dim = projector_entry
        if embeddings.shape[1] != int(input_dim):
            raise RuntimeError(
                f"Projector for {source_key!r} expects {input_dim} features but encoder produced {embeddings.shape[1]}."
            )
        tensor = torch.from_numpy(embeddings.astype(np.float32, copy=False)).to(self.device)
        with torch.inference_mode():
            projected = projector.encode(tensor)
            projected = F.normalize(projected, dim=1, eps=1e-8).float().cpu().numpy()
        return projected.astype(np.float32, copy=False)

    def _pairwise_text_metrics(self, query: str, candidates: Sequence[str]) -> dict[str, np.ndarray]:
        token_scores = np.empty((len(candidates),), dtype=np.float32)
        partial_scores = np.empty((len(candidates),), dtype=np.float32)
        levenshtein_scores = np.empty((len(candidates),), dtype=np.float32)
        for index, candidate in enumerate(candidates):
            token_scores[index] = token_set_ratio(query, candidate)
            partial_scores[index] = partial_ratio(query, candidate)
            levenshtein_scores[index] = -float(levenshtein_distance(query, candidate))
        return {
            "token_set_ratio": token_scores,
            "partial_ratio": partial_scores,
            "levenshtein_distance_score": levenshtein_scores,
        }

    def _cosine_scores(self, query_embedding: np.ndarray, candidate_embeddings: np.ndarray) -> np.ndarray:
        query = query_embedding.astype(np.float32, copy=False)
        candidates = candidate_embeddings.astype(np.float32, copy=False)
        query_norm = max(float(np.linalg.norm(query)), 1e-8)
        normalized_query = query / query_norm
        candidate_norms = np.linalg.norm(candidates, axis=1, keepdims=True)
        candidate_norms = np.clip(candidate_norms, 1e-8, None)
        normalized_candidates = candidates / candidate_norms
        return normalized_candidates @ normalized_query

    def _assign_source_feature_values(
        self,
        feature_map: dict[str, np.ndarray],
        source_key: str,
        values: np.ndarray,
    ) -> None:
        for feature_name in SOURCE_ALIAS_COLUMNS.get(source_key, ()):
            feature_map[feature_name] = values

    def _build_feature_matrix(
        self,
        query: str,
        candidates: Sequence[str],
        query_text_embedding: np.ndarray | None,
        query_font_embeddings: dict[str, np.ndarray],
        batch_size: int,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        feature_map = self._pairwise_text_metrics(query, candidates)

        if "text" in self.required_sources:
            candidate_text = self._backbone().encode_texts(candidates, batch_size=batch_size)
            candidate_text = self._project_if_available("text", candidate_text)
            assert query_text_embedding is not None
            self._assign_source_feature_values(
                feature_map,
                "text",
                self._cosine_scores(query_text_embedding, candidate_text),
            )

        for source_key in self.required_sources:
            if source_key == "text":
                continue
            font_path = FONT_FILES[source_key]
            candidate_font_embeddings = self._backbone().encode_glyphs(
                candidates,
                font_path=font_path,
                batch_size=batch_size,
            )
            candidate_font_embeddings = self._project_if_available(source_key, candidate_font_embeddings)
            self._assign_source_feature_values(
                feature_map,
                source_key,
                self._cosine_scores(query_font_embeddings[source_key], candidate_font_embeddings),
            )

        columns = [feature_map[name] for name in self.bundle.feature_names]
        matrix = np.column_stack(columns).astype(np.float32, copy=False)
        return matrix, feature_map

    def _build_feature_matrix_from_precomputed(
        self,
        query: str,
        candidates: Sequence[str],
        source_slices: dict[str, np.ndarray],
        query_text_embedding: np.ndarray | None,
        query_font_embeddings: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        feature_map = self._pairwise_text_metrics(query, candidates)

        if "text" in self.required_sources:
            assert query_text_embedding is not None
            self._assign_source_feature_values(
                feature_map,
                "text",
                self._cosine_scores(query_text_embedding, source_slices["text"]),
            )

        for source_key in self.required_sources:
            if source_key == "text":
                continue
            self._assign_source_feature_values(
                feature_map,
                source_key,
                self._cosine_scores(
                    query_font_embeddings[source_key],
                    source_slices[source_key],
                ),
            )

        columns = [feature_map[name] for name in self.bundle.feature_names]
        matrix = np.column_stack(columns).astype(np.float32, copy=False)
        return matrix, feature_map

    def _prepare_query_embeddings(self, normalized_query: str, batch_size: int) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
        query_text_embedding = None
        if "text" in self.required_sources:
            query_text_embedding = self._backbone().encode_texts([normalized_query], batch_size=batch_size)
            query_text_embedding = self._project_if_available("text", query_text_embedding)[0]

        query_font_embeddings: dict[str, np.ndarray] = {}
        for source_key in self.required_sources:
            if source_key == "text":
                continue
            raw_embedding = self._backbone().encode_glyphs(
                [normalized_query],
                font_path=FONT_FILES[source_key],
                batch_size=batch_size,
            )
            query_font_embeddings[source_key] = self._project_if_available(source_key, raw_embedding)[0]
        return query_text_embedding, query_font_embeddings

    def compare_pair(
        self,
        left_query: str,
        right_query: str,
        *,
        threshold: float = 0.5,
    ) -> PairwiseComparisonReport:
        left_normalized = normalize_domain_string(left_query)
        right_normalized = normalize_domain_string(right_query)
        if not left_normalized:
            raise ValueError("Please enter a non-empty left-side domain string.")
        if not right_normalized:
            raise ValueError("Please enter a non-empty right-side domain string.")
        if not self.font_feature_names:
            raise RuntimeError("The selected model does not provide any font cosine features.")

        left_host = canonicalize_domain_host(left_query)
        right_host = canonicalize_domain_host(right_query)
        batch_size = 1
        query_text_embedding, query_font_embeddings = self._prepare_query_embeddings(
            left_normalized,
            batch_size=batch_size,
        )
        feature_matrix, feature_map = self._build_feature_matrix(
            left_normalized,
            [right_normalized],
            query_text_embedding,
            query_font_embeddings,
            batch_size=batch_size,
        )
        font_cosines = {
            feature_name: float(feature_map[feature_name][0])
            for feature_name in self.font_feature_names
        }
        mean_font_cosine = float(np.mean(list(font_cosines.values())))
        bounded_threshold = max(0.0, min(1.0, float(threshold)))
        is_spoof = mean_font_cosine >= bounded_threshold

        model_prediction: bool | None = None
        raw_prediction = self.bundle.estimator.predict(feature_matrix)
        if len(raw_prediction) > 0:
            prediction = raw_prediction[0]
            model_prediction = bool(
                prediction == self.positive_label or str(prediction) == str(self.positive_label)
            )

        model_probability: float | None = None
        if hasattr(self.bundle.estimator, "predict_proba"):
            probabilities = self.bundle.estimator.predict_proba(feature_matrix)
            if probabilities.ndim == 2 and probabilities.shape[0] > 0 and probabilities.shape[1] > 0:
                classes = list(getattr(self.bundle.estimator, "classes_", []))
                positive_index = probabilities.shape[1] - 1
                for index, class_label in enumerate(classes):
                    if class_label == self.positive_label or str(class_label) == str(self.positive_label):
                        positive_index = index
                        break
                positive_index = max(0, min(positive_index, probabilities.shape[1] - 1))
                model_probability = float(probabilities[0, positive_index])

        return PairwiseComparisonReport(
            left_query=left_query,
            right_query=right_query,
            left_host=left_host,
            right_host=right_host,
            left_normalized=left_normalized,
            right_normalized=right_normalized,
            threshold=bounded_threshold,
            mean_font_cosine=mean_font_cosine,
            font_columns=[
                FontCosineColumn(key=feature_name, label=self._font_label(feature_name))
                for feature_name in self.font_feature_names
            ],
            font_cosines=font_cosines,
            is_spoof=is_spoof,
            model_prediction=model_prediction,
            model_probability=model_probability,
        )

    def search(
        self,
        query: str,
        *,
        min_mean_font_cosine: float = 0.5,
        top_k: int = 25,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_rows: int | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
        cancel_callback: Callable[[], bool] | None = None,
    ) -> SearchReport:
        started_at = time.time()
        normalized_query = normalize_domain_string(query)
        if not normalized_query:
            raise ValueError("Please enter a non-empty domain string.")
        if not self.font_feature_names:
            raise RuntimeError("The selected model does not provide any font cosine features.")

        batch_size = max(1, int(chunk_size))
        total_rows_target = self._target_row_count(max_rows)
        feature_mode = self._feature_mode()

        def build_progress(
            *,
            status: str,
            stage: str,
            stage_detail: str,
            scanned_rows: int,
            total_threshold_hits: int,
        ) -> dict[str, Any]:
            return {
                "status": status,
                "stage": stage,
                "stage_detail": stage_detail,
                "query": query,
                "normalized_query": normalized_query,
                "scanned_rows": scanned_rows,
                "total_rows_target": total_rows_target,
                "total_threshold_hits": total_threshold_hits,
                "duration_seconds": time.time() - started_at,
                "feature_mode": feature_mode,
            }

        def emit_progress(
            *,
            status: str,
            stage: str,
            stage_detail: str,
            scanned_rows: int,
            total_threshold_hits: int,
        ) -> None:
            if progress_callback is not None:
                progress_callback(
                    build_progress(
                        status=status,
                        stage=stage,
                        stage_detail=stage_detail,
                        scanned_rows=scanned_rows,
                        total_threshold_hits=total_threshold_hits,
                    )
                )

        def raise_if_cancelled(
            *,
            stage: str,
            stage_detail: str,
            scanned_rows: int,
            total_threshold_hits: int,
        ) -> None:
            if cancel_callback is not None and cancel_callback():
                raise SearchCancelled(
                    build_progress(
                        status="cancelled",
                        stage=stage,
                        stage_detail=stage_detail,
                        scanned_rows=scanned_rows,
                        total_threshold_hits=total_threshold_hits,
                    )
                )

        emit_progress(
            status="running",
            stage="prepare_query",
            stage_detail=(
                "Candidate source ready. Preparing query features from the current input."
                if self.precomputed_store is not None
                else "Preparing query features. Candidate features will be generated during the scan."
            ),
            scanned_rows=0,
            total_threshold_hits=0,
        )
        raise_if_cancelled(
            stage="prepare_query",
            stage_detail="Search stopped before query preparation began.",
            scanned_rows=0,
            total_threshold_hits=0,
        )
        query_text_embedding, query_font_embeddings = self._prepare_query_embeddings(normalized_query, batch_size=batch_size)

        matches_heap: list[tuple[float, int, SearchHit]] = []
        overall_heap: list[tuple[float, int, SearchHit]] = []
        scanned_rows = 0
        total_threshold_hits = 0
        counter = 0

        emit_progress(
            status="running",
            stage="scan_candidates",
            stage_detail="Query features ready. Scanning candidate domains.",
            scanned_rows=0,
            total_threshold_hits=0,
        )
        raise_if_cancelled(
            stage="scan_candidates",
            stage_detail="Search stopped before candidate scanning began.",
            scanned_rows=0,
            total_threshold_hits=0,
        )

        if self.precomputed_store is not None:
            chunk_iter: Iterable[tuple[pd.DataFrame, dict[str, np.ndarray]]] = self.precomputed_store.iter_chunks(
                chunk_size=batch_size,
                max_rows=max_rows,
            )
        else:
            chunk_iter = (
                (
                    chunk,
                    {},
                )
                for chunk in pd.read_csv(self.dataset_path, usecols=["domain"], chunksize=batch_size)
                )

        for chunk, source_slices in chunk_iter:
            raise_if_cancelled(
                stage="scan_candidates",
                stage_detail="Search stopped during candidate scanning.",
                scanned_rows=scanned_rows,
                total_threshold_hits=total_threshold_hits,
            )
            if self.precomputed_store is None:
                if max_rows is not None and scanned_rows >= max_rows:
                    break

                if max_rows is not None:
                    remaining = max_rows - scanned_rows
                    if remaining <= 0:
                        break
                    chunk = chunk.head(remaining)

            raw_domains = chunk["domain"].fillna("").astype(str).tolist()
            if self.precomputed_store is not None and "normalized_domain" in chunk.columns:
                normalized_candidates = chunk["normalized_domain"].fillna("").astype(str).tolist()
                _feature_matrix, feature_map = self._build_feature_matrix_from_precomputed(
                    normalized_query,
                    normalized_candidates,
                    source_slices,
                    query_text_embedding,
                    query_font_embeddings,
                )
            else:
                normalized_candidates = [normalize_domain_string(domain) for domain in raw_domains]
                _feature_matrix, feature_map = self._build_feature_matrix(
                    normalized_query,
                    normalized_candidates,
                    query_text_embedding,
                    query_font_embeddings,
                    batch_size=batch_size,
                )
            for index, domain in enumerate(raw_domains):
                font_cosines = {
                    feature_name: float(feature_map[feature_name][index])
                    for feature_name in self.font_feature_names
                }
                mean_font_cosine = float(np.mean(list(font_cosines.values())))
                if mean_font_cosine >= float(min_mean_font_cosine):
                    total_threshold_hits += 1

                hit = SearchHit(
                    domain=domain,
                    mean_font_cosine=mean_font_cosine,
                    font_cosines=font_cosines,
                    normalized_domain=normalized_candidates[index],
                )

                _push_topk(overall_heap, mean_font_cosine, counter, hit, max(1, int(top_k)))
                if mean_font_cosine >= float(min_mean_font_cosine):
                    _push_topk(matches_heap, mean_font_cosine, counter, hit, max(1, int(top_k)))
                counter += 1

            scanned_rows += len(raw_domains)
            emit_progress(
                status="running",
                stage="scan_candidates",
                stage_detail="Scanning candidate domains.",
                scanned_rows=scanned_rows,
                total_threshold_hits=total_threshold_hits,
            )

        emit_progress(
            status="running",
            stage="rank_results",
            stage_detail="Candidate scan finished. Ranking the strongest mean-font-cosine matches.",
            scanned_rows=scanned_rows,
            total_threshold_hits=total_threshold_hits,
        )
        raise_if_cancelled(
            stage="rank_results",
            stage_detail="Search stopped before result ranking finished.",
            scanned_rows=scanned_rows,
            total_threshold_hits=total_threshold_hits,
        )

        matches = [item[2] for item in sorted(matches_heap, key=lambda item: (item[0], -item[1]), reverse=True)]
        top_candidates = [item[2] for item in sorted(overall_heap, key=lambda item: (item[0], -item[1]), reverse=True)]
        return SearchReport(
            query=query,
            normalized_query=normalized_query,
            scanned_rows=scanned_rows,
            total_rows_target=total_rows_target,
            total_threshold_hits=total_threshold_hits,
            duration_seconds=time.time() - started_at,
            feature_mode=self._feature_mode(),
            warnings=self._warnings(max_rows=max_rows),
            font_columns=[
                FontCosineColumn(key=feature_name, label=self._font_label(feature_name))
                for feature_name in self.font_feature_names
            ],
            matches=matches,
            top_candidates=top_candidates,
        )
