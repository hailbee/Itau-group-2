#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS_DIR = REPO_ROOT / "saved_models"
DEFAULT_MODEL_METADATA_PATH = DEFAULT_MODELS_DIR / "model_metadata.json"
STRING_FEATURES = {"token_set_ratio", "levenshtein_distance_score", "partial_ratio"}


@dataclass(frozen=True)
class SourceSpec:
    key: str
    label: str
    default_fraud_prefix: str
    default_real_prefix: str


SOURCE_SPECS: Dict[str, SourceSpec] = {
    "text": SourceSpec(
        key="text",
        label="text embeddings",
        default_fraud_prefix="fraud_txt_emb_",
        default_real_prefix="real_txt_emb_",
    ),
    "deja": SourceSpec(
        key="deja",
        label="Deja image embeddings",
        default_fraud_prefix="fraud_emb_",
        default_real_prefix="real_emb_",
    ),
    "unifont": SourceSpec(
        key="unifont",
        label="Unifont image embeddings",
        default_fraud_prefix="fraud_emb_",
        default_real_prefix="real_emb_",
    ),
    "gentium": SourceSpec(
        key="gentium",
        label="Gentium image embeddings",
        default_fraud_prefix="fraud_emb_",
        default_real_prefix="real_emb_",
    ),
    "libre": SourceSpec(
        key="libre",
        label="Libre image embeddings",
        default_fraud_prefix="fraud_emb_",
        default_real_prefix="real_emb_",
    ),
    "exo2": SourceSpec(
        key="exo2",
        label="Exo2 image embeddings",
        default_fraud_prefix="fraud_emb_",
        default_real_prefix="real_emb_",
    ),
    "doulos": SourceSpec(
        key="doulos",
        label="Doulos image embeddings",
        default_fraud_prefix="fraud_emb_",
        default_real_prefix="real_emb_",
    ),
    "cousine": SourceSpec(
        key="cousine",
        label="Cousine image embeddings",
        default_fraud_prefix="fraud_emb_",
        default_real_prefix="real_emb_",
    ),
}

COSINE_FEATURE_TO_SOURCE: Dict[str, str] = {
    "text_cosine": "text",
    "cosine_downloads": "text",
    "sigliptext_cosine_sim": "text",
    "deja_cosine_sim": "deja",
    "cosine_deja": "deja",
    "cosine_unifont": "unifont",
    "cosine_gentium": "gentium",
    "cosine_libre": "libre",
    "cosine_exo2": "exo2",
    "cosine_doulos": "doulos",
    "cosine_cousine": "cousine",
}

SOURCE_ALIAS_COLUMNS: Dict[str, tuple[str, ...]] = {
    "text": ("text_cosine", "cosine_downloads", "sigliptext_cosine_sim"),
    "deja": ("deja_cosine_sim", "cosine_deja"),
    "unifont": ("cosine_unifont",),
    "gentium": ("cosine_gentium",),
    "libre": ("cosine_libre",),
    "exo2": ("cosine_exo2",),
    "doulos": ("cosine_doulos",),
    "cousine": ("cosine_cousine",),
}


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


@dataclass
class ModelBundle:
    path: Path
    estimator: Any
    feature_names: list[str]
    metadata: dict[str, Any]


@dataclass
class ModelSpec:
    path: Path
    feature_names: list[str]
    metadata: dict[str, Any]
    required_packages: dict[str, str]


@dataclass
class RunConfig:
    fraud_col: str
    real_col: str
    label_col: str
    positive_label: Any
    device: torch.device
    pt_batch_size: int
    data_path: Path | None
    source_data_paths: dict[str, Path | None]
    source_projector_paths: dict[str, Path | None]
    source_prefixes: dict[str, tuple[str, str]]
    default_projector: Path | None
    output_path: Path
    metrics_output_path: Path | None


def load_table(path: Path) -> pd.DataFrame:
    lowered = path.name.lower()
    if lowered.endswith(".parquet") or lowered.endswith(".pq"):
        return pd.read_parquet(path)
    if lowered.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type for {path}. Use .csv, .parquet, or .pq.")


def resolve_model_path(model_arg: str, models_dir: Path) -> Path:
    raw = Path(model_arg)
    if raw.exists():
        return raw.resolve()

    candidates = []
    names = [raw.name]
    if raw.suffix != ".joblib":
        names.append(f"{raw.name}.joblib")

    for name in names:
        candidate = models_dir / name
        if candidate.exists():
            candidates.append(candidate.resolve())

    if not candidates:
        raise FileNotFoundError(
            f"Could not find model {model_arg!r}. Pass a full path or use one of the files in {models_dir}."
        )
    return candidates[0]


def _validate_feature_names(feature_names: Any, model_path: Path) -> list[str]:
    if feature_names is None:
        raise RuntimeError(f"{model_path} is missing feature_names, so main.py cannot determine which inputs it needs.")
    if not isinstance(feature_names, list) or not all(isinstance(name, str) for name in feature_names):
        raise RuntimeError(f"{model_path} has invalid feature_names metadata.")
    return feature_names


def _extract_model_parts(obj: Any, model_path: Path) -> tuple[Any, dict[str, Any], list[str]]:
    if isinstance(obj, dict) and "model" in obj:
        estimator = obj["model"]
        metadata = {k: v for k, v in obj.items() if k != "model"}
        feature_names = metadata.get("feature_names")
    else:
        estimator = obj
        metadata = {}
        feature_names = None

    feature_names = _validate_feature_names(feature_names, model_path)
    return estimator, metadata, feature_names


@lru_cache(maxsize=1)
def _load_model_metadata_manifest() -> dict[str, Any]:
    if not DEFAULT_MODEL_METADATA_PATH.exists():
        return {}
    return json.loads(DEFAULT_MODEL_METADATA_PATH.read_text(encoding="utf-8"))


def load_model_spec(model_path: Path) -> ModelSpec:
    manifest = _load_model_metadata_manifest()
    entry = manifest.get(model_path.name)
    if entry is not None:
        metadata = dict(entry.get("metadata", {}))
        feature_names = _validate_feature_names(entry.get("feature_names") or metadata.get("feature_names"), model_path)
        required_packages = {
            str(key): str(value)
            for key, value in dict(entry.get("required_packages", {})).items()
        }
        return ModelSpec(
            path=model_path,
            feature_names=feature_names,
            metadata=metadata,
            required_packages=required_packages,
        )

    obj = joblib.load(model_path)
    _estimator, metadata, feature_names = _extract_model_parts(obj, model_path)
    return ModelSpec(
        path=model_path,
        feature_names=feature_names,
        metadata=metadata,
        required_packages={},
    )


def _model_load_runtime_error(model_path: Path, exc: Exception, spec: ModelSpec | None) -> RuntimeError:
    lines = [f"Could not load model bundle from {model_path}: {exc}"]
    required_sklearn = None if spec is None else spec.required_packages.get("scikit-learn")
    if required_sklearn is not None:
        try:
            import sklearn

            current_sklearn = sklearn.__version__
        except Exception:
            current_sklearn = "unknown"
        lines.append(
            f"This model expects scikit-learn {required_sklearn}, but the current environment has {current_sklearn}."
        )
    lines.append("Activate the project virtual environment and reinstall the pinned dependencies before scoring models.")
    lines.append("Example:")
    lines.append("python3 -m venv .venv")
    lines.append("source .venv/bin/activate")
    lines.append("pip install -r requirements.txt")
    return RuntimeError("\n".join(lines))


def load_model_bundle(model_path: Path) -> ModelBundle:
    spec: ModelSpec | None = None
    try:
        spec = load_model_spec(model_path)
    except Exception:
        spec = None

    try:
        obj = joblib.load(model_path)
    except Exception as exc:
        raise _model_load_runtime_error(model_path, exc, spec) from exc

    estimator, metadata, feature_names = _extract_model_parts(obj, model_path)

    if not hasattr(estimator, "predict"):
        raise RuntimeError(f"{model_path} does not contain an estimator with predict().")

    if spec is not None:
        metadata = dict(spec.metadata)
        feature_names = list(spec.feature_names)

    return ModelBundle(
        path=model_path,
        estimator=estimator,
        feature_names=feature_names,
        metadata=metadata,
    )


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _string_metric_support() -> tuple[bool, bool, bool]:
    have_rapidfuzz = False
    have_fuzzywuzzy = False
    have_py_lev = False

    try:
        from rapidfuzz import fuzz as _rf_fuzz  # noqa: F401
        from rapidfuzz.distance import Levenshtein as _rf_lev  # noqa: F401

        have_rapidfuzz = True
    except Exception:
        have_rapidfuzz = False

    if not have_rapidfuzz:
        try:
            from fuzzywuzzy import fuzz as _fw_fuzz  # noqa: F401

            have_fuzzywuzzy = True
        except Exception:
            have_fuzzywuzzy = False

    try:
        import Levenshtein as _py_lev  # noqa: F401

        have_py_lev = True
    except Exception:
        have_py_lev = False

    return have_rapidfuzz, have_fuzzywuzzy, have_py_lev


HAVE_RAPIDFUZZ, HAVE_FUZZYWUZZY, HAVE_PY_LEV = _string_metric_support()


def levenshtein_distance(a: str, b: str) -> int:
    if HAVE_RAPIDFUZZ:
        from rapidfuzz.distance import Levenshtein as rf_lev

        return int(rf_lev.distance(a, b))
    if HAVE_PY_LEV:
        import Levenshtein as py_lev

        return int(py_lev.distance(a, b))

    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return int(prev[-1])


def token_set_ratio(a: str, b: str) -> float:
    if HAVE_RAPIDFUZZ:
        from rapidfuzz import fuzz as rf_fuzz

        return float(rf_fuzz.token_set_ratio(a, b)) / 100.0
    if HAVE_FUZZYWUZZY:
        from fuzzywuzzy import fuzz as fw_fuzz

        return float(fw_fuzz.token_set_ratio(a, b)) / 100.0
    raise RuntimeError("Install rapidfuzz or fuzzywuzzy to compute token_set_ratio.")


def partial_ratio(a: str, b: str) -> float:
    if HAVE_RAPIDFUZZ:
        from rapidfuzz import fuzz as rf_fuzz

        return float(rf_fuzz.partial_ratio(a, b)) / 100.0
    if HAVE_FUZZYWUZZY:
        from fuzzywuzzy import fuzz as fw_fuzz

        return float(fw_fuzz.partial_ratio(a, b)) / 100.0
    raise RuntimeError("Install rapidfuzz or fuzzywuzzy to compute partial_ratio.")


def load_checkpoint_safely(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if ckpt and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            return ckpt
        for key in ("model_state", "state_dict", "model_state_dict", "model", "net"):
            candidate = ckpt.get(key)
            if isinstance(candidate, dict) and candidate and all(isinstance(v, torch.Tensor) for v in candidate.values()):
                return candidate
    raise RuntimeError(f"Unsupported checkpoint format in {type(ckpt)}")


def strip_known_prefixes(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("module.", "model.", "net.")
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        current = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if current.startswith(prefix):
                    current = current[len(prefix) :]
                    changed = True
        cleaned[current] = value
    return cleaned


def infer_projector_dims(state_dict: Mapping[str, torch.Tensor]) -> tuple[int, int, int]:
    required = ("head.0.weight", "head.0.bias", "head.2.weight", "head.2.bias")
    for key in required:
        if key not in state_dict:
            raise RuntimeError(f"Checkpoint is missing {key}; expected a saved projection head.")

    w0 = state_dict["head.0.weight"]
    w2 = state_dict["head.2.weight"]
    return int(w0.shape[1]), int(w0.shape[0]), int(w2.shape[0])


def load_projector(projector_path: Path, device: torch.device) -> tuple[ProjectionHead, int]:
    ckpt = load_checkpoint_safely(projector_path)
    state_dict = strip_known_prefixes(extract_state_dict(ckpt))
    in_dim, hidden_dim, out_dim = infer_projector_dims(state_dict)
    model = ProjectionHead(in_dim, hidden_dim, out_dim).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, in_dim


def _sorted_prefixed_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [col for col in df.columns if isinstance(col, str) and col.startswith(prefix)]
    if not cols:
        return []

    def key_fn(col: str) -> tuple[int, str]:
        suffix = col[len(prefix) :]
        if re.fullmatch(r"-?\d+", suffix):
            return int(suffix), col
        return 10**18, col

    return sorted(cols, key=key_fn)


def has_prefixed_columns(df: pd.DataFrame, prefix: str) -> bool:
    return bool(_sorted_prefixed_columns(df, prefix))


def matrix_from_prefix(df: pd.DataFrame, prefix: str) -> np.ndarray:
    cols = _sorted_prefixed_columns(df, prefix)
    if not cols:
        raise RuntimeError(f"Missing embedding columns with prefix {prefix!r}.")
    return df[cols].to_numpy(dtype=np.float32, copy=False)


@torch.inference_mode()
def projected_cosine(
    projector: ProjectionHead,
    input_dim: int,
    fraud_mat: np.ndarray,
    real_mat: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if fraud_mat.shape != real_mat.shape:
        raise RuntimeError(f"Embedding shape mismatch: {fraud_mat.shape} vs {real_mat.shape}.")
    if int(fraud_mat.shape[1]) != input_dim:
        raise RuntimeError(
            f"Embedding dim mismatch: table has {fraud_mat.shape[1]} columns, projector expects {input_dim}."
        )

    fraud_tensor = torch.from_numpy(fraud_mat.astype(np.float32, copy=False))
    real_tensor = torch.from_numpy(real_mat.astype(np.float32, copy=False))
    sims = torch.empty((len(fraud_mat),), dtype=torch.float32, device="cpu")

    for start in range(0, len(fraud_mat), max(1, batch_size)):
        end = min(start + max(1, batch_size), len(fraud_mat))
        fraud_batch = fraud_tensor[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        real_batch = real_tensor[start:end].to(device=device, dtype=torch.float32, non_blocking=True)
        fraud_proj = F.normalize(projector.encode(fraud_batch), dim=1)
        real_proj = F.normalize(projector.encode(real_batch), dim=1)
        sims[start:end] = F.cosine_similarity(fraud_proj, real_proj, dim=1).detach().cpu()

    return sims.numpy().astype(np.float32, copy=False)


def normalize_binary_labels(series: pd.Series, positive_label: Any) -> np.ndarray:
    direct = (series == positive_label).to_numpy(dtype=np.int32, copy=False)
    if direct.max(initial=0) == 1 or direct.min(initial=0) == 1:
        return direct
    return (series.astype(str) == str(positive_label)).to_numpy(dtype=np.int32, copy=False)


def feature_source_for(bundle: ModelBundle, feature_name: str) -> str:
    if feature_name in COSINE_FEATURE_TO_SOURCE:
        return COSINE_FEATURE_TO_SOURCE[feature_name]
    if feature_name == "cosine_sim":
        prefix = str(bundle.metadata.get("fraud_prefix", ""))
        return "text" if "_txt_" in prefix else "deja"
    raise RuntimeError(f"Unknown feature {feature_name!r} in {bundle.path.name}.")


def alias_candidates_for_source(source_key: str) -> list[str]:
    aliases = list(SOURCE_ALIAS_COLUMNS.get(source_key, ()))
    if source_key:
        aliases.append("cosine_sim")
    seen: list[str] = []
    for alias in aliases:
        if alias not in seen:
            seen.append(alias)
    return seen


def bundle_prefixes(bundle: ModelBundle, source_key: str) -> tuple[str, str]:
    spec = SOURCE_SPECS[source_key]
    metadata = bundle.metadata

    if source_key == "text":
        if "downloads_fraud_prefix" in metadata and "downloads_real_prefix" in metadata:
            return str(metadata["downloads_fraud_prefix"]), str(metadata["downloads_real_prefix"])
        if "fraud_prefix" in metadata and "real_prefix" in metadata and "_txt_" in str(metadata["fraud_prefix"]):
            return str(metadata["fraud_prefix"]), str(metadata["real_prefix"])
    if source_key == "deja":
        if "deja_fraud_prefix" in metadata and "deja_real_prefix" in metadata:
            return str(metadata["deja_fraud_prefix"]), str(metadata["deja_real_prefix"])
        if "fraud_prefix" in metadata and "real_prefix" in metadata and "_txt_" not in str(metadata["fraud_prefix"]):
            return str(metadata["fraud_prefix"]), str(metadata["real_prefix"])

    fraud_key = f"{source_key}_fraud_prefix"
    real_key = f"{source_key}_real_prefix"
    if fraud_key in metadata and real_key in metadata:
        return str(metadata[fraud_key]), str(metadata[real_key])

    return spec.default_fraud_prefix, spec.default_real_prefix


def unique_paths(paths: Iterable[Path | None]) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        if path is None:
            continue
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            ordered.append(resolved)
    return ordered


def load_runtime_tables(config: RunConfig) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    if config.data_path is not None:
        tables["data"] = load_table(config.data_path)
    for source_key, path in config.source_data_paths.items():
        if path is not None:
            tables[source_key] = load_table(path)
    validate_table_alignment(tables, config.fraud_col, config.real_col, config.label_col)
    return tables


def validate_table_alignment(
    tables: Mapping[str, pd.DataFrame],
    fraud_col: str,
    real_col: str,
    label_col: str,
) -> None:
    if len(tables) <= 1:
        return

    reference_key = next(iter(tables))
    reference_df = tables[reference_key]
    for key, df in tables.items():
        if key == reference_key:
            continue
        if len(df) != len(reference_df):
            raise RuntimeError(
                f"Row-count mismatch between {reference_key} ({len(reference_df)}) and {key} ({len(df)})."
            )
        for col in (fraud_col, real_col, label_col):
            if col in reference_df.columns and col in df.columns:
                left = reference_df[col].fillna("").astype(str).to_numpy()
                right = df[col].fillna("").astype(str).to_numpy()
                if not np.array_equal(left, right):
                    raise RuntimeError(f"Column {col!r} differs between {reference_key} and {key}.")


def choose_raw_table(tables: Mapping[str, pd.DataFrame], fraud_col: str, real_col: str) -> pd.DataFrame | None:
    preferred = ["data", *SOURCE_SPECS.keys()]
    for key in preferred:
        df = tables.get(key)
        if df is not None and fraud_col in df.columns and real_col in df.columns:
            return df
    for df in tables.values():
        if fraud_col in df.columns and real_col in df.columns:
            return df
    return None


def _candidate_columns(feature_name: str, source_key: str, allow_generic_cosine: bool) -> list[str]:
    candidates = [feature_name]
    for alias in SOURCE_ALIAS_COLUMNS.get(source_key, ()):
        if alias not in candidates:
            candidates.append(alias)
    if allow_generic_cosine and "cosine_sim" not in candidates:
        candidates.append("cosine_sim")
    return candidates


def _numeric_column(df: pd.DataFrame, candidates: Sequence[str]) -> np.ndarray | None:
    for candidate in candidates:
        if candidate in df.columns:
            return pd.to_numeric(df[candidate], errors="raise").to_numpy(dtype=np.float32, copy=False)
    return None


def resolve_direct_feature(
    feature_name: str,
    source_key: str,
    tables: Mapping[str, pd.DataFrame],
) -> np.ndarray | None:
    shared = tables.get("data")
    if shared is not None:
        allow_generic = feature_name == "cosine_sim"
        direct = _numeric_column(shared, _candidate_columns(feature_name, source_key, allow_generic))
        if direct is not None:
            return direct

    source_df = tables.get(source_key)
    if source_df is not None:
        direct = _numeric_column(source_df, _candidate_columns(feature_name, source_key, True))
        if direct is not None:
            return direct
    return None


def compute_string_features(raw_df: pd.DataFrame, fraud_col: str, real_col: str) -> dict[str, np.ndarray]:
    if raw_df is None:
        raise RuntimeError("This model needs fraudulent/real name columns, but no table with those columns was provided.")

    fraud_names = raw_df[fraud_col].fillna("").astype(str).tolist()
    real_names = raw_df[real_col].fillna("").astype(str).tolist()
    n_rows = len(raw_df)
    lev = np.empty((n_rows,), dtype=np.int32)
    token = np.empty((n_rows,), dtype=np.float32)
    partial = np.empty((n_rows,), dtype=np.float32)

    for idx, (fraud_name, real_name) in enumerate(zip(fraud_names, real_names)):
        lev[idx] = levenshtein_distance(fraud_name, real_name)
        token[idx] = token_set_ratio(fraud_name, real_name)
        partial[idx] = partial_ratio(fraud_name, real_name)

    return {
        "token_set_ratio": token,
        "levenshtein_distance_score": (-lev).astype(np.float32),
        "partial_ratio": partial,
    }


def resolve_projector_path(config: RunConfig, source_key: str) -> Path | None:
    return config.source_projector_paths.get(source_key) or config.default_projector


def resolve_embedding_table(
    tables: Mapping[str, pd.DataFrame],
    source_key: str,
    fraud_prefix: str,
    real_prefix: str,
) -> pd.DataFrame | None:
    source_df = tables.get(source_key)
    if source_df is not None and has_prefixed_columns(source_df, fraud_prefix) and has_prefixed_columns(source_df, real_prefix):
        return source_df

    shared = tables.get("data")
    if shared is not None and has_prefixed_columns(shared, fraud_prefix) and has_prefixed_columns(shared, real_prefix):
        return shared

    return None


def build_feature_matrix(
    bundle: ModelBundle,
    config: RunConfig,
    tables: Mapping[str, pd.DataFrame],
) -> tuple[np.ndarray, pd.DataFrame | None]:
    raw_df = choose_raw_table(tables, config.fraud_col, config.real_col)
    string_feature_cache: dict[str, np.ndarray] | None = None
    projector_cache: dict[Path, tuple[ProjectionHead, int]] = {}
    feature_values: dict[str, np.ndarray] = {}

    for feature_name in bundle.feature_names:
        if feature_name in feature_values:
            continue

        if feature_name in STRING_FEATURES:
            if string_feature_cache is None:
                string_feature_cache = compute_string_features(raw_df, config.fraud_col, config.real_col)
            feature_values[feature_name] = string_feature_cache[feature_name]
            continue

        source_key = feature_source_for(bundle, feature_name)

        direct = resolve_direct_feature(feature_name, source_key, tables)
        if direct is not None:
            feature_values[feature_name] = direct
            continue

        fraud_prefix, real_prefix = config.source_prefixes[source_key]
        embedding_df = resolve_embedding_table(tables, source_key, fraud_prefix, real_prefix)
        projector_path = resolve_projector_path(config, source_key)

        if embedding_df is None or projector_path is None:
            raise RuntimeError(
                "Missing input for feature "
                f"{feature_name!r}. Supply a precomputed column in --data/--{source_key}-data, "
                f"or provide embedding columns ({fraud_prefix}*, {real_prefix}*) with --{source_key}-projector."
            )

        if projector_path not in projector_cache:
            projector_cache[projector_path] = load_projector(projector_path, config.device)
        projector, input_dim = projector_cache[projector_path]

        fraud_mat = matrix_from_prefix(embedding_df, fraud_prefix)
        real_mat = matrix_from_prefix(embedding_df, real_prefix)
        feature_values[feature_name] = projected_cosine(
            projector=projector,
            input_dim=input_dim,
            fraud_mat=fraud_mat,
            real_mat=real_mat,
            device=config.device,
            batch_size=config.pt_batch_size,
        )

    feature_matrix = np.column_stack([feature_values[name] for name in bundle.feature_names]).astype(
        np.float32,
        copy=False,
    )
    return feature_matrix, raw_df


def compute_metrics(
    raw_df: pd.DataFrame | None,
    config: RunConfig,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
) -> dict[str, float | int]:
    metrics: dict[str, float | int] = {
        "rows": int(len(predictions)),
        "predicted_positive_count": int(predictions.sum()),
    }

    if probabilities is not None:
        metrics["mean_positive_probability"] = float(np.mean(probabilities))

    if raw_df is None or config.label_col not in raw_df.columns:
        return metrics

    y_true = normalize_binary_labels(raw_df[config.label_col], config.positive_label)
    metrics["accuracy"] = float(accuracy_score(y_true, predictions))
    metrics["precision"] = float(precision_score(y_true, predictions, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, predictions, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, predictions, zero_division=0))

    if probabilities is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))

    return metrics


def prediction_frame(
    raw_df: pd.DataFrame | None,
    config: RunConfig,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
) -> pd.DataFrame:
    if raw_df is None:
        out = pd.DataFrame({"row_index": np.arange(len(predictions), dtype=np.int32)})
    else:
        keep_cols = [col for col in (config.fraud_col, config.real_col, config.label_col) if col in raw_df.columns]
        out = raw_df[keep_cols].copy()
        out.insert(0, "row_index", np.arange(len(out), dtype=np.int32))

    out["pred_label"] = predictions.astype(np.int32)
    if probabilities is not None:
        out["pred_positive_probability"] = probabilities.astype(np.float32)
    return out


def output_path_for(model_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg).resolve()
    return (REPO_ROOT / "outputs" / f"{model_path.stem}_predictions.csv").resolve()


def metrics_output_path_for(model_path: Path, metrics_output_arg: str | None) -> Path | None:
    if metrics_output_arg is None:
        return None
    if metrics_output_arg:
        return Path(metrics_output_arg).resolve()
    return (REPO_ROOT / "outputs" / f"{model_path.stem}_metrics.json").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the saved business-name matching models.",
    )

    action_group = parser.add_mutually_exclusive_group(required=True)
    action_group.add_argument("--list-models", action="store_true", help="List the available .joblib models.")
    action_group.add_argument("--describe-model", help="Show the inputs required by one model.")
    action_group.add_argument("--model-path", help="Run predictions with one model.")

    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help="Directory containing the saved .joblib models.",
    )

    parser.add_argument("--data", help="CSV or Parquet file with raw columns and/or precomputed feature columns.")
    parser.add_argument(
        "--default-projector",
        help="Default .pt projector to use when a source-specific projector is not provided.",
    )
    parser.add_argument("--fraud-col", default=None, help="Override the fraudulent name column.")
    parser.add_argument("--real-col", default=None, help="Override the real name column.")
    parser.add_argument("--label-col", default=None, help="Override the label column.")
    parser.add_argument("--positive-label", default=None, help="Override the positive label value.")
    parser.add_argument("--device", default=None, help="cuda, mps, or cpu. Defaults to auto.")
    parser.add_argument("--pt-batch-size", type=int, default=8192, help="Batch size for projector-based cosine scoring.")
    parser.add_argument("--output", default=None, help="Prediction output path. Defaults to outputs/<model>_predictions.csv.")
    parser.add_argument("--metrics-output", default=None, help="Optional path for metrics JSON.")

    for source_key, spec in SOURCE_SPECS.items():
        parser.add_argument(
            f"--{source_key}-data",
            dest=f"{source_key}_data",
            default=None,
            help=f"CSV or Parquet for {spec.label}. It can include named feature columns, cosine_sim, or embedding columns.",
        )
        parser.add_argument(
            f"--{source_key}-projector",
            dest=f"{source_key}_projector",
            default=None,
            help=f".pt projector used to convert {spec.label} into cosine features.",
        )
        parser.add_argument(
            f"--{source_key}-fraud-prefix",
            dest=f"{source_key}_fraud_prefix",
            default=None,
            help=f"Override the fraudulent embedding prefix for {spec.label}.",
        )
        parser.add_argument(
            f"--{source_key}-real-prefix",
            dest=f"{source_key}_real_prefix",
            default=None,
            help=f"Override the real embedding prefix for {spec.label}.",
        )

    return parser.parse_args()


def build_run_config(args: argparse.Namespace, bundle: ModelBundle) -> RunConfig:
    fraud_col = args.fraud_col or bundle.metadata.get("fraud_col") or "fraudulent_name"
    real_col = args.real_col or bundle.metadata.get("real_col") or "real_name"
    label_col = args.label_col or bundle.metadata.get("label_col") or "label"
    positive_label = args.positive_label if args.positive_label is not None else bundle.metadata.get("positive_label", 1)

    source_data_paths = {
        source_key: (Path(getattr(args, f"{source_key}_data")).resolve() if getattr(args, f"{source_key}_data") else None)
        for source_key in SOURCE_SPECS
    }
    source_projector_paths = {
        source_key: (
            Path(getattr(args, f"{source_key}_projector")).resolve()
            if getattr(args, f"{source_key}_projector")
            else None
        )
        for source_key in SOURCE_SPECS
    }
    source_prefixes = {}
    for source_key in SOURCE_SPECS:
        default_fraud_prefix, default_real_prefix = bundle_prefixes(bundle, source_key)
        fraud_prefix = getattr(args, f"{source_key}_fraud_prefix") or default_fraud_prefix
        real_prefix = getattr(args, f"{source_key}_real_prefix") or default_real_prefix
        source_prefixes[source_key] = (fraud_prefix, real_prefix)

    return RunConfig(
        fraud_col=str(fraud_col),
        real_col=str(real_col),
        label_col=str(label_col),
        positive_label=positive_label,
        device=choose_device(args.device),
        pt_batch_size=max(1, int(args.pt_batch_size)),
        data_path=Path(args.data).resolve() if args.data else None,
        source_data_paths=source_data_paths,
        source_projector_paths=source_projector_paths,
        source_prefixes=source_prefixes,
        default_projector=Path(args.default_projector).resolve() if args.default_projector else None,
        output_path=output_path_for(bundle.path, args.output),
        metrics_output_path=metrics_output_path_for(bundle.path, args.metrics_output),
    )


def required_sources(bundle: ModelBundle | ModelSpec) -> dict[str, list[str]]:
    needs: dict[str, list[str]] = {}
    for feature_name in bundle.feature_names:
        if feature_name in STRING_FEATURES:
            needs.setdefault("raw", []).append(feature_name)
            continue
        source_key = feature_source_for(bundle, feature_name)
        needs.setdefault(source_key, []).append(feature_name)
    return needs


def describe_model(bundle: ModelBundle | ModelSpec) -> str:
    needs = required_sources(bundle)
    fraud_col = bundle.metadata.get("fraud_col", "fraudulent_name")
    real_col = bundle.metadata.get("real_col", "real_name")
    label_col = bundle.metadata.get("label_col", "label")
    estimator_name = type(bundle.estimator).__name__ if isinstance(bundle, ModelBundle) else "metadata-only"
    lines = [
        f"Model: {bundle.path}",
        f"Estimator: {estimator_name}",
        f"Features: {', '.join(bundle.feature_names)}",
        f"Default columns: fraud_col={fraud_col!r}, real_col={real_col!r}, label_col={label_col!r}",
    ]

    if "raw" in needs:
        lines.append(
            f"Raw text required for {', '.join(needs['raw'])}: columns {fraud_col!r} and {real_col!r}."
        )

    for source_key, features in needs.items():
        if source_key == "raw":
            continue
        spec = SOURCE_SPECS[source_key]
        fraud_prefix, real_prefix = bundle_prefixes(bundle, source_key)
        aliases = ", ".join(alias_candidates_for_source(source_key))
        lines.append(
            f"{spec.label} for {', '.join(features)}: provide columns named {aliases} "
            f"or pass a table with embedding prefixes {fraud_prefix!r}/{real_prefix!r} together with --{source_key}-projector."
        )

    return "\n".join(lines)


def print_model_listing(models_dir: Path) -> None:
    model_paths = sorted(models_dir.glob("*.joblib"))
    if not model_paths:
        raise RuntimeError(f"No .joblib files found in {models_dir}.")

    for model_path in model_paths:
        spec = load_model_spec(model_path)
        features = ", ".join(spec.feature_names)
        print(f"{model_path.name}: {features}")


def run_model(bundle: ModelBundle, config: RunConfig) -> None:
    tables = load_runtime_tables(config)
    feature_matrix, raw_df = build_feature_matrix(bundle, config, tables)

    predictions = bundle.estimator.predict(feature_matrix).astype(np.int32, copy=False)
    probabilities = None
    if hasattr(bundle.estimator, "predict_proba"):
        probabilities = bundle.estimator.predict_proba(feature_matrix)[:, 1].astype(np.float32, copy=False)

    metrics = compute_metrics(raw_df, config, predictions, probabilities)
    output_df = prediction_frame(raw_df, config, predictions, probabilities)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(config.output_path, index=False)

    if config.metrics_output_path is not None:
        config.metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        config.metrics_output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Model: {bundle.path.name}")
    print(f"Rows scored: {len(output_df)}")
    print(f"Features used: {', '.join(bundle.feature_names)}")
    print(f"Predictions saved to: {config.output_path}")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    if config.metrics_output_path is not None:
        print(f"Metrics saved to: {config.metrics_output_path}")


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir).resolve()

    if args.list_models:
        print_model_listing(models_dir)
        return

    model_arg = args.describe_model or args.model_path
    assert model_arg is not None
    model_path = resolve_model_path(model_arg, models_dir)

    if args.describe_model:
        print(describe_model(load_model_spec(model_path)))
        return

    bundle = load_model_bundle(model_path)
    config = build_run_config(args, bundle)
    run_model(bundle, config)


if __name__ == "__main__":
    main()
