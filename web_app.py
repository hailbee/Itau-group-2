from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import threading
import time
from typing import Any
from uuid import uuid4

import pandas as pd
from flask import Flask, jsonify, render_template, request

from domain_matcher import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL_PATH,
    DomainMatcher,
    PairwiseComparisonReport,
    SearchCancelled,
    SearchHit,
    SearchReport,
    canonicalize_domain_host,
    normalize_domain_string,
)


app = Flask(__name__)

JOB_LOCK = threading.Lock()
SEARCH_JOBS: dict[str, dict[str, Any]] = {}
DEFAULT_MODEL_KEY = DEFAULT_MODEL_PATH.name

RESULT_CONTAINER_CANDIDATES = ("results", "matches", "rows")


def _job_cancel_requested(job_id: str) -> bool:
    with JOB_LOCK:
        job = SEARCH_JOBS.get(job_id)
        return bool(job and job.get("cancel_requested"))


@lru_cache(maxsize=1)
def get_matcher(model_key: str = DEFAULT_MODEL_KEY) -> DomainMatcher:
    model_path = (DEFAULT_MODEL_PATH.parent / model_key).resolve()
    matcher = DomainMatcher(model_path=model_path)
    if not matcher.font_feature_names:
        raise RuntimeError(
            "The default 5-font web app model is missing the font cosine features required for ranking."
        )
    return matcher


def _int_field(name: str, default: int | None, values: dict[str, Any]) -> int | None:
    raw = str(values.get(name, "") or "")
    if not raw.strip():
        return default
    return int(raw)


def _float_field(name: str, default: float, values: dict[str, Any]) -> float:
    raw = str(values.get(name, "") or "")
    if not raw.strip():
        return default
    return float(raw)


def _empty_form_values() -> dict[str, Any]:
    return {
        "query": "",
        "top_k": 25,
        "min_mean_font_cosine": 0.5,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "max_rows": "",
    }


def _empty_compare_form_values() -> dict[str, Any]:
    return {
        "left_query": "",
        "right_query": "",
        "threshold": 0.5,
    }


def _serialize_report(report: SearchReport) -> dict[str, Any]:
    return asdict(report)


def _serialize_pairwise_report(report: PairwiseComparisonReport) -> dict[str, Any]:
    return asdict(report)


def _update_job(job_id: str, **updates: Any) -> None:
    with JOB_LOCK:
        if job_id in SEARCH_JOBS:
            SEARCH_JOBS[job_id].update(updates)


@lru_cache(maxsize=8)
def _load_exact_domain_index(csv_path_str: str) -> dict[str, str]:
    csv_path = Path(csv_path_str)
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path, usecols=["domain"])
    df["domain"] = df["domain"].fillna("").astype(str)

    mapping: dict[str, str] = {}
    for domain in df["domain"].tolist():
        canonical_domain = canonicalize_domain_host(domain)
        if canonical_domain and canonical_domain not in mapping:
            mapping[canonical_domain] = domain
    return mapping


def _safe_get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _safe_set(obj: Any, name: str, value: Any) -> bool:
    if isinstance(obj, dict):
        obj[name] = value
        return True
    try:
        setattr(obj, name, value)
        return True
    except Exception:
        try:
            object.__setattr__(obj, name, value)
            return True
        except Exception:
            return False


def _normalize_text_value(value: Any) -> str:
    text = "" if value is None else str(value)
    try:
        return normalize_domain_string(text)
    except Exception:
        return text.strip().lower()


def _render_index(
    *,
    report: SearchReport | None = None,
    search_error: str | None = None,
    form_values: dict[str, Any] | None = None,
    compare_report: PairwiseComparisonReport | None = None,
    compare_error: str | None = None,
    compare_form_values: dict[str, Any] | None = None,
    active_tab: str = "search",
    status_code: int = 200,
):
    return (
        render_template(
            "index.html",
            report=report,
            error=search_error,
            form_values=form_values or _empty_form_values(),
            compare_report=compare_report,
            compare_error=compare_error,
            compare_form_values=compare_form_values or _empty_compare_form_values(),
            active_tab=active_tab,
        ),
        status_code,
    )


def _result_lists(report: Any) -> list[list[Any]]:
    lists: list[list[Any]] = []
    seen_ids: set[int] = set()
    for field_name in ("matches", "top_candidates", *RESULT_CONTAINER_CANDIDATES):
        value = _safe_get(report, field_name, None)
        if isinstance(value, list) and id(value) not in seen_ids:
            lists.append(value)
            seen_ids.add(id(value))
    return lists


def _result_canonical_domain(result: Any) -> str:
    return canonicalize_domain_host(_safe_get(result, "domain", ""))


def _force_exact_result_scores(
    result: Any,
    exact_domain: str,
    normalized_query: str,
    font_feature_names: list[str],
) -> None:
    _safe_set(result, "exact_match", True)
    _safe_set(result, "domain", exact_domain)
    _safe_set(result, "normalized_domain", normalized_query)
    _safe_set(result, "mean_font_cosine", 1.0)
    _safe_set(result, "font_cosines", {feature_name: 1.0 for feature_name in font_feature_names})


def _synthetic_exact_result(
    exact_domain: str,
    normalized_query: str,
    font_feature_names: list[str],
) -> SearchHit:
    return SearchHit(
        domain=exact_domain,
        mean_font_cosine=1.0,
        font_cosines={feature_name: 1.0 for feature_name in font_feature_names},
        exact_match=True,
        normalized_domain=normalized_query,
    )


def _apply_exact_result_to_list(
    results: list[Any],
    *,
    exact_domain: str,
    normalized_query: str,
    font_feature_names: list[str],
    top_k: int,
) -> None:
    exact_host = canonicalize_domain_host(exact_domain)
    exact_pos = None
    for idx, result in enumerate(results):
        if _result_canonical_domain(result) == exact_host:
            _force_exact_result_scores(result, exact_domain, normalized_query, font_feature_names)
            exact_pos = idx
            break

    if exact_pos is None:
        results.insert(
            0,
            _synthetic_exact_result(
                exact_domain,
                normalized_query,
                font_feature_names,
            ),
        )
    elif exact_pos != 0:
        results.insert(0, results.pop(exact_pos))

    if top_k > 0 and len(results) > top_k:
        del results[top_k:]


def _apply_exact_benign_hit_override(
    report: SearchReport,
    matcher: DomainMatcher,
    query: str,
    *,
    top_k: int,
) -> SearchReport:
    normalized_query = str(_safe_get(report, "normalized_query", "") or "")
    if not normalized_query:
        normalized_query = _normalize_text_value(query)
        _safe_set(report, "normalized_query", normalized_query)

    index_path = matcher.dataset_path
    precomputed_store = getattr(matcher, "precomputed_store", None)
    if precomputed_store is not None:
        store_dir = getattr(precomputed_store, "store_dir", None)
        if store_dir is not None:
            candidate_path = Path(store_dir) / "domains.csv"
            if candidate_path.exists():
                index_path = candidate_path

    exact_query = canonicalize_domain_host(query)
    if not exact_query:
        return report

    exact_index = _load_exact_domain_index(str(index_path))
    exact_domain = exact_index.get(exact_query)
    if exact_domain is None:
        return report

    for results in _result_lists(report):
        _apply_exact_result_to_list(
            results,
            exact_domain=exact_domain,
            normalized_query=normalized_query,
            font_feature_names=matcher.font_feature_names,
            top_k=top_k,
        )

    return report


def _search_with_exact_override(matcher: DomainMatcher, form_values: dict[str, Any]) -> SearchReport:
    top_k = max(1, _int_field("top_k", 25, form_values) or 25)
    report = matcher.search(
        form_values["query"],
        top_k=top_k,
        min_mean_font_cosine=max(0.0, min(1.0, _float_field("min_mean_font_cosine", 0.5, form_values))),
        chunk_size=max(1, _int_field("chunk_size", DEFAULT_CHUNK_SIZE, form_values) or DEFAULT_CHUNK_SIZE),
        max_rows=_int_field("max_rows", None, form_values),
        progress_callback=form_values.get("progress_callback"),
        cancel_callback=form_values.get("cancel_callback"),
    )
    return _apply_exact_benign_hit_override(
        report,
        matcher,
        form_values["query"],
        top_k=top_k,
    )


def _compare_form_values(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "left_query": payload.get("left_query", ""),
        "right_query": payload.get("right_query", ""),
        "threshold": payload.get("threshold", "0.5"),
    }


def _compare_pair(matcher: DomainMatcher, form_values: dict[str, Any]) -> PairwiseComparisonReport:
    return matcher.compare_pair(
        form_values["left_query"],
        form_values["right_query"],
        threshold=max(0.0, min(1.0, _float_field("threshold", 0.5, form_values))),
    )


def _run_search_job(job_id: str, form_values: dict[str, Any]) -> None:
    started_at = time.time()
    try:
        _update_job(
            job_id,
            status="starting",
            progress={
                "status": "starting",
                "stage": "candidate_source",
                "stage_detail": "Starting worker and loading the 5-font model.",
                "scanned_rows": 0,
                "total_rows_target": 0,
                "total_threshold_hits": 0,
                "duration_seconds": 0.0,
                "feature_mode": "initializing matcher",
            },
            updated_at=time.time(),
        )
        if _job_cancel_requested(job_id):
            raise SearchCancelled(
                {
                    "status": "cancelled",
                    "stage": "candidate_source",
                    "stage_detail": "Search stopped before the matcher finished starting.",
                    "scanned_rows": 0,
                    "total_rows_target": 0,
                    "total_threshold_hits": 0,
                    "duration_seconds": 0.0,
                    "feature_mode": "initializing matcher",
                }
            )
        matcher = get_matcher()
        if _job_cancel_requested(job_id):
            raise SearchCancelled(
                {
                    "status": "cancelled",
                    "stage": "prepare_query",
                    "stage_detail": "Search stopped before query preparation began.",
                    "scanned_rows": 0,
                    "total_rows_target": 0,
                    "total_threshold_hits": 0,
                    "duration_seconds": time.time() - started_at,
                    "feature_mode": matcher._feature_mode(),
                }
            )

        def on_progress(progress: dict[str, Any]) -> None:
            cancel_requested = _job_cancel_requested(job_id)
            progress_status = "cancelling" if cancel_requested and progress.get("status") == "running" else progress.get("status", "running")
            progress_payload = dict(progress)
            progress_payload["status"] = progress_status
            _update_job(
                job_id,
                status="cancelling" if cancel_requested else "running",
                progress=progress_payload,
                updated_at=time.time(),
            )

        search_form_values = dict(form_values)
        search_form_values["progress_callback"] = on_progress
        search_form_values["cancel_callback"] = lambda: _job_cancel_requested(job_id)
        report = _search_with_exact_override(matcher, search_form_values)
        _update_job(
            job_id,
            status="completed",
            progress={
                "status": "completed",
                "stage": "completed",
                "stage_detail": "Search complete. Final mean-font-cosine ranking is ready.",
                "query": report.query,
                "normalized_query": report.normalized_query,
                "scanned_rows": report.scanned_rows,
                "total_rows_target": report.total_rows_target,
                "total_threshold_hits": report.total_threshold_hits,
                "duration_seconds": report.duration_seconds,
                "feature_mode": report.feature_mode,
            },
            report=_serialize_report(report),
            completed_at=time.time(),
            duration_seconds=time.time() - started_at,
        )
    except SearchCancelled as exc:
        progress = dict(exc.progress or {})
        progress.setdefault("status", "cancelled")
        progress.setdefault("stage", "scan_candidates")
        progress.setdefault("stage_detail", "Search stopped by user.")
        progress["duration_seconds"] = time.time() - started_at
        _update_job(
            job_id,
            status="cancelled",
            error=None,
            progress=progress,
            completed_at=time.time(),
            duration_seconds=time.time() - started_at,
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            completed_at=time.time(),
            duration_seconds=time.time() - started_at,
        )


@app.get("/")
def index():
    return _render_index()


@app.post("/api/search")
def start_search():
    payload = request.get_json(silent=True) or request.form.to_dict()
    form_values = {
        "query": payload.get("query", ""),
        "top_k": payload.get("top_k", "25"),
        "min_mean_font_cosine": payload.get("min_mean_font_cosine", "0.5"),
        "chunk_size": payload.get("chunk_size", str(DEFAULT_CHUNK_SIZE)),
        "max_rows": payload.get("max_rows", ""),
    }
    job_id = uuid4().hex
    with JOB_LOCK:
        SEARCH_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "cancel_requested": False,
            "form_values": form_values,
            "progress": {
                "status": "queued",
                "stage": "candidate_source",
                "stage_detail": "Queued and waiting to start.",
                "scanned_rows": 0,
                "total_rows_target": 0,
                "total_threshold_hits": 0,
                "duration_seconds": 0.0,
                "feature_mode": "pending",
            },
            "error": None,
            "report": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

    worker = threading.Thread(target=_run_search_job, args=(job_id, form_values), daemon=True)
    worker.start()
    return jsonify({"job_id": job_id, "status": "queued"})


@app.post("/api/compare")
def compare_api():
    payload = request.get_json(silent=True) or request.form.to_dict()
    form_values = _compare_form_values(payload)
    try:
        matcher = get_matcher()
        report = _compare_pair(matcher, form_values)
        return jsonify({"report": _serialize_pairwise_report(report)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/api/search/<job_id>")
def search_status(job_id: str):
    with JOB_LOCK:
        job = SEARCH_JOBS.get(job_id)
    if job is None:
        return jsonify({"error": f"Unknown job id: {job_id}"}), 404
    return jsonify(job)


@app.post("/api/search/<job_id>/cancel")
def cancel_search(job_id: str):
    with JOB_LOCK:
        job = SEARCH_JOBS.get(job_id)
        if job is None:
            return jsonify({"error": f"Unknown job id: {job_id}"}), 404

        if job["status"] in {"queued", "starting", "running", "cancelling"}:
            job["cancel_requested"] = True
            progress = dict(job.get("progress") or {})
            progress["status"] = "cancelling"
            progress["stage_detail"] = "Stop requested. Finishing the current step."
            job["progress"] = progress
            job["status"] = "cancelling"
            job["updated_at"] = time.time()

    return jsonify(job)


@app.post("/search")
def search():
    form_values = {
        "query": request.form.get("query", ""),
        "top_k": request.form.get("top_k", "25"),
        "min_mean_font_cosine": request.form.get("min_mean_font_cosine", "0.5"),
        "chunk_size": request.form.get("chunk_size", str(DEFAULT_CHUNK_SIZE)),
        "max_rows": request.form.get("max_rows", ""),
    }
    try:
        matcher = get_matcher()
        report = _search_with_exact_override(matcher, form_values)
        return _render_index(
            report=report,
            form_values=form_values,
            active_tab="search",
        )
    except Exception as exc:
        return _render_index(
            search_error=str(exc),
            form_values=form_values,
            active_tab="search",
            status_code=400,
        )


@app.post("/compare")
def compare():
    form_values = _compare_form_values(request.form)
    try:
        matcher = get_matcher()
        report = _compare_pair(matcher, form_values)
        return _render_index(
            compare_report=report,
            compare_form_values=form_values,
            active_tab="compare",
        )
    except Exception as exc:
        return _render_index(
            compare_error=str(exc),
            compare_form_values=form_values,
            active_tab="compare",
            status_code=400,
        )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
