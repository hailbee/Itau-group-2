from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import threading
import time
from typing import Any
from uuid import uuid4

from flask import Flask, jsonify, render_template, request

from domain_matcher import DEFAULT_CHUNK_SIZE, DEFAULT_MODEL_PATH, DomainMatcher, SearchCancelled, SearchReport


app = Flask(__name__)

JOB_LOCK = threading.Lock()
SEARCH_JOBS: dict[str, dict[str, Any]] = {}
MODEL_OPTIONS = (
    {
        "key": "total_5f_model.joblib",
        "label": "5 Font Model",
        "description": "Text metrics, text embeddings, and five font embeddings.",
    },
    {
        "key": "total_1f_model.joblib",
        "label": "1 Font Model",
        "description": "Text metrics, text embeddings, and the Deja font embedding.",
    },
    {
        "key": "metrics_model.joblib",
        "label": "Text Metrics",
        "description": "String-only scoring from token, partial, and edit-distance metrics.",
    },
    {
        "key": "sigliptext_model.joblib",
        "label": "Text Embeddings",
        "description": "SigLIP text embedding cosine similarity only.",
    },
)
MODEL_OPTIONS_BY_KEY = {option["key"]: option for option in MODEL_OPTIONS}
DEFAULT_MODEL_KEY = DEFAULT_MODEL_PATH.name


def _job_cancel_requested(job_id: str) -> bool:
    with JOB_LOCK:
        job = SEARCH_JOBS.get(job_id)
        return bool(job and job.get("cancel_requested"))


def _model_option(model_key: str) -> dict[str, str]:
    option = MODEL_OPTIONS_BY_KEY.get(model_key)
    if option is None:
        raise ValueError(f"Unknown search model: {model_key}")
    return option


@lru_cache(maxsize=1)
def get_matcher(model_key: str = DEFAULT_MODEL_KEY) -> DomainMatcher:
    model_path = (DEFAULT_MODEL_PATH.parent / model_key).resolve()
    return DomainMatcher(model_path=model_path)


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
        "model": DEFAULT_MODEL_KEY,
        "top_k": 25,
        "min_confidence": 0.5,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "max_rows": "",
    }


def _model_field(values: dict[str, Any]) -> str:
    raw = str(values.get("model", "") or "").strip()
    model_key = raw or DEFAULT_MODEL_KEY
    _model_option(model_key)
    return model_key


def _model_label(model_key: str) -> str:
    return _model_option(model_key)["label"]


def _serialize_report(report: SearchReport, model_key: str) -> dict[str, Any]:
    payload = asdict(report)
    payload["selected_model_key"] = model_key
    payload["selected_model_label"] = _model_label(model_key)
    return payload


def _update_job(job_id: str, **updates: Any) -> None:
    with JOB_LOCK:
        if job_id in SEARCH_JOBS:
            SEARCH_JOBS[job_id].update(updates)


def _run_search_job(job_id: str, form_values: dict[str, Any]) -> None:
    started_at = time.time()
    model_key = _model_field(form_values)
    model_label = _model_label(model_key)
    try:
        _update_job(
            job_id,
            status="starting",
            progress={
                "status": "starting",
                "stage": "candidate_source",
                "stage_detail": f"Starting worker and loading {model_label}.",
                "scanned_rows": 0,
                "total_rows_target": 0,
                "total_predicted_positive": 0,
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
                    "total_predicted_positive": 0,
                    "duration_seconds": 0.0,
                    "feature_mode": "initializing matcher",
                    "selected_model_label": model_label,
                }
            )
        matcher = get_matcher(model_key)
        if _job_cancel_requested(job_id):
            raise SearchCancelled(
                {
                    "status": "cancelled",
                    "stage": "prepare_query",
                    "stage_detail": "Search stopped before query preparation began.",
                    "scanned_rows": 0,
                    "total_rows_target": 0,
                    "total_predicted_positive": 0,
                    "duration_seconds": time.time() - started_at,
                    "feature_mode": matcher._feature_mode(),
                    "selected_model_label": model_label,
                }
            )

        def on_progress(progress: dict[str, Any]) -> None:
            cancel_requested = _job_cancel_requested(job_id)
            progress_status = "cancelling" if cancel_requested and progress.get("status") == "running" else progress.get("status", "running")
            progress_payload = dict(progress)
            progress_payload["status"] = progress_status
            progress_payload["selected_model_label"] = model_label
            _update_job(
                job_id,
                status="cancelling" if cancel_requested else "running",
                progress=progress_payload,
                updated_at=time.time(),
            )

        report = matcher.search(
            form_values["query"],
            top_k=max(1, _int_field("top_k", 25, form_values) or 25),
            min_confidence=max(0.0, min(1.0, _float_field("min_confidence", 0.5, form_values))),
            chunk_size=max(1, _int_field("chunk_size", DEFAULT_CHUNK_SIZE, form_values) or DEFAULT_CHUNK_SIZE),
            max_rows=_int_field("max_rows", None, form_values),
            progress_callback=on_progress,
            cancel_callback=lambda: _job_cancel_requested(job_id),
        )
        _update_job(
            job_id,
            status="completed",
            progress={
                "status": "completed",
                "stage": "completed",
                "stage_detail": "Search complete. Final ranking is ready.",
                "query": report.query,
                "normalized_query": report.normalized_query,
                "scanned_rows": report.scanned_rows,
                "total_rows_target": report.total_rows_target,
                "total_predicted_positive": report.total_predicted_positive,
                "duration_seconds": report.duration_seconds,
                "feature_mode": report.feature_mode,
                "selected_model_label": model_label,
            },
            report=_serialize_report(report, model_key),
            completed_at=time.time(),
            duration_seconds=time.time() - started_at,
        )
    except SearchCancelled as exc:
        progress = dict(exc.progress or {})
        progress.setdefault("status", "cancelled")
        progress.setdefault("stage", "scan_candidates")
        progress.setdefault("stage_detail", "Search stopped by user.")
        progress.setdefault("selected_model_label", model_label)
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
    return render_template(
        "index.html",
        report=None,
        error=None,
        form_values=_empty_form_values(),
        model_options=MODEL_OPTIONS,
        selected_model_label=_model_label(DEFAULT_MODEL_KEY),
    )


@app.post("/api/search")
def start_search():
    try:
        payload = request.get_json(silent=True) or request.form.to_dict()
        form_values = {
            "query": payload.get("query", ""),
            "model": _model_field(payload),
            "top_k": payload.get("top_k", "25"),
            "min_confidence": payload.get("min_confidence", "0.5"),
            "chunk_size": payload.get("chunk_size", str(DEFAULT_CHUNK_SIZE)),
            "max_rows": payload.get("max_rows", ""),
        }
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
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
                "total_predicted_positive": 0,
                "duration_seconds": 0.0,
                "feature_mode": "pending",
                "selected_model_label": _model_label(form_values["model"]),
            },
            "error": None,
            "report": None,
            "created_at": time.time(),
            "updated_at": time.time(),
        }

    worker = threading.Thread(target=_run_search_job, args=(job_id, form_values), daemon=True)
    worker.start()
    return jsonify({"job_id": job_id, "status": "queued"})


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
        "model": request.form.get("model", DEFAULT_MODEL_KEY),
        "top_k": request.form.get("top_k", "25"),
        "min_confidence": request.form.get("min_confidence", "0.5"),
        "chunk_size": request.form.get("chunk_size", str(DEFAULT_CHUNK_SIZE)),
        "max_rows": request.form.get("max_rows", ""),
    }
    try:
        model_key = _model_field(form_values)
        form_values["model"] = model_key
        matcher = get_matcher(model_key)
        report = matcher.search(
            form_values["query"],
            top_k=max(1, _int_field("top_k", 25, form_values) or 25),
            min_confidence=max(0.0, min(1.0, _float_field("min_confidence", 0.5, form_values))),
            chunk_size=max(1, _int_field("chunk_size", DEFAULT_CHUNK_SIZE, form_values) or DEFAULT_CHUNK_SIZE),
            max_rows=_int_field("max_rows", None, form_values),
        )
        return render_template(
            "index.html",
            report=report,
            error=None,
            form_values=form_values,
            model_options=MODEL_OPTIONS,
            selected_model_label=_model_label(model_key),
        )
    except Exception as exc:
        model_key = form_values.get("model", DEFAULT_MODEL_KEY)
        return render_template(
            "index.html",
            report=None,
            error=str(exc),
            form_values=form_values,
            model_options=MODEL_OPTIONS,
            selected_model_label=_model_label(model_key) if model_key in MODEL_OPTIONS_BY_KEY else _model_label(DEFAULT_MODEL_KEY),
        ), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
