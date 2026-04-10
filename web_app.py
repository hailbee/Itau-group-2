from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import threading
import time
from typing import Any
from uuid import uuid4

from flask import Flask, jsonify, render_template, request

from domain_matcher import DEFAULT_CHUNK_SIZE, DomainMatcher, SearchCancelled, SearchReport


app = Flask(__name__)

JOB_LOCK = threading.Lock()
SEARCH_JOBS: dict[str, dict[str, Any]] = {}


def _job_cancel_requested(job_id: str) -> bool:
    with JOB_LOCK:
        job = SEARCH_JOBS.get(job_id)
        return bool(job and job.get("cancel_requested"))


@lru_cache(maxsize=1)
def get_matcher() -> DomainMatcher:
    return DomainMatcher()


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
        "min_confidence": 0.5,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "max_rows": "",
    }


def _serialize_report(report: SearchReport) -> dict[str, Any]:
    return asdict(report)


def _update_job(job_id: str, **updates: Any) -> None:
    with JOB_LOCK:
        if job_id in SEARCH_JOBS:
            SEARCH_JOBS[job_id].update(updates)


def _run_search_job(job_id: str, form_values: dict[str, Any]) -> None:
    started_at = time.time()
    try:
        _update_job(
            job_id,
            status="starting",
            progress={
                "status": "starting",
                "stage": "candidate_source",
                "stage_detail": "Starting worker and resolving candidate source.",
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
                    "total_predicted_positive": 0,
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
    return render_template(
        "index.html",
        report=None,
        error=None,
        form_values=_empty_form_values(),
    )


@app.post("/api/search")
def start_search():
    payload = request.get_json(silent=True) or request.form.to_dict()
    form_values = {
        "query": payload.get("query", ""),
        "top_k": payload.get("top_k", "25"),
        "min_confidence": payload.get("min_confidence", "0.5"),
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
                "total_predicted_positive": 0,
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
        "min_confidence": request.form.get("min_confidence", "0.5"),
        "chunk_size": request.form.get("chunk_size", str(DEFAULT_CHUNK_SIZE)),
        "max_rows": request.form.get("max_rows", ""),
    }
    try:
        matcher = get_matcher()
        report = matcher.search(
            form_values["query"],
            top_k=max(1, _int_field("top_k", 25, form_values) or 25),
            min_confidence=max(0.0, min(1.0, _float_field("min_confidence", 0.5, form_values))),
            chunk_size=max(1, _int_field("chunk_size", DEFAULT_CHUNK_SIZE, form_values) or DEFAULT_CHUNK_SIZE),
            max_rows=_int_field("max_rows", None, form_values),
        )
        return render_template("index.html", report=report, error=None, form_values=form_values)
    except Exception as exc:
        return render_template("index.html", report=None, error=str(exc), form_values=form_values), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
