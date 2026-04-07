"""Background job management for the web app."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from . import pipeline_runner


@dataclass
class JobEntry:
    job_id: str
    video_id: str
    params: dict
    status: str = "queued"          # "queued" | "running" | "done" | "error"
    phase: str = "Queued"
    progress_pct: int = 0
    error: str | None = None
    result_paths: dict | None = None
    created_at: float = field(default_factory=time.time)


_jobs: dict[str, JobEntry] = {}
_lock = threading.Lock()


def submit_job(video_id: str, params: dict) -> str:
    job_id = uuid.uuid4().hex
    entry = JobEntry(job_id=job_id, video_id=video_id, params=params)
    with _lock:
        _jobs[job_id] = entry
    t = threading.Thread(target=_run_job, args=(job_id,), daemon=True)
    t.start()
    return job_id


def get_job(job_id: str) -> JobEntry | None:
    with _lock:
        return _jobs.get(job_id)


def list_jobs() -> list[JobEntry]:
    with _lock:
        return sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)


def update_job(job_id: str, **kwargs) -> None:
    with _lock:
        entry = _jobs.get(job_id)
        if entry:
            for k, v in kwargs.items():
                setattr(entry, k, v)


def _run_job(job_id: str) -> None:
    update_job(job_id, status="running")

    entry = get_job(job_id)
    if not entry:
        return

    p = entry.params

    def progress(phase: str, pct: int) -> None:
        update_job(job_id, phase=phase, progress_pct=pct)

    try:
        paths = pipeline_runner.run_pipeline_with_params(
            video_id=entry.video_id,
            output_dir=p.get("output_dir", "./dataset"),
            skip_qa=p.get("skip_qa", False),
            detection_mode=p.get("detection_mode", "ensemble"),
            phase3_params=p.get("phase3", {}),
            phase4_params=p.get("phase4", {}),
            phase7_params=p.get("phase7", {}),
            progress_callback=progress,
        )
        update_job(job_id, status="done", progress_pct=100, phase="Done", result_paths=paths)
    except Exception as exc:
        update_job(job_id, status="error", error=str(exc))
