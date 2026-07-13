"""Durable transcription job records and queue admission."""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import utc_now_iso

LANGUAGES = {"auto", "en", "es"}
ACTIVE_STATUSES = {"queued", "running"}
TERMINAL_STATUSES = {"succeeded", "failed"}
JOB_DIR_RE = re.compile(r"^[0-9a-f]{32}$")
JOB_STATE_NAME = "job.json"


@dataclass
class JobRecord:
    """Persisted state and artifact paths for one worker job."""

    job_id: str
    language: str
    status: str
    work_dir: Path
    source_path: Path
    srt_path: Path
    txt_path: Path
    created_at: str
    updated_at: str
    created_at_epoch: float
    updated_at_epoch: float
    error: dict[str, Any] | None = None

    @property
    def status_url(self) -> str:
        return f"/api/transcriptions/{self.job_id}"

    def to_state(self) -> dict[str, Any]:
        return {
            "jobId": self.job_id,
            "language": self.language,
            "status": self.status,
            "sourceName": self.source_path.name,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "createdAtEpoch": self.created_at_epoch,
            "updatedAtEpoch": self.updated_at_epoch,
            "error": self.error,
        }

    @classmethod
    def from_state(cls, job_dir: Path, state: Mapping[str, Any]) -> JobRecord:
        source_name = Path(str(state["sourceName"])).name
        if (
            state.get("jobId") != job_dir.name
            or source_name != state["sourceName"]
            or not source_name.startswith(f"{job_dir.name}.")
        ):
            raise ValueError("Invalid persisted job identity.")

        record = cls(
            job_id=job_dir.name,
            language=str(state["language"]),
            status=str(state["status"]),
            work_dir=job_dir,
            source_path=job_dir / source_name,
            srt_path=job_dir / "transcript.srt",
            txt_path=job_dir / "transcript.txt",
            created_at=str(state["createdAt"]),
            updated_at=str(state["updatedAt"]),
            created_at_epoch=float(state["createdAtEpoch"]),
            updated_at_epoch=float(state["updatedAtEpoch"]),
            error=state.get("error"),
        )
        if record.language not in LANGUAGES:
            raise ValueError("Invalid persisted job language.")
        if record.status not in ACTIVE_STATUSES | TERMINAL_STATUSES:
            raise ValueError("Invalid persisted job status.")
        return record


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class JobStore:
    """Thread-safe durable job store with queue admission and recovery."""

    def __init__(
        self,
        work_dir: Path,
        ttl_seconds: int,
        max_pending_jobs: int | None = None,
    ) -> None:
        self.work_dir = work_dir
        self.ttl_seconds = ttl_seconds
        self.max_pending_jobs = max_pending_jobs
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self._restore_jobs()
        self.cleanup_expired()

    def create(self, language: str, extension: str) -> JobRecord:
        with self._lock:
            if self._queue_is_full():
                raise ApiError(
                    429,
                    "SERVER_BUSY",
                    "The transcription queue is full. Try again later.",
                )

            record = self._new_record(language, extension)
            self._jobs[record.job_id] = record
            if self._persist(record):
                return record

            self._jobs.pop(record.job_id, None)
            shutil.rmtree(record.work_dir, ignore_errors=True)
            raise ApiError(
                500,
                "JOB_STATE_FAILED",
                "Could not initialize durable job state.",
            )

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def remove(self, job_id: str) -> None:
        with self._lock:
            record = self._jobs.pop(job_id, None)
        if record is not None:
            shutil.rmtree(record.work_dir, ignore_errors=True)

    def mark_running(self, job_id: str) -> None:
        self._update(job_id, status="running", error=None)

    def mark_succeeded(self, job_id: str) -> None:
        self._update(job_id, status="succeeded", error=None)

    def mark_failed(
        self,
        job_id: str,
        code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        error: dict[str, Any] = {"code": code, "message": message}
        if details:
            error["details"] = dict(details)
        self._update(job_id, status="failed", error=error)

    def active_count(self) -> int:
        with self._lock:
            return self._active_count()

    def cleanup_expired(self) -> None:
        if self.ttl_seconds <= 0:
            return

        cutoff = time.time() - self.ttl_seconds
        expired: list[JobRecord] = []
        with self._lock:
            for job_id, record in list(self._jobs.items()):
                if (
                    record.status in TERMINAL_STATUSES
                    and record.updated_at_epoch <= cutoff
                ):
                    expired.append(record)
                    self._jobs.pop(job_id, None)

        for record in expired:
            shutil.rmtree(record.work_dir, ignore_errors=True)
        self.cleanup_expired_directories(cutoff)

    def cleanup_expired_directories(self, cutoff_epoch: float) -> None:
        with self._lock:
            active_dirs = {record.work_dir.resolve() for record in self._jobs.values()}

        try:
            candidates = list(self.work_dir.iterdir())
        except OSError:
            return

        for candidate in candidates:
            if not candidate.is_dir() or not JOB_DIR_RE.fullmatch(candidate.name):
                continue
            with contextlib.suppress(OSError):
                resolved = candidate.resolve()
                if (
                    resolved not in active_dirs
                    and candidate.stat().st_mtime <= cutoff_epoch
                ):
                    shutil.rmtree(candidate, ignore_errors=True)

    def _new_record(self, language: str, extension: str) -> JobRecord:
        job_id = uuid.uuid4().hex
        now_epoch = time.time()
        now = utc_now_iso()
        job_dir = self.work_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        return JobRecord(
            job_id=job_id,
            language=language,
            status="queued",
            work_dir=job_dir,
            source_path=job_dir / f"{job_id}{extension}",
            srt_path=job_dir / "transcript.srt",
            txt_path=job_dir / "transcript.txt",
            created_at=now,
            updated_at=now,
            created_at_epoch=now_epoch,
            updated_at_epoch=now_epoch,
        )

    def _queue_is_full(self) -> bool:
        return (
            self.max_pending_jobs is not None
            and self._active_count() >= self.max_pending_jobs
        )

    def _active_count(self) -> int:
        return sum(record.status in ACTIVE_STATUSES for record in self._jobs.values())

    def _update(
        self,
        job_id: str,
        *,
        status: str,
        error: dict[str, Any] | None,
    ) -> None:
        now_epoch = time.time()
        now = utc_now_iso()
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.status = status
            record.error = error
            record.updated_at = now
            record.updated_at_epoch = now_epoch
            self._persist(record)

    def _persist(self, record: JobRecord) -> bool:
        path = record.work_dir / JOB_STATE_NAME
        temp_path = path.with_suffix(".tmp")
        try:
            temp_path.write_text(
                json.dumps(record.to_state(), separators=(",", ":")),
                encoding="utf-8",
            )
            temp_path.replace(path)
            return True
        except OSError:
            with contextlib.suppress(OSError):
                temp_path.unlink()
            return False

    def _restore_jobs(self) -> None:
        try:
            candidates = list(self.work_dir.iterdir())
        except OSError:
            return

        for job_dir in candidates:
            if not job_dir.is_dir() or not JOB_DIR_RE.fullmatch(job_dir.name):
                continue
            try:
                state = json.loads(
                    (job_dir / JOB_STATE_NAME).read_text(encoding="utf-8")
                )
                record = JobRecord.from_state(job_dir, state)
                self._recover_interrupted_record(record)
                self._jobs[record.job_id] = record
            except (KeyError, OSError, TypeError, ValueError):
                continue

    def _recover_interrupted_record(self, record: JobRecord) -> None:
        failure: tuple[str, str] | None = None
        if record.status in ACTIVE_STATUSES:
            failure = (
                "WORKER_RESTARTED",
                "The worker restarted before transcription completed.",
            )
            with contextlib.suppress(OSError):
                record.source_path.unlink()
        elif record.status == "succeeded" and not (
            record.srt_path.is_file() and record.txt_path.is_file()
        ):
            failure = (
                "ARTIFACT_NOT_FOUND",
                "The completed transcript artifacts are missing.",
            )

        if failure is None:
            return
        record.status = "failed"
        record.error = {"code": failure[0], "message": failure[1]}
        record.updated_at = utc_now_iso()
        record.updated_at_epoch = time.time()
        self._persist(record)
