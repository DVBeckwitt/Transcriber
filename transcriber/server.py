import argparse
import contextlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import secrets
import shutil
import sys
import threading
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

from .__main__ import (
    MEDIA_EXTENSIONS,
    RunConfig,
    build_config,
    output_paths_for_input,
    parse_args as parse_transcriber_args,
    parse_srt_cues,
    project_dir,
    transcribe_file,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8092
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024 * 1024
DEFAULT_MAX_WORKERS = 1
DEFAULT_JOB_TTL_SECONDS = 24 * 60 * 60
TOKEN_HEADER = "X-Transcribe-Proxy-Token"
UPLOAD_CHUNK_BYTES = 1024 * 1024
LANGUAGES = {"auto", "en", "es"}
TERMINAL_STATUSES = {"succeeded", "failed"}
JOB_DIR_RE = re.compile(r"^[0-9a-f]{32}$")

TranscribeRunner = Callable[..., int]


def default_work_dir() -> Path:
    return project_dir() / ".transcriber_server_jobs"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ServerConfig:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    proxy_token: str | None = None
    work_dir: Path = field(default_factory=default_work_dir)
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    max_workers: int = DEFAULT_MAX_WORKERS
    job_ttl_seconds: int = DEFAULT_JOB_TTL_SECONDS
    device: str = "cuda"
    compute_type: str = "float16"


@dataclass
class JobRecord:
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
    error: dict[str, str] | None = None

    @property
    def status_url(self) -> str:
        return f"/api/transcriptions/{self.job_id}"


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class JobStore:
    def __init__(self, work_dir: Path, ttl_seconds: int) -> None:
        self.work_dir = work_dir
        self.ttl_seconds = ttl_seconds
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def create(self, language: str, extension: str) -> JobRecord:
        job_id = uuid.uuid4().hex
        now_epoch = time.time()
        now = utc_now_iso()
        job_dir = self.work_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        record = JobRecord(
            job_id=job_id,
            language=language,
            status="queued",
            work_dir=job_dir,
            source_path=job_dir / f"transcript{extension}",
            srt_path=job_dir / "transcript.srt",
            txt_path=job_dir / "transcript.txt",
            created_at=now,
            updated_at=now,
            created_at_epoch=now_epoch,
            updated_at_epoch=now_epoch,
        )
        with self._lock:
            self._jobs[job_id] = record
        return record

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

    def mark_failed(self, job_id: str, code: str, message: str) -> None:
        self._update(job_id, status="failed", error={"code": code, "message": message})

    def _update(self, job_id: str, *, status: str, error: dict[str, str] | None) -> None:
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

    def cleanup_expired(self) -> None:
        if self.ttl_seconds <= 0:
            return
        cutoff = time.time() - self.ttl_seconds
        expired: list[JobRecord] = []
        with self._lock:
            for job_id, record in list(self._jobs.items()):
                if record.status in TERMINAL_STATUSES and record.updated_at_epoch <= cutoff:
                    expired.append(record)
                    self._jobs.pop(job_id, None)
        for record in expired:
            shutil.rmtree(record.work_dir, ignore_errors=True)
        self.cleanup_expired_directories(cutoff)

    def cleanup_expired_directories(self, cutoff_epoch: float) -> None:
        active_dirs: set[Path] = set()
        with self._lock:
            active_dirs = {record.work_dir.resolve() for record in self._jobs.values()}

        candidates: list[Path] = []
        with contextlib.suppress(OSError):
            candidates = list(self.work_dir.iterdir())
        for candidate in candidates:
            if not candidate.is_dir() or not JOB_DIR_RE.fullmatch(candidate.name):
                continue
            with contextlib.suppress(OSError):
                resolved = candidate.resolve()
                if resolved in active_dirs:
                    continue
                if candidate.stat().st_mtime <= cutoff_epoch:
                    shutil.rmtree(candidate, ignore_errors=True)


def normalize_config(config: ServerConfig) -> ServerConfig:
    token = (config.proxy_token or "").strip()
    if not token:
        raise ValueError("TRANSCRIBE_PROXY_TOKEN or --proxy-token is required.")
    if int(config.port) < 1 or int(config.port) > 65535:
        raise ValueError("--port must be between 1 and 65535.")
    if int(config.max_upload_bytes) <= 0:
        raise ValueError("--max-upload-bytes must be greater than 0.")
    if int(config.max_workers) < 1:
        raise ValueError("--max-workers must be at least 1.")
    if int(config.job_ttl_seconds) <= 0:
        raise ValueError("--job-ttl-seconds must be greater than 0.")
    compute_type = str(config.compute_type).strip()
    if compute_type not in {"float16", "float32", "int8"}:
        raise ValueError("--compute-type must be one of: float16, float32, int8.")
    device = str(config.device).strip()
    if not device:
        raise ValueError("--device must not be empty.")
    host = str(config.host).strip() or DEFAULT_HOST
    return replace(
        config,
        host=host,
        port=int(config.port),
        proxy_token=token,
        work_dir=Path(config.work_dir).expanduser(),
        max_upload_bytes=int(config.max_upload_bytes),
        max_workers=int(config.max_workers),
        job_ttl_seconds=int(config.job_ttl_seconds),
        device=device,
        compute_type=compute_type,
    )


def _load_fastapi_deps() -> dict[str, Any]:
    try:
        from fastapi import FastAPI, Request
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
        from starlette.datastructures import UploadFile as StarletteUploadFile
        from starlette.exceptions import HTTPException as StarletteHTTPException
    except ImportError as exc:
        raise RuntimeError("Server dependencies are not installed. Install with: pip install -e .[server]") from exc
    return {
        "FastAPI": FastAPI,
        "FileResponse": FileResponse,
        "JSONResponse": JSONResponse,
        "PlainTextResponse": PlainTextResponse,
        "Request": Request,
        "RequestValidationError": RequestValidationError,
        "StarletteHTTPException": StarletteHTTPException,
        "StarletteUploadFile": StarletteUploadFile,
    }


def error_payload(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


def job_payload(record: JobRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "jobId": record.job_id,
        "status": record.status,
        "statusUrl": record.status_url,
        "language": record.language,
        "createdAt": record.created_at,
        "updatedAt": record.updated_at,
    }
    if record.error is not None:
        payload["error"] = record.error
    if record.status == "succeeded":
        payload["transcripts"] = {
            "text": f"{record.status_url}/transcript.txt",
            "srt": f"{record.status_url}/transcript.srt",
        }
    return payload


async def save_upload(upload: Any, destination: Path, max_bytes: int) -> None:
    total = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("wb") as handle:
            while True:
                chunk = await upload.read(UPLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                if total + len(chunk) > max_bytes:
                    raise ApiError(413, "UPLOAD_TOO_LARGE", "Uploaded file exceeds the configured size limit.")
                handle.write(chunk)
                total += len(chunk)
    except ApiError:
        with contextlib.suppress(OSError):
            destination.unlink()
        raise
    except OSError as exc:
        with contextlib.suppress(OSError):
            destination.unlink()
        raise ApiError(500, "UPLOAD_STORE_FAILED", "Could not store the uploaded file.") from exc


async def parse_multipart_form(request: Any, max_upload_bytes: int) -> Any:
    try:
        try:
            return await request.form(max_files=1, max_fields=2, max_part_size=max_upload_bytes)
        except TypeError:
            return await request.form()
    except Exception as exc:
        detail = str(exc).lower()
        if "maximum size" in detail or "max_part_size" in detail or "too large" in detail:
            raise ApiError(413, "UPLOAD_TOO_LARGE", "Uploaded file exceeds the configured size limit.") from exc
        raise ApiError(400, "INVALID_MULTIPART", "Invalid multipart form data.") from exc


def build_job_config(source_path: Path, language: str, server_config: ServerConfig) -> RunConfig:
    args = parse_transcriber_args(
        [
            "--input",
            str(source_path),
            "--lang",
            language,
            "--mode",
            "quality",
            "--no-speaker-labels",
            "--device",
            server_config.device,
            "--compute-type",
            server_config.compute_type,
        ]
    )
    return build_config(args, interactive=False)


def extract_plain_transcript(srt_path: Path) -> str:
    text = srt_path.read_text(encoding="utf-8", errors="ignore")
    cues = parse_srt_cues(text)
    if cues:
        return "\n".join(cue.text.strip() for cue in cues if cue.text.strip()).rstrip() + "\n"

    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.isdigit() or ("-->" in stripped and "," in stripped):
            continue
        lines.append(stripped)
    return "\n".join(lines).rstrip() + ("\n" if lines else "")


def run_transcription_job(
    job_id: str,
    store: JobStore,
    server_config: ServerConfig,
    transcribe_runner: TranscribeRunner,
) -> None:
    record = store.get(job_id)
    if record is None:
        return

    store.mark_running(job_id)
    reports: list[str] = []
    try:
        cfg = build_job_config(record.source_path, record.language, server_config)
        outputs = output_paths_for_input(record.source_path, cfg, create_dirs=True)
        rc = transcribe_runner(cfg, record.source_path, report=reports.append)
        if rc != 0:
            raise RuntimeError("transcription failed")
        if not outputs.srt_path.exists():
            raise RuntimeError("transcript was not created")
        if outputs.srt_path != record.srt_path:
            shutil.copyfile(outputs.srt_path, record.srt_path)
        record.txt_path.write_text(extract_plain_transcript(record.srt_path), encoding="utf-8")
        store.mark_succeeded(job_id)
    except Exception:
        store.mark_failed(job_id, "TRANSCRIPTION_FAILED", "Transcription failed.")
    finally:
        with contextlib.suppress(OSError):
            record.source_path.unlink()


def require_job(store: JobStore, job_id: str) -> JobRecord:
    record = store.get(job_id)
    if record is None:
        raise ApiError(404, "JOB_NOT_FOUND", "Job not found.")
    return record


def require_completed_artifact(store: JobStore, job_id: str, artifact: str) -> Path:
    record = require_job(store, job_id)
    if record.status != "succeeded":
        raise ApiError(409, "JOB_NOT_READY", "Transcript is not ready.")
    path = record.txt_path if artifact == "txt" else record.srt_path
    if not path.exists() or not path.is_file():
        raise ApiError(404, "ARTIFACT_NOT_FOUND", "Transcript artifact not found.")
    return path


def create_app(config: ServerConfig, transcribe_runner: TranscribeRunner = transcribe_file) -> Any:
    deps = _load_fastapi_deps()
    FastAPI = deps["FastAPI"]
    FileResponse = deps["FileResponse"]
    JSONResponse = deps["JSONResponse"]
    PlainTextResponse = deps["PlainTextResponse"]
    Request = deps["Request"]
    RequestValidationError = deps["RequestValidationError"]
    StarletteHTTPException = deps["StarletteHTTPException"]
    StarletteUploadFile = deps["StarletteUploadFile"]

    config = normalize_config(config)
    store = JobStore(config.work_dir, config.job_ttl_seconds)
    executor = ThreadPoolExecutor(max_workers=config.max_workers, thread_name_prefix="transcriber-worker")
    app = FastAPI(title="Transcriber Worker", docs_url=None, redoc_url=None, openapi_url=None)

    def json_error(status_code: int, code: str, message: str) -> Any:
        return JSONResponse(status_code=status_code, content=error_payload(code, message))

    @app.middleware("http")
    async def require_proxy_token(request: Any, call_next: Callable[..., Any]) -> Any:
        provided = request.headers.get(TOKEN_HEADER, "")
        if not provided or not secrets.compare_digest(provided, config.proxy_token or ""):
            return json_error(401, "UNAUTHORIZED", "Missing or invalid proxy token.")
        return await call_next(request)

    @app.middleware("http")
    async def add_security_headers(request: Any, call_next: Callable[..., Any]) -> Any:
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-store")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Any, exc: ApiError) -> Any:
        return json_error(exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(request: Any, exc: Exception) -> Any:
        return json_error(422, "VALIDATION_ERROR", "Invalid request.")

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Any, exc: Any) -> Any:
        if exc.status_code == 404:
            return json_error(404, "NOT_FOUND", "Not found.")
        if exc.status_code == 405:
            return json_error(405, "METHOD_NOT_ALLOWED", "Method not allowed.")
        return json_error(int(exc.status_code), "HTTP_ERROR", "Request could not be handled.")

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Any, exc: Exception) -> Any:
        return json_error(500, "INTERNAL_ERROR", "Internal server error.")

    @app.get("/api/transcriptions/health")
    async def health() -> dict[str, Any]:
        store.cleanup_expired()
        return {
            "status": "ok",
            "maxUploadBytes": config.max_upload_bytes,
            "maxWorkers": config.max_workers,
        }

    @app.post("/api/transcriptions")
    async def create_transcription(request: Request) -> Any:
        store.cleanup_expired()
        form = await parse_multipart_form(request, config.max_upload_bytes)

        upload = form.get("file")
        if not isinstance(upload, StarletteUploadFile):
            raise ApiError(422, "MISSING_FILE", "Multipart field 'file' is required.")

        try:
            raw_language = form.get("language", "auto")
            if isinstance(raw_language, StarletteUploadFile):
                raise ApiError(422, "INVALID_LANGUAGE", "Language must be one of: auto, en, es.")
            language = str(raw_language or "auto").strip().lower()
            if language not in LANGUAGES:
                raise ApiError(422, "INVALID_LANGUAGE", "Language must be one of: auto, en, es.")

            extension = Path(upload.filename or "").suffix.lower()
            if extension not in MEDIA_EXTENSIONS:
                supported = ", ".join(sorted(MEDIA_EXTENSIONS))
                raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE", f"Unsupported media file extension. Supported: {supported}.")

            record = store.create(language, extension)
            try:
                await save_upload(upload, record.source_path, config.max_upload_bytes)
            except ApiError:
                store.remove(record.job_id)
                raise
        finally:
            await upload.close()

        executor.submit(run_transcription_job, record.job_id, store, config, transcribe_runner)
        return JSONResponse(status_code=202, content=job_payload(record))

    @app.get("/api/transcriptions/{job_id}/transcript.txt")
    async def download_transcript_txt(job_id: str) -> Any:
        store.cleanup_expired()
        path = require_completed_artifact(store, job_id, "txt")
        return PlainTextResponse(path.read_text(encoding="utf-8", errors="ignore"))

    @app.get("/api/transcriptions/{job_id}/transcript.srt")
    async def download_transcript_srt(job_id: str) -> Any:
        store.cleanup_expired()
        path = require_completed_artifact(store, job_id, "srt")
        return FileResponse(path, media_type="application/x-subrip", filename="transcript.srt")

    @app.get("/api/transcriptions/{job_id}")
    async def get_transcription(job_id: str) -> dict[str, Any]:
        store.cleanup_expired()
        return job_payload(require_job(store, job_id))

    def shutdown_executor() -> None:
        executor.shutdown(wait=False, cancel_futures=False)

    if hasattr(app, "add_event_handler"):
        app.add_event_handler("shutdown", shutdown_executor)
    else:
        app.router.add_event_handler("shutdown", shutdown_executor)
    app.state.transcriber_config = config
    app.state.transcriber_jobs = store
    return app


def int_env(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


def path_env(env: Mapping[str, str], name: str, default: Path) -> Path:
    raw = env.get(name)
    return Path(raw).expanduser() if raw and raw.strip() else default


def build_arg_parser(env: Mapping[str, str] | None = None) -> argparse.ArgumentParser:
    env = os.environ if env is None else env
    parser = argparse.ArgumentParser(description="Run the Transcriber LAN worker API.")
    parser.add_argument("--host", default=env.get("TRANSCRIBE_SERVER_HOST", DEFAULT_HOST))
    parser.add_argument("--port", type=int, default=int_env(env, "TRANSCRIBE_SERVER_PORT", DEFAULT_PORT))
    parser.add_argument(
        "--proxy-token",
        default=env.get("TRANSCRIBE_PROXY_TOKEN"),
        help="Shared proxy token. Prefer TRANSCRIBE_PROXY_TOKEN so the token is not visible in process listings.",
    )
    parser.add_argument("--work-dir", default=str(path_env(env, "TRANSCRIBE_WORK_DIR", default_work_dir())))
    parser.add_argument(
        "--max-upload-bytes",
        type=int,
        default=int_env(env, "TRANSCRIBE_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES),
    )
    parser.add_argument("--max-workers", type=int, default=int_env(env, "TRANSCRIBE_MAX_WORKERS", DEFAULT_MAX_WORKERS))
    parser.add_argument(
        "--job-ttl-seconds",
        type=int,
        default=int_env(env, "TRANSCRIBE_JOB_TTL_SECONDS", DEFAULT_JOB_TTL_SECONDS),
    )
    parser.add_argument("--device", default=env.get("TRANSCRIBE_DEVICE", "cuda"))
    parser.add_argument(
        "--compute-type",
        choices=("float16", "float32", "int8"),
        default=env.get("TRANSCRIBE_COMPUTE_TYPE", "float16"),
    )
    return parser


def config_from_env_and_args(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> ServerConfig:
    env = os.environ if env is None else env
    parser = build_arg_parser(env)
    args = parser.parse_args(list(argv) if argv is not None else None)
    return ServerConfig(
        host=args.host,
        port=args.port,
        proxy_token=args.proxy_token,
        work_dir=Path(args.work_dir),
        max_upload_bytes=args.max_upload_bytes,
        max_workers=args.max_workers,
        job_ttl_seconds=args.job_ttl_seconds,
        device=args.device,
        compute_type=args.compute_type,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config = normalize_config(config_from_env_and_args(argv))
        app = create_app(config)
    except (RuntimeError, ValueError) as exc:
        print(f"transcriber-server: {exc}", file=sys.stderr)
        return 2

    try:
        import uvicorn
    except ImportError:
        print("transcriber-server: uvicorn is not installed. Install with: pip install -e .[server]", file=sys.stderr)
        return 2

    uvicorn.run(app, host=config.host, port=config.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
