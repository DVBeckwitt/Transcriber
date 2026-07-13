"""Authenticated FastAPI worker for durable transcription jobs."""

import argparse
import contextlib
import os
import re
import secrets
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .__main__ import MEDIA_EXTENSIONS, output_paths_for_input, transcribe_file
from .config import RunConfig, build_config
from .config import parse_args as parse_transcriber_args
from .jobs import (  # noqa: F401 - compatibility re-exports
    ACTIVE_STATUSES,
    JOB_DIR_RE,
    JOB_STATE_NAME,
    LANGUAGES,
    TERMINAL_STATUSES,
    ApiError,
    JobRecord,
    JobStore,
)
from .subtitles import parse_srt_cues, render_low_confidence_markup
from .utils import project_dir, read_text_tail, utc_now_iso  # noqa: F401

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8092
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024 * 1024
DEFAULT_MAX_WORKERS = 1
DEFAULT_JOB_TTL_SECONDS = 24 * 60 * 60
DEFAULT_PENDING_JOBS_PER_WORKER = 2
TOKEN_HEADER = "X-Transcribe-Proxy-Token"
UPLOAD_CHUNK_BYTES = 1024 * 1024
MAX_FAILURE_REPORT_LINES = 20
MAX_FAILURE_LOG_CHARS = 16 * 1024
MAX_FAILURE_LOG_LINES = 80
WEBM_MIME_TYPES = {"audio/webm", "video/webm", "audio/x-webm", "video/x-webm"}
SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(token|password|secret|api[_-]?key)(\b\s*[:=]\s*)([^\s,;]+)"
)
HF_TOKEN_RE = re.compile(r"\bhf_[A-Za-z0-9_-]{10,}\b")
LOCAL_PATH_RE = re.compile(r"[A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]*")
DIAGNOSTIC_NOISE_PREFIXES = (
    "input:",
    "output:",
    "llm:",
    "log:",
    "lock:",
    "outdir:",
    "lang:",
    "translate:",
    "mode:",
    "model:",
    "diarize:",
    "smooth:",
    "cleanup:",
    "decode:",
    "see the log:",
    "check the log:",
)
DIAGNOSTIC_FAILURE_TERMS = (
    "runtimeerror:",
    "exception:",
    "error:",
    "failed",
    "failure",
    "not found",
    "no active speech",
    "no subtitle cues",
    "returned no",
)

TranscribeRunner = Callable[..., int]


def default_work_dir() -> Path:
    return project_dir() / ".transcriber_server_jobs"


@dataclass(frozen=True)
class ServerConfig:
    """Validated worker process settings."""

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    proxy_token: str | None = None
    work_dir: Path = field(default_factory=default_work_dir)
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES
    max_workers: int = DEFAULT_MAX_WORKERS
    max_pending_jobs: int | None = None
    job_ttl_seconds: int = DEFAULT_JOB_TTL_SECONDS
    device: str = "cuda"
    compute_type: str = "float16"
    warm_vram: bool = False


@dataclass(frozen=True)
class FastAPIDeps:
    fastapi: Any
    request: Any
    file_response: Any
    json_response: Any
    plain_text_response: Any
    request_validation_error: Any
    http_exception: Any
    upload_file: Any


def normalize_config(config: ServerConfig) -> ServerConfig:
    token = (config.proxy_token or "").strip()
    max_workers = int(config.max_workers)
    max_pending_jobs = (
        max_workers * DEFAULT_PENDING_JOBS_PER_WORKER
        if config.max_pending_jobs is None
        else int(config.max_pending_jobs)
    )
    if not token:
        raise ValueError("TRANSCRIBE_PROXY_TOKEN or --proxy-token is required.")
    if int(config.port) < 1 or int(config.port) > 65535:
        raise ValueError("--port must be between 1 and 65535.")
    if int(config.max_upload_bytes) <= 0:
        raise ValueError("--max-upload-bytes must be greater than 0.")
    if max_workers < 1:
        raise ValueError("--max-workers must be at least 1.")
    if max_pending_jobs < max_workers:
        raise ValueError("--max-pending-jobs must be at least --max-workers.")
    if int(config.job_ttl_seconds) <= 0:
        raise ValueError("--job-ttl-seconds must be greater than 0.")
    if config.warm_vram and max_workers != 1:
        raise ValueError(
            "--warm-vram requires --max-workers 1 so one process owns the cached GPU models."
        )
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
        max_workers=max_workers,
        max_pending_jobs=max_pending_jobs,
        job_ttl_seconds=int(config.job_ttl_seconds),
        device=device,
        compute_type=compute_type,
        warm_vram=bool(config.warm_vram),
    )


def _load_fastapi_deps() -> FastAPIDeps:
    try:
        from fastapi import FastAPI, Request
        from fastapi.exceptions import RequestValidationError
        from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
        from starlette.datastructures import UploadFile
        from starlette.exceptions import HTTPException
    except ImportError as exc:
        raise RuntimeError(
            "Server dependencies are not installed. "
            "Install with: pip install -e .[server]"
        ) from exc
    return FastAPIDeps(
        fastapi=FastAPI,
        request=Request,
        file_response=FileResponse,
        json_response=JSONResponse,
        plain_text_response=PlainTextResponse,
        request_validation_error=RequestValidationError,
        http_exception=HTTPException,
        upload_file=UploadFile,
    )


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
                    raise ApiError(
                        413,
                        "UPLOAD_TOO_LARGE",
                        "Uploaded file exceeds the configured size limit.",
                    )
                handle.write(chunk)
                total += len(chunk)
    except ApiError:
        with contextlib.suppress(OSError):
            destination.unlink()
        raise
    except OSError as exc:
        with contextlib.suppress(OSError):
            destination.unlink()
        raise ApiError(
            500, "UPLOAD_STORE_FAILED", "Could not store the uploaded file."
        ) from exc


async def parse_multipart_form(request: Any, max_upload_bytes: int) -> Any:
    try:
        try:
            return await request.form(
                max_files=1, max_fields=2, max_part_size=max_upload_bytes
            )
        except TypeError:
            return await request.form()
    except Exception as exc:
        detail = str(exc).lower()
        if (
            "maximum size" in detail
            or "max_part_size" in detail
            or "too large" in detail
        ):
            raise ApiError(
                413,
                "UPLOAD_TOO_LARGE",
                "Uploaded file exceeds the configured size limit.",
            ) from exc
        raise ApiError(
            400, "INVALID_MULTIPART", "Invalid multipart form data."
        ) from exc


def upload_media_extension(upload: Any) -> str:
    extension = Path(upload.filename or "").suffix.lower()
    if extension in MEDIA_EXTENSIONS:
        return extension

    content_type = (
        str(getattr(upload, "content_type", "") or "").split(";", 1)[0].strip().lower()
    )
    if content_type in WEBM_MIME_TYPES:
        return ".webm"

    supported = ", ".join(sorted(MEDIA_EXTENSIONS))
    raise ApiError(
        415,
        "UNSUPPORTED_MEDIA_TYPE",
        f"Unsupported media file extension. Supported: {supported}.",
    )


def transcription_cli_args(
    source_path: Path, language: str, server_config: ServerConfig
) -> list[str]:
    return [
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
        "--warm-vram" if server_config.warm_vram else "--no-warm-vram",
    ]


def build_job_config(
    source_path: Path, language: str, server_config: ServerConfig
) -> RunConfig:
    return build_config(
        parse_transcriber_args(
            transcription_cli_args(source_path, language, server_config)
        ),
        interactive=False,
    )


def extract_plain_transcript(srt_path: Path) -> str:
    text = srt_path.read_text(encoding="utf-8", errors="ignore")
    cues = parse_srt_cues(text)
    if cues:
        return (
            "\n".join(
                render_low_confidence_markup(cue.text.strip(), "srt")
                for cue in cues
                if cue.text.strip()
            ).rstrip()
            + "\n"
        )

    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if (
            not stripped
            or stripped.isdigit()
            or ("-->" in stripped and "," in stripped)
        ):
            continue
        lines.append(render_low_confidence_markup(stripped, "srt"))
    return "\n".join(lines).rstrip() + ("\n" if lines else "")


def scrub_diagnostic_text(text: str) -> str:
    scrubbed = text.replace("\x00", "")
    scrubbed = SECRET_ASSIGNMENT_RE.sub(r"\1\2<redacted>", scrubbed)
    scrubbed = HF_TOKEN_RE.sub("hf_<redacted>", scrubbed)
    scrubbed = LOCAL_PATH_RE.sub("<local-path>", scrubbed)
    return scrubbed.strip()


def scrub_diagnostic_lines(lines: Sequence[str], limit: int) -> list[str]:
    scrubbed: list[str] = []
    for line in lines[-limit:]:
        cleaned = scrub_diagnostic_text(str(line))
        if cleaned:
            scrubbed.append(cleaned)
    return scrubbed


def read_failure_log_lines(log_path: Path | None) -> list[str]:
    if log_path is None:
        return []
    text = read_text_tail(log_path, max_chars=MAX_FAILURE_LOG_CHARS)
    return [line for line in text.splitlines() if line.strip()][-MAX_FAILURE_LOG_LINES:]


def diagnostic_line_is_noise(line: str) -> bool:
    lower = line.lower().strip()
    return lower.startswith(DIAGNOSTIC_NOISE_PREFIXES) or lower in {
        "arguments: ()",
        "call stack:",
    }


def diagnostic_line_is_failure(line: str) -> bool:
    lower = line.lower()
    return any(term in lower for term in DIAGNOSTIC_FAILURE_TERMS)


def normalize_failure_message(line: str) -> str:
    for prefix in ("RuntimeError:", "Exception:", "Error:"):
        if line.startswith(prefix):
            return line[len(prefix) :].strip() or "Transcription failed."
    return line


def summarize_failure_message(
    exc: Exception, reports: Sequence[str], log_lines: Sequence[str]
) -> str:
    for lines in (log_lines, reports):
        for line in reversed(scrub_diagnostic_lines(lines, max(len(lines), 1))):
            if diagnostic_line_is_noise(line):
                continue
            if diagnostic_line_is_failure(line):
                return normalize_failure_message(line)

    fallback = scrub_diagnostic_text(str(exc))
    if fallback and fallback.lower() != "transcription failed":
        return normalize_failure_message(fallback)
    return "Transcription failed."


def failure_details(
    exc: Exception,
    reports: Sequence[str],
    log_lines: Sequence[str],
    log_path: Path | None,
) -> dict[str, Any]:
    details: dict[str, Any] = {"exceptionType": type(exc).__name__}
    if log_path is not None:
        details["logName"] = log_path.name

    report_lines = scrub_diagnostic_lines(reports, MAX_FAILURE_REPORT_LINES)
    if report_lines:
        details["reports"] = report_lines

    log_tail = scrub_diagnostic_lines(log_lines, MAX_FAILURE_LOG_LINES)
    if log_tail:
        details["logTail"] = log_tail

    return details


def transcription_subprocess_command(
    source_path: Path,
    language: str,
    server_config: ServerConfig,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "transcriber",
        *transcription_cli_args(source_path, language, server_config),
    ]


def run_transcription_subprocess(
    record: JobRecord,
    server_config: ServerConfig,
    reports: list[str],
) -> int:
    process_log = record.work_dir / "transcriber-process.log"
    with process_log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            transcription_subprocess_command(
                record.source_path, record.language, server_config
            ),
            cwd=project_dir(),
            env=os.environ.copy(),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        return_code = process.wait()

    process_lines = [
        line
        for line in read_text_tail(
            process_log, max_chars=MAX_FAILURE_LOG_CHARS
        ).splitlines()
        if line.strip()
    ]
    reports.extend(process_lines[-MAX_FAILURE_LOG_LINES:])
    return return_code


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
    outputs = None
    try:
        cfg = build_job_config(record.source_path, record.language, server_config)
        outputs = output_paths_for_input(record.source_path, cfg, create_dirs=True)
        isolated = transcribe_runner is transcribe_file and not server_config.warm_vram
        if isolated:
            rc = run_transcription_subprocess(record, server_config, reports)
        else:
            rc = transcribe_runner(cfg, record.source_path, report=reports.append)
        if rc != 0:
            if isolated:
                raise RuntimeError(
                    f"transcription subprocess failed with exit code {rc}."
                )
            raise RuntimeError("transcription failed")
        if not outputs.srt_path.exists():
            raise RuntimeError("transcript was not created")
        if outputs.srt_path != record.srt_path:
            shutil.copyfile(outputs.srt_path, record.srt_path)
        record.txt_path.write_text(
            extract_plain_transcript(record.srt_path), encoding="utf-8"
        )
        store.mark_succeeded(job_id)
    except Exception as exc:
        log_path = outputs.log_path if outputs is not None else None
        log_lines = read_failure_log_lines(log_path)
        store.mark_failed(
            job_id,
            "TRANSCRIPTION_FAILED",
            summarize_failure_message(exc, reports, log_lines),
            details=failure_details(exc, reports, log_lines, log_path),
        )
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


def _json_error(deps: FastAPIDeps, status_code: int, code: str, message: str) -> Any:
    return deps.json_response(
        status_code=status_code, content=error_payload(code, message)
    )


def _register_middleware(
    app: Any, config: ServerConfig, store: JobStore, deps: FastAPIDeps
) -> None:
    @app.middleware("http")
    async def require_proxy_token(request: Any, call_next: Callable[..., Any]) -> Any:
        provided = request.headers.get(TOKEN_HEADER, "")
        if not provided or not secrets.compare_digest(
            provided, config.proxy_token or ""
        ):
            return _json_error(
                deps, 401, "UNAUTHORIZED", "Missing or invalid proxy token."
            )
        store.cleanup_expired()
        return await call_next(request)

    @app.middleware("http")
    async def add_security_headers(request: Any, call_next: Callable[..., Any]) -> Any:
        response = await call_next(request)
        for header, value in (
            ("Cache-Control", "no-store"),
            ("X-Content-Type-Options", "nosniff"),
            ("Referrer-Policy", "no-referrer"),
        ):
            response.headers.setdefault(header, value)
        return response


def _register_exception_handlers(app: Any, deps: FastAPIDeps) -> None:
    @app.exception_handler(ApiError)
    async def handle_api_error(request: Any, exc: ApiError) -> Any:
        return _json_error(deps, exc.status_code, exc.code, exc.message)

    @app.exception_handler(deps.request_validation_error)
    async def handle_request_validation(request: Any, exc: Exception) -> Any:
        return _json_error(deps, 422, "VALIDATION_ERROR", "Invalid request.")

    @app.exception_handler(deps.http_exception)
    async def handle_http_error(request: Any, exc: Any) -> Any:
        status_code = int(exc.status_code)
        code, message = {
            404: ("NOT_FOUND", "Not found."),
            405: ("METHOD_NOT_ALLOWED", "Method not allowed."),
        }.get(status_code, ("HTTP_ERROR", "Request could not be handled."))
        return _json_error(deps, status_code, code, message)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Any, exc: Exception) -> Any:
        return _json_error(deps, 500, "INTERNAL_ERROR", "Internal server error.")


async def _store_uploaded_job(
    request: Any,
    config: ServerConfig,
    store: JobStore,
    upload_type: Any,
) -> JobRecord:
    if store.active_count() >= (config.max_pending_jobs or 0):
        raise ApiError(
            429, "SERVER_BUSY", "The transcription queue is full. Try again later."
        )

    form = await parse_multipart_form(request, config.max_upload_bytes)
    upload = form.get("file")
    if not isinstance(upload, upload_type):
        raise ApiError(422, "MISSING_FILE", "Multipart field 'file' is required.")

    try:
        raw_language = form.get("language", "auto")
        if isinstance(raw_language, upload_type):
            raise ApiError(
                422,
                "INVALID_LANGUAGE",
                "Language must be one of: auto, en, es.",
            )
        language = str(raw_language or "auto").strip().lower()
        if language not in LANGUAGES:
            raise ApiError(
                422,
                "INVALID_LANGUAGE",
                "Language must be one of: auto, en, es.",
            )

        record = store.create(language, upload_media_extension(upload))
        try:
            await save_upload(upload, record.source_path, config.max_upload_bytes)
        except BaseException:
            store.remove(record.job_id)
            raise
        return record
    finally:
        await upload.close()


def _submit_job(
    record: JobRecord,
    store: JobStore,
    config: ServerConfig,
    executor: ThreadPoolExecutor,
    transcribe_runner: TranscribeRunner,
) -> None:
    try:
        executor.submit(
            run_transcription_job,
            record.job_id,
            store,
            config,
            transcribe_runner,
        )
    except RuntimeError as exc:
        store.remove(record.job_id)
        raise ApiError(
            503, "SERVER_SHUTTING_DOWN", "The worker is shutting down."
        ) from exc


def _register_routes(
    app: Any,
    config: ServerConfig,
    store: JobStore,
    executor: ThreadPoolExecutor,
    transcribe_runner: TranscribeRunner,
    deps: FastAPIDeps,
) -> None:
    Request = deps.request

    @app.get("/api/transcriptions/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "maxUploadBytes": config.max_upload_bytes,
            "maxWorkers": config.max_workers,
            "activeJobs": store.active_count(),
            "maxPendingJobs": config.max_pending_jobs,
        }

    @app.post("/api/transcriptions")
    async def create_transcription(request: Request) -> Any:
        record = await _store_uploaded_job(request, config, store, deps.upload_file)
        _submit_job(record, store, config, executor, transcribe_runner)
        return deps.json_response(status_code=202, content=job_payload(record))

    @app.get("/api/transcriptions/{job_id}/transcript.txt")
    async def download_transcript_txt(job_id: str) -> Any:
        path = require_completed_artifact(store, job_id, "txt")
        return deps.plain_text_response(
            path.read_text(encoding="utf-8", errors="ignore")
        )

    @app.get("/api/transcriptions/{job_id}/transcript.srt")
    async def download_transcript_srt(job_id: str) -> Any:
        path = require_completed_artifact(store, job_id, "srt")
        return deps.file_response(
            path, media_type="application/x-subrip", filename="transcript.srt"
        )

    @app.get("/api/transcriptions/{job_id}")
    async def get_transcription(job_id: str) -> dict[str, Any]:
        return job_payload(require_job(store, job_id))


def _register_shutdown(app: Any, executor: ThreadPoolExecutor) -> None:
    def shutdown_executor() -> None:
        executor.shutdown(wait=False, cancel_futures=False)

    if hasattr(app, "add_event_handler"):
        app.add_event_handler("shutdown", shutdown_executor)
    else:
        app.router.add_event_handler("shutdown", shutdown_executor)


def create_app(
    config: ServerConfig, transcribe_runner: TranscribeRunner = transcribe_file
) -> Any:
    deps = _load_fastapi_deps()
    config = normalize_config(config)
    store = JobStore(config.work_dir, config.job_ttl_seconds, config.max_pending_jobs)
    executor = ThreadPoolExecutor(
        max_workers=config.max_workers, thread_name_prefix="transcriber-worker"
    )
    app = deps.fastapi(
        title="Transcriber Worker",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    _register_middleware(app, config, store, deps)
    _register_exception_handlers(app, deps)
    _register_routes(app, config, store, executor, transcribe_runner, deps)
    _register_shutdown(app, executor)
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


def bool_env(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean.")


def path_env(env: Mapping[str, str], name: str, default: Path) -> Path:
    raw = env.get(name)
    return Path(raw).expanduser() if raw and raw.strip() else default


def build_arg_parser(env: Mapping[str, str] | None = None) -> argparse.ArgumentParser:
    env = os.environ if env is None else env
    pending_jobs = env.get("TRANSCRIBE_MAX_PENDING_JOBS")
    pending_jobs_default = (
        int_env(env, "TRANSCRIBE_MAX_PENDING_JOBS", 0)
        if pending_jobs and pending_jobs.strip()
        else None
    )
    parser = argparse.ArgumentParser(description="Run the Transcriber LAN worker API.")
    parser.add_argument(
        "--host", default=env.get("TRANSCRIBE_SERVER_HOST", DEFAULT_HOST)
    )
    parser.add_argument(
        "--port", type=int, default=int_env(env, "TRANSCRIBE_SERVER_PORT", DEFAULT_PORT)
    )
    parser.add_argument(
        "--proxy-token",
        default=env.get("TRANSCRIBE_PROXY_TOKEN"),
        help="Shared proxy token. Prefer TRANSCRIBE_PROXY_TOKEN so the token is not visible in process listings.",
    )
    parser.add_argument(
        "--work-dir",
        default=str(path_env(env, "TRANSCRIBE_WORK_DIR", default_work_dir())),
    )
    parser.add_argument(
        "--max-upload-bytes",
        type=int,
        default=int_env(env, "TRANSCRIBE_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int_env(env, "TRANSCRIBE_MAX_WORKERS", DEFAULT_MAX_WORKERS),
    )
    parser.add_argument(
        "--max-pending-jobs",
        type=int,
        default=pending_jobs_default,
        help="Maximum queued plus running jobs (default: twice max workers).",
    )
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
    parser.add_argument(
        "--warm-vram",
        action=argparse.BooleanOptionalAction,
        default=bool_env(env, "TRANSCRIBE_WARM_VRAM", False),
        help="Keep compatible WhisperX models loaded between jobs for faster repeated runs (default: off).",
    )
    return parser


def config_from_env_and_args(
    argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None
) -> ServerConfig:
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
        max_pending_jobs=args.max_pending_jobs,
        job_ttl_seconds=args.job_ttl_seconds,
        device=args.device,
        compute_type=args.compute_type,
        warm_vram=args.warm_vram,
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
        print(
            "transcriber-server: uvicorn is not installed. Install with: pip install -e .[server]",
            file=sys.stderr,
        )
        return 2

    uvicorn.run(app, host=config.host, port=config.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
