"""WhisperX model execution, transcription orchestration, and CLI entry point."""

from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib
import inspect
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from . import translation as _translation
from .config import (  # noqa: F401 - compatibility re-exports
    DEFAULT_POLL_INTERVAL,
    DEFAULT_SETTLE_SECONDS,
    DEFAULT_STALE_LOCK_SECONDS,
    DEFAULT_WATCH_DIR,
    LEGACY_ALIASES,
    MEDIA_FILTER,
    MODE_PRESETS,
    SPANISH_TRANSLATION_MODEL,
    TRANSLATION_BATCH_SIZE,
    TRANSLATION_CONTEXT_WINDOW,
    TRANSLATION_LENGTH_PENALTY,
    TRANSLATION_MARKER_END,
    TRANSLATION_MARKER_START,
    TRANSLATION_MAX_NEW_TOKENS,
    TRANSLATION_NO_REPEAT_NGRAM_SIZE,
    TRANSLATION_NUM_BEAMS,
    LegacyOptions,
    ModePreset,
    RunConfig,
    build_asr_prompt,
    build_config,
    load_glossary_file,
    load_text_lines_file,
    looks_like_glossary_file,
    parse_args,
    parse_glossary_entries,
    parse_legacy,
    parse_temperature_schedule,
    pick_media_file,
    prompt_choice,
    prompt_input_path,
    prompt_language,
    prompt_mode,
    resolve_input_path,
)
from .subtitles import (  # noqa: F401 - compatibility re-exports
    DEFAULT_HIGH_NO_SPEECH_PROB,
    DEFAULT_LOW_CONFIDENCE_LOGPROB,
    DEFAULT_LOW_CONFIDENCE_WORD_PROB,
    DEFAULT_MIN_SPEAKER_TURN_MS,
    DEFAULT_MIN_SPEAKER_TURN_TOKENS,
    LEGACY_LOW_CONFIDENCE_MARKER,
    LOW_CONFIDENCE_MARKER_NOISE_HINTS,
    LOW_CONFIDENCE_MARKER_NOISE_WORDS,
    LOW_CONFIDENCE_MARKER_RE,
    LOW_CONFIDENCE_PLACEHOLDER,
    SUBTITLE_MAX_CHARS_PER_LINE,
    SUBTITLE_MAX_DURATION_SECONDS,
    SUBTITLE_MAX_LINES,
    SUBTITLE_PREFERRED_BREAK_CHARS,
    SUBTITLE_TARGET_CPS,
    SRTCue,
    TimedToken,
    apply_confidence_cleanup,
    build_segment_fallback_cues,
    build_srt_cues_from_result,
    cue_candidate_is_valid,
    extract_timed_tokens,
    finalize_timed_cue,
    format_cue_text,
    format_token_text,
    ms_to_timestamp,
    normalize_speaker_label,
    normalize_subtitle_whitespace,
    parse_srt_cues,
    probability_value,
    reading_speed_cps,
    render_low_confidence_markup,
    render_srt_cues,
    seconds_to_ms,
    segment_is_low_confidence,
    segment_to_timed_tokens,
    should_soft_break,
    smooth_timed_tokens,
    speaker_prefix,
    split_cue_for_subtitles,
    split_speaker_prefix,
    split_text_into_chunks,
    strip_low_confidence_marker_noise,
    timestamp_to_ms,
    token_low_confidence_marker,
    word_is_low_confidence,
    wrap_subtitle_lines,
    wrap_subtitle_lines_exact,
    write_direct_srt_from_result,
)
from .translation import (  # noqa: F401 - compatibility re-exports
    TranslationCue,
    _build_translation_cues,
    _render_translated_cues,
    _resolve_translation_texts,
    _translate_with_settings,
    _translation_glossary,
    apply_glossary_placeholders,
    build_translation_prompt,
    chunked_text,
    extract_between_markers,
    load_spanish_to_english_translator,
    load_translation_glossary,
    log_translation_prompt,
    replace_glossary_placeholders,
    translation_context_for_cue,
)
from .utils import (  # noqa: F401 - compatibility re-exports
    LOG_TAIL_READ_CHARS,
    project_dir,
    read_text_tail,
    utc_now_iso,
)

MEDIA_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".weba",
}

LOG_DIR_NAME = "logs"
WATCHER_LOG_NAME = "transcriber-watcher.log"
WATCH_RETRY_COOLDOWN_SECONDS = 300.0
FALLBACK_TEMP_DIR_NAME = ".tmp_transcriber_temp"
LOCK_SUFFIX = ".transcribing.lock"
AUDIO_PREPROCESS_TIMEOUT_SECONDS = 30 * 60
SUBPROCESS_ERROR_DETAIL_LIMIT = 500
FFMPEG_PATH_ENV_VAR = "TRANSCRIBE_FFMPEG"


Reporter = Callable[[str], None]
ModelT = TypeVar("ModelT")

_WARM_ASR_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}
_WARM_ALIGN_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}
_WARM_DIARIZATION_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}
_WARM_MODEL_CACHE_LOCK = threading.RLock()


@dataclass
class OutputPaths:
    """Filesystem locations produced for one input recording."""

    output_dir: Path
    srt_path: Path
    llm_path: Path
    log_path: Path
    lock_path: Path


@dataclass
class PendingWatchFile:
    size: int
    mtime_ns: int
    stable_since: float
    last_attempt_at: float | None = None

    def update_signature(self, size: int, mtime_ns: int, now: float) -> bool:
        if self.size == size and self.mtime_ns == mtime_ns:
            return False
        self.size = size
        self.mtime_ns = mtime_ns
        self.stable_since = now
        self.last_attempt_at = None
        return True

    def is_ready(self, now: float, settle_seconds: float) -> bool:
        if now - self.stable_since < settle_seconds:
            return False
        return (
            self.last_attempt_at is None
            or now - self.last_attempt_at >= WATCH_RETRY_COOLDOWN_SECONDS
        )


@dataclass(frozen=True)
class TranscriptionAttemptResult:
    return_code: int
    detected_language: str | None
    fell_back_from_diarization: bool


DIARIZATION_FALLBACK_PATTERNS = (
    "could not download 'pyannote/speaker-diarization-3.1' pipeline.",
    "visit https://hf.co/pyannote/speaker-diarization-3.1 to accept the user conditions.",
    "attributeerror: 'nonetype' object has no attribute 'to'",
    "attributeerror: module 'whisperx' has no attribute 'diarizationpipeline'",
    "could not find whisperx symbol 'diarizationpipeline'",
    "could not find whisperx symbol 'assign_word_speakers'",
    "unpicklingerror",
    "unsupported global: global",
)


# Reporting and model lifecycle


def report_lines(report: Reporter, *lines: str) -> None:
    for line in lines:
        report(line)


def report_block(report: Reporter, *lines: str) -> None:
    report_lines(report, "", *lines, "")


def call_with_supported_kwargs(
    func: Callable[..., Any], *args: Any, **kwargs: Any
) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)

    parameters = signature.parameters
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    ):
        return func(*args, **kwargs)

    filtered = {key: value for key, value in kwargs.items() if key in parameters}
    return func(*args, **filtered)


def resolve_whisperx_symbol(whisperx_module: Any, symbol_name: str) -> Any:
    if hasattr(whisperx_module, symbol_name):
        return getattr(whisperx_module, symbol_name)

    for module_name in ("diarize",):
        with contextlib.suppress(Exception):
            module = importlib.import_module(
                f"{whisperx_module.__name__}.{module_name}"
            )
            if hasattr(module, symbol_name):
                return getattr(module, symbol_name)

    raise AttributeError(
        f"Could not find whisperx symbol {symbol_name!r} in {whisperx_module.__name__} or its supported compatibility modules."
    )


def flush_gpu_memory(device: str | None = None) -> None:
    normalized_device = str(device or "").strip().lower()
    if normalized_device and not normalized_device.startswith("cuda"):
        return

    try:
        import torch
    except Exception:
        return

    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return

    is_available = getattr(cuda, "is_available", None)
    if callable(is_available):
        try:
            if not is_available():
                return
        except Exception:
            return

    with contextlib.suppress(Exception):
        gc.collect()

    for cleanup_name in ("empty_cache", "ipc_collect"):
        cleanup = getattr(cuda, cleanup_name, None)
        if callable(cleanup):
            with contextlib.suppress(Exception):
                cleanup()


def clear_warm_model_caches() -> None:
    with _WARM_MODEL_CACHE_LOCK:
        _WARM_ASR_MODEL_CACHE.clear()
        _WARM_ALIGN_MODEL_CACHE.clear()
        _WARM_DIARIZATION_MODEL_CACHE.clear()


def cleanup_after_transcription_run(cfg: RunConfig) -> None:
    if cfg.warm_vram:
        return
    clear_warm_model_caches()
    flush_gpu_memory(cfg.device)


def load_cached_model(
    cache: dict[tuple[Any, ...], ModelT],
    key: tuple[Any, ...],
    loader: Callable[[], ModelT],
    *,
    warm: bool,
) -> ModelT:
    if not warm:
        return loader()

    with _WARM_MODEL_CACHE_LOCK:
        cached = cache.get(key)
        if cached is None:
            cached = loader()
            cache[key] = cached
        return cached


def load_whisperx_asr_model(
    whisperx_module: Any,
    cfg: RunConfig,
    whisper_task: str,
    whisper_language: str | None,
) -> Any:
    load_kwargs: dict[str, Any] = {
        "device": cfg.device,
        "compute_type": cfg.compute_type,
        "task": whisper_task,
        "asr_options": {"beam_size": cfg.beam_size, "patience": cfg.patience},
        "vad_method": "silero",
    }
    if whisper_language:
        load_kwargs["language"] = whisper_language

    cache_key = (
        cfg.model,
        cfg.device,
        cfg.compute_type,
        whisper_task,
        whisper_language or "",
        cfg.beam_size,
        cfg.patience,
    )
    return load_cached_model(
        _WARM_ASR_MODEL_CACHE,
        cache_key,
        lambda: call_with_supported_kwargs(
            whisperx_module.load_model, cfg.model, **load_kwargs
        ),
        warm=cfg.warm_vram,
    )


def load_whisperx_align_model(
    whisperx_module: Any, cfg: RunConfig, language_code: str
) -> tuple[Any, Any]:
    return load_cached_model(
        _WARM_ALIGN_MODEL_CACHE,
        (language_code, cfg.device),
        lambda: call_with_supported_kwargs(
            whisperx_module.load_align_model,
            language_code=language_code,
            device=cfg.device,
        ),
        warm=cfg.warm_vram,
    )


def hf_token_cache_digest(hf_token: str | None) -> str:
    """Create a stable cache key without retaining the raw access token."""
    return hashlib.sha256((hf_token or "").encode("utf-8")).hexdigest()


def load_whisperx_diarization_model(
    whisperx_module: Any, cfg: RunConfig, hf_token: str | None
) -> Any:
    pipeline = resolve_whisperx_symbol(whisperx_module, "DiarizationPipeline")
    return load_cached_model(
        _WARM_DIARIZATION_MODEL_CACHE,
        (cfg.device, hf_token_cache_digest(hf_token)),
        lambda: call_with_supported_kwargs(
            pipeline,
            use_auth_token=hf_token or "",
            device=cfg.device,
        ),
        warm=cfg.warm_vram,
    )


# Translation compatibility facade


def translate_spanish_texts(
    texts: Sequence[str],
    device: str,
    batch_size: int = 4,
    num_beams: int = TRANSLATION_NUM_BEAMS,
    max_new_tokens: int = 256,
    no_repeat_ngram_size: int = TRANSLATION_NO_REPEAT_NGRAM_SIZE,
) -> list[str]:
    """Translate text while retaining the historical dependency patch points."""

    return _translation.translate_spanish_texts(
        texts,
        device=device,
        batch_size=batch_size,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        no_repeat_ngram_size=no_repeat_ngram_size,
        load_translator=load_spanish_to_english_translator,
        cleanup_gpu=flush_gpu_memory,
    )


def translate_srt_to_english(
    srt_path: Path,
    device: str,
    glossary: dict[str, str] | None = None,
    glossary_spec: str | None = None,
    context_window: int = TRANSLATION_CONTEXT_WINDOW,
    batch_size: int = TRANSLATION_BATCH_SIZE,
    num_beams: int = TRANSLATION_NUM_BEAMS,
    max_new_tokens: int = TRANSLATION_MAX_NEW_TOKENS,
    no_repeat_ngram_size: int = TRANSLATION_NO_REPEAT_NGRAM_SIZE,
    log_path: Path | None = None,
) -> None:
    """Translate an SRT while retaining the historical translator patch point."""

    _translation.translate_srt_to_english(
        srt_path,
        device,
        glossary=glossary,
        glossary_spec=glossary_spec,
        context_window=context_window,
        batch_size=batch_size,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        no_repeat_ngram_size=no_repeat_ngram_size,
        log_path=log_path,
        translate_texts=translate_spanish_texts,
    )


# Transcript artifacts and filesystem preparation


LLM_TRANSCRIPT_PREFACE = (
    "You are given an automatic transcript.\n"
    "Refine it into a cleaner transcript in the same language.\n"
    "Preserve speaker labels if present.\n"
    "Do not translate, summarize, or rewrite more than necessary.\n"
    "Fix obvious punctuation, capitalization, spacing, and clear recognition mistakes.\n"
    "Keep the meaning and cadence close to the source.\n"
    "Use square brackets for brief editorial notes such as [inaudible], [crosstalk], or [name unclear].\n"
    "If a word or short phrase is low confidence, leave the em dash placeholder as-is.\n"
    "If you are not confident enough to refine a passage cleanly, keep it cautious instead of guessing.\n"
    "Output only the refined transcript.\n\n"
    "TRANSCRIPT:\n"
)


def llm_text_lines_from_srt_lines(lines: Iterable[str]) -> list[str]:
    text_lines = []
    for line in lines:
        s = line.strip()
        if not s or s.isdigit() or ("-->" in s and "," in s):
            continue
        text_lines.append(render_low_confidence_markup(s, "llm"))
    return text_lines


def build_llm_file(srt_path: Path, llm_path: Path) -> None:
    if not srt_path.exists():
        return

    text_lines = llm_text_lines_from_srt_lines(
        srt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    )
    llm_path.write_text(
        LLM_TRANSCRIPT_PREFACE + "\n".join(text_lines), encoding="utf-8"
    )


def finalize_transcript_outputs(srt_path: Path, llm_path: Path) -> None:
    if not srt_path.exists():
        return

    raw_lines = srt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    finalized_lines = [render_low_confidence_markup(line, "srt") for line in raw_lines]
    llm_path.write_text(
        LLM_TRANSCRIPT_PREFACE
        + "\n".join(llm_text_lines_from_srt_lines(finalized_lines)),
        encoding="utf-8",
    )
    srt_path.write_text("\n".join(finalized_lines).rstrip() + "\n", encoding="utf-8")


def build_lock_payload(input_path: Path) -> str:
    return (
        f"source_path={input_path}\n"
        f"created_at={utc_now_iso()}\n"
        f"pid={os.getpid()}\n"
        f"hostname={socket.gethostname()}\n"
    )


def temp_dir_candidates(base_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    for env_name in ("TMPDIR", "TEMP", "TMP"):
        raw = os.environ.get(env_name, "").strip()
        if not raw:
            continue
        key = os.path.normcase(raw)
        if key in seen:
            continue
        seen.add(key)
        candidates.append(Path(raw))

    local_appdata = os.environ.get("LOCALAPPDATA", "").strip()
    if local_appdata:
        managed = Path(local_appdata) / "Transcriber" / "tmp"
        key = os.path.normcase(str(managed))
        if key not in seen:
            seen.add(key)
            candidates.append(managed)

    fallback = base_dir / FALLBACK_TEMP_DIR_NAME
    key = os.path.normcase(str(fallback))
    if key not in seen:
        candidates.append(fallback)

    return candidates


def probe_temp_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False

    probe_path = path / f".probe-{os.getpid()}-{time.time_ns()}"
    try:
        probe_path.write_bytes(b"ok")
        return True
    except OSError:
        return False
    finally:
        with contextlib.suppress(OSError):
            probe_path.unlink()


def configure_temp_dir(base_dir: Path) -> Path:
    for candidate in temp_dir_candidates(base_dir):
        candidate = candidate.expanduser()
        if not probe_temp_dir(candidate):
            continue

        resolved = candidate.resolve()
        temp_path = str(resolved)
        for env_name in ("TMPDIR", "TEMP", "TMP"):
            os.environ[env_name] = temp_path
        tempfile.tempdir = temp_path
        return resolved

    raise RuntimeError(
        "Could not create a usable temporary directory for WhisperX. Check TMP/TEMP permissions."
    )


def ffmpeg_candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    explicit_path = (os.environ.get(FFMPEG_PATH_ENV_VAR) or "").strip().strip('"')
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())

    path_match = shutil.which("ffmpeg")
    if path_match:
        candidates.append(Path(path_match))

    chocolatey_root = Path(
        os.environ.get("ChocolateyInstall", r"C:\ProgramData\chocolatey")
    )
    candidates.extend(
        [
            Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
            chocolatey_root / "bin" / "ffmpeg.exe",
            Path.home()
            / "scoop"
            / "apps"
            / "ffmpeg"
            / "current"
            / "bin"
            / "ffmpeg.exe",
        ]
    )
    return candidates


def resolve_ffmpeg_executable() -> str | None:
    seen: set[str] = set()
    for candidate in ffmpeg_candidate_paths():
        key = os.path.normcase(str(candidate))
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return str(candidate)
    return None


def ensure_ffmpeg_available_for_child_processes() -> str:
    ffmpeg_path = resolve_ffmpeg_executable()
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg executable not found. Install ffmpeg, set TRANSCRIBE_FFMPEG, "
            r"or put ffmpeg on PATH, for example C:\ffmpeg\bin\ffmpeg.exe."
        )
    if Path(ffmpeg_path).stem.lower() != "ffmpeg":
        raise RuntimeError(
            "Resolved ffmpeg executable must be named ffmpeg because WhisperX loads audio "
            'by launching "ffmpeg" by name. Set TRANSCRIBE_FFMPEG to a standard ffmpeg '
            "executable or put ffmpeg on PATH."
        )

    ffmpeg_dir = str(Path(ffmpeg_path).expanduser().resolve().parent)
    current_path = os.environ.get("PATH", "")
    normalized_ffmpeg_dir = os.path.normcase(os.path.normpath(ffmpeg_dir))
    normalized_path_parts = {
        os.path.normcase(os.path.normpath(path_part.strip().strip('"')))
        for path_part in current_path.split(os.pathsep)
        if path_part.strip()
    }

    if normalized_ffmpeg_dir not in normalized_path_parts:
        os.environ["PATH"] = (
            ffmpeg_dir
            if not current_path
            else f"{ffmpeg_dir}{os.pathsep}{current_path}"
        )

    return ffmpeg_path


def build_audio_preprocess_command(
    input_path: Path, output_path: Path, ffmpeg_path: str = "ffmpeg"
) -> list[str]:
    return [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-map",
        "0:a:0?",
        "-vn",
        "-sn",
        "-dn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        "highpass=f=60,lowpass=f=8000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]


def format_subprocess_error_detail(
    value: object, limit: int = SUBPROCESS_ERROR_DETAIL_LIMIT
) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()} ... [truncated]"


def preprocess_audio_for_whisperx(
    input_path: Path, temp_dir: Path, report: Reporter = print
) -> Path:
    output_path = temp_dir / f"{input_path.stem}.preprocessed.wav"
    ffmpeg_path = resolve_ffmpeg_executable()
    if ffmpeg_path is None:
        report(
            "[transcriber] ffmpeg not found on PATH or common install paths; using the original input audio."
        )
        return input_path
    command = build_audio_preprocess_command(input_path, output_path, ffmpeg_path)

    try:
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=AUDIO_PREPROCESS_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        report(
            f"[transcriber] ffmpeg not found at {ffmpeg_path}; using the original input audio."
        )
        return input_path
    except subprocess.TimeoutExpired:
        report(
            "[transcriber] Audio preprocessing timed out "
            f"after {AUDIO_PREPROCESS_TIMEOUT_SECONDS:g}s; using the original input audio."
        )
        with contextlib.suppress(OSError):
            output_path.unlink()
        return input_path
    except subprocess.CalledProcessError as exc:
        stderr = format_subprocess_error_detail(exc.stderr)
        detail = f": {stderr}" if stderr else ""
        report(
            f"[transcriber] Audio preprocessing failed; using the original input audio{detail}"
        )
        with contextlib.suppress(OSError):
            output_path.unlink()
        return input_path
    except Exception as exc:
        report(
            f"[transcriber] Audio preprocessing failed; using the original input audio: {exc}"
        )
        with contextlib.suppress(OSError):
            output_path.unlink()
        return input_path

    if not output_path.exists() or output_path.stat().st_size == 0:
        report(
            "[transcriber] Audio preprocessing produced no output; using the original input audio."
        )
        with contextlib.suppress(OSError):
            output_path.unlink()
        return input_path

    return output_path


def output_paths_for_input(
    input_path: Path, cfg: RunConfig, create_dirs: bool = False
) -> OutputPaths:
    output_dir = input_path.parent
    log_dir = log_dir_for_output(output_dir, create_dirs=create_dirs)
    if create_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
    base = input_path.stem
    return OutputPaths(
        output_dir=output_dir,
        srt_path=output_dir / f"{base}.srt",
        llm_path=output_dir / f"{base}_llm.txt",
        log_path=log_dir / f"{base}_whisperx.log",
        lock_path=output_dir / f"{base}{LOCK_SUFFIX}",
    )


def log_dir_for_output(output_dir: Path, *, create_dirs: bool) -> Path:
    preferred = project_dir() / LOG_DIR_NAME
    if not create_dirs or ensure_log_dir_usable(preferred):
        return preferred
    if ensure_log_dir_usable(output_dir):
        return output_dir

    temp_log_dir = Path(tempfile.gettempdir()) / "transcriber-logs"
    if ensure_log_dir_usable(temp_log_dir):
        return temp_log_dir
    return preferred


def ensure_log_dir_usable(log_dir: Path) -> bool:
    probe_path: Path | None = None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=log_dir,
            prefix=".transcriber-log-probe-",
            suffix=".tmp",
            delete=False,
        ) as probe:
            probe_path = Path(probe.name)
            probe.write("")
        probe_path.unlink()
        return True
    except OSError:
        if probe_path is not None:
            with contextlib.suppress(OSError):
                probe_path.unlink()
        return False


def is_stale_lock(lock_path: Path, stale_lock_seconds: float) -> bool:
    try:
        age_seconds = time.time() - lock_path.stat().st_mtime
    except OSError:
        return False
    return age_seconds >= stale_lock_seconds


def try_remove_lock(lock_path: Path) -> bool:
    with contextlib.suppress(OSError):
        lock_path.unlink()
        return True
    return False


def acquire_lock(
    input_path: Path,
    lock_path: Path,
    stale_lock_seconds: float,
    report: Reporter,
) -> bool:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_lock_payload(input_path)

    for _ in range(2):
        try:
            with lock_path.open("x", encoding="utf-8") as fh:
                fh.write(payload)
            return True
        except FileExistsError:
            if is_stale_lock(lock_path, stale_lock_seconds):
                report(f'Clearing stale lock "{lock_path.name}".')
                if try_remove_lock(lock_path):
                    continue
            try:
                lock_text = lock_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                lock_text = ""
            lock_info = {}
            for line in lock_text.splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                lock_info[key.strip()] = value.strip()
            owner = lock_info.get("hostname") or "unknown host"
            pid = lock_info.get("pid")
            created_at = lock_info.get("created_at") or "unknown time"
            report(
                f'Skipping "{input_path.name}" because it is already locked '
                f"(host={owner}, pid={pid}, created_at={created_at})."
            )
            return False
    return False


def release_lock(lock_path: Path, report: Reporter | None = None) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        if report is not None:
            report(f'Could not remove lock "{lock_path}": {exc}')


def load_hf_token(base_dir: Path) -> str | None:
    env_token = os.environ.get("HF_TOKEN", "").strip()
    if env_token:
        return env_token.strip('"').strip("'")

    for name in ("hf_token.txt", "HF_TOKEN.txt"):
        token_file = base_dir / name
        if not token_file.exists():
            continue
        raw = token_file.read_text(encoding="utf-8-sig", errors="ignore")
        for line in raw.splitlines():
            token = line.strip().strip('"').strip("'")
            if token:
                return token
    return None


@contextlib.contextmanager
def allow_trusted_checkpoint_loads() -> Iterable[None]:
    try:
        import torch
    except Exception:
        yield
        return

    original_load = torch.load

    def patched_load(*args: Any, **kwargs: Any) -> Any:
        if "weights_only" not in kwargs or kwargs["weights_only"] is None:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_load


# WhisperX execution


def should_fallback_without_diarization(log_path: Path) -> bool:
    text = read_text_tail(log_path).lower()
    return any(pattern in text for pattern in DIARIZATION_FALLBACK_PATTERNS)


def diarization_error_allows_fallback(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(pattern in text for pattern in DIARIZATION_FALLBACK_PATTERNS)


def parse_detected_language_from_log(log_path: Path) -> str | None:
    text = read_text_tail(log_path)
    matches = re.findall(r"Detected language:\s*([A-Za-z-]+)\s*\(", text)
    if not matches:
        return None
    return matches[-1].strip().lower() or None


def _whisperx_transcribe_kwargs(
    cfg: RunConfig, whisper_task: str, whisper_language: str | None
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "task": whisper_task,
        "temperature": cfg.temperature_schedule or cfg.temperature,
        "print_progress": False,
        "condition_on_previous_text": cfg.condition_on_previous_text,
    }
    optional_values = {
        "initial_prompt": cfg.asr_prompt,
        "best_of": cfg.best_of,
        "compression_ratio_threshold": cfg.compression_ratio_threshold,
        "logprob_threshold": cfg.logprob_threshold,
        "no_speech_threshold": cfg.no_speech_threshold,
        "language": whisper_language,
    }
    kwargs.update(
        {name: value for name, value in optional_values.items() if value is not None}
    )
    return kwargs


def _transcribe_with_whisperx(
    model: Any,
    cfg: RunConfig,
    audio: Any,
    whisper_task: str,
    whisper_language: str | None,
) -> tuple[dict[str, Any], str | None]:
    print("[transcriber] Transcribing audio...")
    result = call_with_supported_kwargs(
        model.transcribe,
        audio,
        **_whisperx_transcribe_kwargs(cfg, whisper_task, whisper_language),
    )
    if not isinstance(result, dict):
        raise RuntimeError(
            f"Unexpected WhisperX transcription result type: {type(result)!r}"
        )
    detected_language = str(result.get("language") or "").strip().lower() or None
    return result, detected_language


def _align_whisperx_result(
    whisperx_module: Any,
    cfg: RunConfig,
    audio: Any,
    result: dict[str, Any],
    language_code: str,
) -> dict[str, Any]:
    print(f"[transcriber] Aligning words for language={language_code}...")
    try:
        align_model, metadata = load_whisperx_align_model(
            whisperx_module, cfg, language_code
        )
        aligned = call_with_supported_kwargs(
            whisperx_module.align,
            result.get("segments", []),
            align_model,
            metadata,
            audio,
            cfg.device,
            return_char_alignments=False,
        )
        if not isinstance(aligned, dict):
            raise RuntimeError(
                f"Unexpected WhisperX alignment result type: {type(aligned)!r}"
            )
        return aligned
    except Exception as exc:
        print(f"[transcriber] Alignment failed; continuing without alignment: {exc}")
        return result


def _diarize_whisperx_result(
    whisperx_module: Any,
    cfg: RunConfig,
    audio: Any,
    result: dict[str, Any],
    hf_token: str | None,
) -> dict[str, Any]:
    print("[transcriber] Running diarization...")
    try:
        with allow_trusted_checkpoint_loads():
            diarization_model = load_whisperx_diarization_model(
                whisperx_module, cfg, hf_token
            )
            diarization_segments = call_with_supported_kwargs(diarization_model, audio)
        assign_word_speakers = resolve_whisperx_symbol(
            whisperx_module, "assign_word_speakers"
        )
        return assign_word_speakers(diarization_segments, result)
    except Exception as exc:
        if not diarization_error_allows_fallback(exc):
            raise
        print(
            "[transcriber] Diarization unavailable or blocked; "
            f"continuing without diarization: {exc}"
        )
        return result


def run_whisperx_direct(
    cfg: RunConfig,
    input_path: Path,
    srt_path: Path,
    hf_token: str | None,
    diarize: bool,
) -> str | None:
    lock = _WARM_MODEL_CACHE_LOCK if cfg.warm_vram else contextlib.nullcontext()
    with lock:
        import whisperx

        whisper_language = None if cfg.language == "auto" else cfg.language
        whisper_task = "translate" if cfg.translate_to_english else "transcribe"

        ensure_ffmpeg_available_for_child_processes()
        print(f"[transcriber] Loading model {cfg.model} on {cfg.device}...")
        model = load_whisperx_asr_model(whisperx, cfg, whisper_task, whisper_language)
        print(f"[transcriber] Loading audio: {input_path}")
        audio = whisperx.load_audio(str(input_path))

        result, detected_language = _transcribe_with_whisperx(
            model, cfg, audio, whisper_task, whisper_language
        )
        align_language = (
            "en"
            if cfg.translate_to_english
            else (detected_language or whisper_language or "en")
        )
        result = _align_whisperx_result(whisperx, cfg, audio, result, align_language)
        if diarize:
            result = _diarize_whisperx_result(whisperx, cfg, audio, result, hf_token)

        apply_confidence_cleanup(result, cfg)
        print("[transcriber] Writing subtitle-sized SRT from in-memory timings...")
        write_direct_srt_from_result(result, srt_path, cfg)
        return detected_language


# Compatibility alias retained for callers that used the former private worker.
_run_whisperx_direct = run_whisperx_direct


def run_whisperx_direct_logged(
    cfg: RunConfig,
    input_path: Path,
    srt_path: Path,
    hf_token: str | None,
    diarize: bool,
    log_path: Path,
    append: bool = False,
) -> tuple[int, str | None]:
    mode = "a" if append else "w"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open(mode, encoding="utf-8", errors="ignore") as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    detected_language = run_whisperx_direct(
                        cfg, input_path, srt_path, hf_token, diarize
                    )
                    if detected_language is None:
                        detected_language = parse_detected_language_from_log(log_path)
                    return_code = 0
                except Exception:
                    import traceback

                    traceback.print_exc()
                    return_code = 1
                    detected_language = None
        return return_code, detected_language
    finally:
        cleanup_after_transcription_run(cfg)


# Transcription orchestration


def _translation_summary(cfg: RunConfig) -> str:
    if cfg.language == "auto":
        return (
            "whisperx (direct English output)"
            if cfg.translate_to_english
            else "auto (Spanish -> English when detected)"
        )
    return "whisperx direct" if cfg.translate_to_english else "off"


def _decode_summary(cfg: RunConfig) -> str:
    temperatures = cfg.temperature_schedule or (cfg.temperature,)
    return (
        f"temps={','.join(f'{temperature:g}' for temperature in temperatures)} "
        f"best_of={cfg.best_of if cfg.best_of is not None else 'auto'} "
        f"cr={cfg.compression_ratio_threshold if cfg.compression_ratio_threshold is not None else 'auto'} "
        f"logprob={cfg.logprob_threshold if cfg.logprob_threshold is not None else 'auto'} "
        f"no_speech={cfg.no_speech_threshold if cfg.no_speech_threshold is not None else 'auto'} "
        f"prev_text={'on' if cfg.condition_on_previous_text else 'off'}"
    )


def print_summary(
    cfg: RunConfig, input_path: Path, outputs: OutputPaths, report: Reporter = print
) -> None:
    cleanup = "on" if cfg.confidence_cleanup else "off"
    if cfg.confidence_cleanup:
        cleanup += f" ({cfg.confidence_cleanup_mode})"

    lines = [
        f'Input:     "{input_path}"',
        f'Output:    "{outputs.srt_path}"',
        f'LLM:       "{outputs.llm_path}"',
        f'Log:       "{outputs.log_path}"',
        f'Lock:      "{outputs.lock_path}"',
        f'OutDir:    "{outputs.output_dir}"',
        f"Lang:      {cfg.language}",
        f"Translate: {_translation_summary(cfg)}",
        f"Mode:      {cfg.mode}",
        f"Model:     {cfg.model}",
        f"Diarize:   {'on' if cfg.diarize else 'off'}",
        f"Smooth:    {'on' if cfg.diarize and cfg.diarize_smoothing else 'off'}",
        f"Cleanup:   {cleanup}",
    ]
    if cfg.glossary_path:
        lines.append(f'Glossary:  "{cfg.glossary_path}"')
    if cfg.asr_prompt:
        lines.append("ASRPrompt: on")
    lines.append(f"Decode:    {_decode_summary(cfg)}")
    if cfg.dry_run:
        lines.append("DryRun:    on")
    report_block(report, *lines)


def describe_dry_run_plan(
    cfg: RunConfig, input_path: Path, outputs: OutputPaths, report: Reporter
) -> None:
    report_lines(
        report,
        "Dry run. No files will be changed.",
        f'Would acquire lock: "{outputs.lock_path}"',
        f'Would run WhisperX with output dir: "{outputs.output_dir}"',
        f'Would write transcript: "{outputs.srt_path}"',
        f'Would write LLM prompt file: "{outputs.llm_path}"',
        f'Would write log: "{outputs.log_path}"',
        f'Would leave the source file in place: "{input_path}"',
    )


def _append_diarization_retry_log(log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8", errors="ignore") as log:
        log.write(
            "\n[transcriber] Diarization unavailable or blocked; "
            "retrying without --diarize.\n"
        )


def _run_transcription_attempts(
    cfg: RunConfig,
    input_path: Path,
    outputs: OutputPaths,
    hf_token: str | None,
    report: Reporter,
) -> TranscriptionAttemptResult:
    diarize = cfg.diarize
    attempt = 0

    while True:
        return_code, detected_language = run_whisperx_direct_logged(
            cfg,
            input_path,
            outputs.srt_path,
            hf_token,
            diarize=diarize,
            log_path=outputs.log_path,
            append=attempt > 0,
        )
        diarization_failed = diarize and should_fallback_without_diarization(
            outputs.log_path
        )
        if return_code == 0:
            return TranscriptionAttemptResult(
                return_code=0,
                detected_language=detected_language,
                fell_back_from_diarization=diarization_failed,
            )
        if not diarization_failed:
            return TranscriptionAttemptResult(
                return_code=return_code,
                detected_language=detected_language,
                fell_back_from_diarization=False,
            )

        report_lines(
            report,
            "",
            "Diarization unavailable or blocked. Retrying without diarization...",
        )
        _append_diarization_retry_log(outputs.log_path)
        diarize = False
        attempt += 1


def _maybe_translate_auto_spanish(
    cfg: RunConfig,
    outputs: OutputPaths,
    detected_language: str | None,
    report: Reporter,
) -> None:
    if cfg.language == "auto":
        report(f"Detected language: {detected_language or 'unknown'}.")

    should_translate = (
        not cfg.translate_to_english
        and cfg.language == "auto"
        and detected_language == "es"
    )
    if not should_translate:
        return

    if cfg.glossary_path and not Path(cfg.glossary_path).expanduser().exists():
        report(f'Glossary file not found: "{cfg.glossary_path}"')
    report("Translating Spanish transcript to English text...")
    try:
        translate_srt_to_english(
            outputs.srt_path,
            cfg.device,
            glossary=cfg.glossary,
            glossary_spec=cfg.glossary_path,
            context_window=cfg.translation_context_window,
            batch_size=cfg.translation_batch_size,
            num_beams=cfg.translation_num_beams,
            max_new_tokens=cfg.translation_max_new_tokens,
            no_repeat_ngram_size=cfg.translation_no_repeat_ngram_size,
            log_path=outputs.log_path,
        )
    except Exception as exc:
        report_block(
            report,
            f"Translation failed: {exc}",
            "Keeping the original transcript text.",
        )


def _finalize_existing_transcript(
    cfg: RunConfig,
    outputs: OutputPaths,
    detected_language: str | None,
    report: Reporter,
) -> None:
    if not outputs.srt_path.exists():
        return
    _maybe_translate_auto_spanish(cfg, outputs, detected_language, report)
    with contextlib.suppress(Exception):
        finalize_transcript_outputs(outputs.srt_path, outputs.llm_path)


def _report_transcription_outcome(
    cfg: RunConfig,
    outputs: OutputPaths,
    result: TranscriptionAttemptResult,
    report: Reporter,
) -> int:
    if result.return_code != 0:
        report_block(
            report,
            f"WhisperX failed (exit code {result.return_code}).",
            f'See the log: "{outputs.log_path}"',
        )
        return result.return_code

    if not outputs.srt_path.exists():
        report_block(
            report,
            "Done, but SRT not found where expected:",
            f'  "{outputs.srt_path}"',
            "Check the log:",
            f'  "{outputs.log_path}"',
        )
        return 1

    lines = [
        "Done.",
        f'SRT: "{outputs.srt_path}"',
        f'LLM: "{outputs.llm_path}"',
    ]
    if cfg.mode == "fast" and not cfg.diarize:
        lines.append("Note: fast mode used (speaker diarization disabled).")
    if result.fell_back_from_diarization:
        lines.extend(
            [
                "Note: completed without speaker diarization.",
                "To enable diarization, accept terms with the SAME HF account at:",
                "  https://hf.co/pyannote/speaker-diarization-3.1",
                "  https://hf.co/pyannote/segmentation-3.0",
            ]
        )
    report_block(report, *lines)
    return 0


def transcribe_file(
    cfg: RunConfig,
    input_path: Path,
    report: Reporter = print,
    stale_lock_seconds: float = DEFAULT_STALE_LOCK_SECONDS,
) -> int:
    error = input_path_error(input_path)
    if error is not None:
        report_block(report, error)
        return 1

    outputs = output_paths_for_input(input_path, cfg, create_dirs=not cfg.dry_run)
    print_summary(cfg, input_path, outputs, report=report)
    if cfg.dry_run:
        describe_dry_run_plan(cfg, input_path, outputs, report)
        return 0

    try:
        ensure_ffmpeg_available_for_child_processes()
    except RuntimeError as exc:
        report_block(report, str(exc))
        return 1

    if not acquire_lock(input_path, outputs.lock_path, stale_lock_seconds, report):
        return 0

    try:
        hf_token = load_hf_token(project_dir()) if cfg.diarize else None
        if cfg.diarize and not hf_token:
            report_block(
                report,
                "Missing Hugging Face token.",
                f'Create "{project_dir() / "hf_token.txt"}" '
                '(or "HF_TOKEN.txt") or set HF_TOKEN.',
            )
            return 1

        with tempfile.TemporaryDirectory(prefix="transcriber-audio-") as temp_dir:
            prepared_input = preprocess_audio_for_whisperx(
                input_path, Path(temp_dir), report=report
            )
            if prepared_input != input_path:
                report(f'Audio preprocessing: "{prepared_input}"')
            result = _run_transcription_attempts(
                cfg, prepared_input, outputs, hf_token, report
            )

        _finalize_existing_transcript(cfg, outputs, result.detected_language, report)
        return _report_transcription_outcome(cfg, outputs, result, report)
    finally:
        release_lock(outputs.lock_path, report=report)


# Watch mode and command-line entry point


def watcher_log(log_path: Path, message: str) -> None:
    stamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    try:
        print(stamped, flush=True)
    except Exception:
        pass
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8", errors="ignore") as log:
            log.write(stamped + "\n")
    except Exception:
        pass


def make_watch_reporter(log_path: Path) -> Reporter:
    def report(message: str) -> None:
        if not message.strip():
            return
        for line in message.splitlines():
            if line.strip():
                watcher_log(log_path, line)

    return report


def is_watchable_media(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS


def input_path_error(input_path: Path) -> str | None:
    if not input_path.exists():
        return f'File not found: "{input_path}"'
    if not input_path.is_file():
        return f'Input is not a file: "{input_path}"'
    if input_path.suffix.lower() not in MEDIA_EXTENSIONS:
        supported = ", ".join(sorted(MEDIA_EXTENSIONS))
        return f'Unsupported media file: "{input_path}" (supported: {supported})'
    return None


def iter_watch_candidates(watch_dir: Path) -> Iterable[Path]:
    try:
        candidates = sorted(watch_dir.iterdir(), key=lambda path: path.name.lower())
    except FileNotFoundError:
        return
    for path in candidates:
        if is_watchable_media(path):
            yield path


def file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def needs_transcription(input_path: Path, cfg: RunConfig) -> bool:
    outputs = output_paths_for_input(input_path, cfg)
    if not outputs.srt_path.exists():
        return True
    try:
        input_mtime_ns = input_path.stat().st_mtime_ns
        output_mtime_ns = outputs.srt_path.stat().st_mtime_ns
    except OSError:
        return True
    return input_mtime_ns > output_mtime_ns


def _process_watch_candidate(
    cfg: RunConfig,
    path: Path,
    key: str,
    pending: dict[str, PendingWatchFile],
    now: float,
    settle_seconds: float,
    report: Reporter,
) -> None:
    if not needs_transcription(path, cfg):
        pending.pop(key, None)
        return

    try:
        size, mtime_ns = file_signature(path)
    except OSError:
        pending.pop(key, None)
        return

    pending_file = pending.get(key)
    if pending_file is None:
        pending[key] = PendingWatchFile(size=size, mtime_ns=mtime_ns, stable_since=now)
        report(f'Detected "{path.name}". Waiting for the file to settle.')
        return

    if pending_file.update_signature(size, mtime_ns, now):
        return
    if not pending_file.is_ready(now, settle_seconds):
        return

    pending_file.last_attempt_at = now
    report(f'Starting transcription for "{path.name}".')
    return_code = transcribe_file(cfg, path, report=report)
    outputs = output_paths_for_input(path, cfg)
    if return_code == 0 and outputs.srt_path.exists():
        pending.pop(key, None)
        report(f'Finished "{path.name}" -> "{outputs.srt_path.name}".')
        return

    report(
        f'Failed "{path.name}". Will retry in '
        f"{WATCH_RETRY_COOLDOWN_SECONDS:g}s if the transcript is still missing."
    )


def _scan_watch_directory(
    cfg: RunConfig,
    watch_dir: Path,
    pending: dict[str, PendingWatchFile],
    settle_seconds: float,
    report: Reporter,
) -> None:
    now = time.monotonic()
    current_paths: set[str] = set()
    for path in iter_watch_candidates(watch_dir):
        key = str(path.resolve())
        current_paths.add(key)
        _process_watch_candidate(cfg, path, key, pending, now, settle_seconds, report)

    for missing_key in pending.keys() - current_paths:
        pending.pop(missing_key, None)


def run_watch_loop(
    cfg: RunConfig,
    watch_dir: Path,
    poll_interval: float,
    settle_seconds: float,
) -> int:
    watch_dir = watch_dir.expanduser()
    try:
        watch_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(
            "\nCould not create or access watch directory:\n"
            f'  "{watch_dir}"\n'
            f"  {exc}\n"
        )
        return 1

    report = make_watch_reporter(project_dir() / LOG_DIR_NAME / WATCHER_LOG_NAME)
    report(f'Watcher started for "{watch_dir}".')
    report(
        "Defaults: "
        f"lang={cfg.language}, mode={cfg.mode}, model={cfg.model}, "
        f'diarize={"on" if cfg.diarize else "off"}, device={cfg.device}, '
        f"compute_type={cfg.compute_type}."
    )
    report(
        f"Polling every {poll_interval:g}s. A file must stay unchanged for "
        f"{settle_seconds:g}s before transcription starts."
    )

    pending: dict[str, PendingWatchFile] = {}
    while True:
        try:
            _scan_watch_directory(cfg, watch_dir, pending, settle_seconds, report)
            time.sleep(poll_interval)
        except KeyboardInterrupt:
            report("Watcher stopped.")
            return 0
        except Exception as exc:
            report(f"Watcher error: {exc}")
            time.sleep(poll_interval)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(list(argv) if argv is not None else sys.argv[1:])

    if args.watch and args.input:
        print("\nUse --watch-dir with --watch instead of --input.\n")
        return 2
    if args.poll_interval <= 0:
        print("\n--poll-interval must be greater than 0.\n")
        return 2
    if args.settle_seconds < 0:
        print("\n--settle-seconds must be 0 or greater.\n")
        return 2

    try:
        configure_temp_dir(project_dir())
    except RuntimeError as exc:
        print(f"\n{exc}\n")
        return 1

    cfg = build_config(args, interactive=not args.watch)

    if args.watch:
        return run_watch_loop(
            cfg=cfg,
            watch_dir=Path(args.watch_dir),
            poll_interval=args.poll_interval,
            settle_seconds=args.settle_seconds,
        )

    input_path = resolve_input_path(args.input)
    if input_path is None:
        print("\nNo file selected.\n")
        return 0

    return transcribe_file(cfg, input_path)


if __name__ == "__main__":
    raise SystemExit(main())
