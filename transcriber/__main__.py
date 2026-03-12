from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import re
import runpy
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Sequence


MEDIA_FILTER = (
    "Audio/Video",
    "*.wav *.mp3 *.m4a *.flac *.aac *.ogg *.wma *.mp4 *.mov *.mkv *.webm",
)
MEDIA_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".aac",
    ".ogg",
    ".wma",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
}
DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_SETTLE_SECONDS = 5.0
DEFAULT_WATCH_DIR = Path.home() / "OneDrive" / "recordings"
WATCH_RETRY_COOLDOWN_SECONDS = 300.0
WATCHER_LOG_NAME = "transcriber-watcher.log"
FALLBACK_TEMP_DIR_NAME = ".tmp_transcriber_temp"
SHARED_OUTPUT_DIR_NAME = "shared"


@dataclass
class LegacyOptions:
    language: str | None = None
    language_locked: bool = False
    task: str | None = None
    task_locked: bool = False
    mode: str | None = None
    mode_locked: bool = False
    model: str | None = None
    model_locked: bool = False


@dataclass
class RunConfig:
    language: str
    task: str
    mode: str
    model: str
    batch_size: int
    beam_size: int
    patience: float
    temperature: float
    diarize: bool
    device: str
    compute_type: str


@dataclass
class OutputPaths:
    srt_path: Path
    llm_path: Path
    log_path: Path


@dataclass
class PendingWatchFile:
    size: int
    mtime_ns: int
    stable_since: float
    last_attempt_at: float | None = None


Reporter = Callable[[str], None]


UNSUPPORTED_GLOBAL_RE = re.compile(r"Unsupported global: GLOBAL ([A-Za-z0-9_\.]+)")
DIARIZATION_FALLBACK_PATTERNS = (
    "could not download 'pyannote/speaker-diarization-3.1' pipeline.",
    "visit https://hf.co/pyannote/speaker-diarization-3.1 to accept the user conditions.",
    "attributeerror: 'nonetype' object has no attribute 'to'",
    "unpicklingerror",
    "unsupported global: global",
)
ALIGNMENT_FALLBACK_PATTERNS = (
    "performing alignment...",
    "punkt_tab",
    "nltk/tokenize/punkt.py",
    "nltk/tabdata.py",
    "attributeerror: 'bool' object has no attribute 'split'",
)


def project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


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
        "Could not create a usable temporary directory for WhisperX. "
        "Check TMP/TEMP permissions."
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WhisperX one-click transcriber (quality + fast modes)."
    )
    parser.add_argument(
        "legacy",
        nargs="*",
        help="Legacy tokens: [language] [model] [task] [mode]",
    )
    parser.add_argument("--input", "-i", help="Audio/video file path.")
    parser.add_argument("--lang", choices=("en", "es"), help="Language: en or es.")
    parser.add_argument(
        "--task",
        choices=("transcribe", "translate"),
        help="WhisperX task: transcribe or translate.",
    )
    parser.add_argument(
        "--mode",
        choices=("quality", "fast"),
        help="Run mode preset: quality or fast.",
    )
    parser.add_argument("--model", help="Whisper model name, e.g. large-v3, medium.")
    parser.add_argument("--device", default="cuda", help="Inference device (default: cuda).")
    parser.add_argument(
        "--compute-type",
        default="float16",
        choices=("float16", "float32", "int8"),
        help="Computation dtype (default: float16).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously watch a folder and transcribe new media files.",
    )
    parser.add_argument(
        "--watch-dir",
        default=str(DEFAULT_WATCH_DIR),
        help=f'Folder to watch when using --watch (default: "{DEFAULT_WATCH_DIR}").',
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help=f"Seconds between watch scans (default: {DEFAULT_POLL_INTERVAL:g}).",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=DEFAULT_SETTLE_SECONDS,
        help=(
            "How long a file must stay unchanged before watch mode starts transcription "
            f"(default: {DEFAULT_SETTLE_SECONDS:g})."
        ),
    )
    diarize_group = parser.add_mutually_exclusive_group()
    diarize_group.add_argument(
        "--diarize",
        dest="force_diarize",
        action="store_true",
        help="Force diarization on.",
    )
    diarize_group.add_argument(
        "--no-diarize",
        dest="force_no_diarize",
        action="store_true",
        help="Force diarization off.",
    )
    return parser.parse_args(argv)


def parse_legacy(tokens: Iterable[str]) -> LegacyOptions:
    opt = LegacyOptions()
    for token in tokens:
        t = token.strip()
        if not t:
            continue
        low = t.lower()
        if low in {"e", "en"}:
            opt.language = "en"
            opt.language_locked = True
            continue
        if low in {"s", "es"}:
            opt.language = "es"
            opt.language_locked = True
            continue
        if low in {"t", "tr", "translate"}:
            opt.task = "translate"
            opt.task_locked = True
            continue
        if low == "transcribe":
            opt.task = "transcribe"
            opt.task_locked = True
            continue
        if low in {"f", "fast"}:
            opt.mode = "fast"
            opt.mode_locked = True
            continue
        if low in {"q", "quality"}:
            opt.mode = "quality"
            opt.mode_locked = True
            continue
        if not opt.model_locked:
            opt.model = t
            opt.model_locked = True
    return opt


def pick_media_file() -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    try:
        path = filedialog.askopenfilename(
            title="Select an audio/video file",
            filetypes=[MEDIA_FILTER, ("All files", "*.*")],
        )
    finally:
        root.destroy()
    return path or None


def prompt_input_path() -> str | None:
    if not sys.stdin.isatty():
        return None
    raw = input("Media file path (or Enter to cancel): ").strip()
    return raw or None


def prompt_language(current: str) -> str:
    if not sys.stdin.isatty():
        return current
    choice = input("Language [Enter=English, s=Spanish]: ").strip().lower()
    if choice in {"s", "es"}:
        return "es"
    if choice in {"e", "en", ""}:
        return "en"
    return current


def prompt_spanish_task(current: str) -> str:
    if not sys.stdin.isatty():
        return current
    choice = input("Output [Enter=Spanish transcript, t=translate to English]: ").strip().lower()
    if choice in {"t", "tr", "translate"}:
        return "translate"
    return "transcribe"


def prompt_mode(current: str) -> str:
    if not sys.stdin.isatty():
        return current
    choice = input("Run mode [Enter=quality, f=fast]: ").strip().lower()
    if choice in {"f", "fast"}:
        return "fast"
    if choice in {"q", "quality", ""}:
        return "quality"
    return current


def build_config(args: argparse.Namespace, interactive: bool = True) -> RunConfig:
    legacy = parse_legacy(args.legacy)

    language = args.lang or legacy.language or "en"
    language_locked = bool(args.lang) or legacy.language_locked

    task = args.task or legacy.task or "transcribe"
    task_locked = bool(args.task) or legacy.task_locked

    mode = args.mode or legacy.mode or "quality"
    mode_locked = bool(args.mode) or legacy.mode_locked

    model = args.model or legacy.model
    model_locked = bool(args.model) or legacy.model_locked

    if interactive and not language_locked:
        language = prompt_language(language)
    if interactive and language == "es" and not task_locked:
        task = prompt_spanish_task(task)
    if interactive and not mode_locked:
        mode = prompt_mode(mode)

    if mode == "fast":
        if not model_locked:
            model = "medium"
        batch_size = 16
        beam_size = 2
        patience = 1.0
        temperature = 0.0
        diarize_default = False
    else:
        if not model_locked:
            model = "large-v3"
        batch_size = 8
        beam_size = 8
        patience = 1.2
        temperature = 0.0
        diarize_default = True

    if args.force_diarize:
        diarize = True
    elif args.force_no_diarize:
        diarize = False
    else:
        diarize = diarize_default

    return RunConfig(
        language=language,
        task=task,
        mode=mode,
        model=model or "large-v3",
        batch_size=batch_size,
        beam_size=beam_size,
        patience=patience,
        temperature=temperature,
        diarize=diarize,
        device=args.device,
        compute_type=args.compute_type,
    )


def resolve_input_path(arg_input: str | None) -> Path | None:
    if arg_input:
        p = Path(arg_input).expanduser()
        return p if p.exists() else p

    chosen = pick_media_file()
    if chosen:
        return Path(chosen)

    manual = prompt_input_path()
    if manual:
        return Path(manual).expanduser()

    return None


def output_paths_for_input(input_path: Path) -> OutputPaths:
    output_dir = input_path.parent / SHARED_OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    base = input_path.stem
    return OutputPaths(
        srt_path=output_dir / f"{base}.srt",
        llm_path=output_dir / f"{base}_llm.txt",
        log_path=output_dir / f"{base}_whisperx.log",
    )


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


def add_common_safe_globals() -> None:
    try:
        import torch
    except Exception:
        return

    safe: list[object] = []
    try:
        import omegaconf

        safe.extend([omegaconf.listconfig.ListConfig, omegaconf.dictconfig.DictConfig])
    except Exception:
        pass

    try:
        from torch.torch_version import TorchVersion

        safe.append(TorchVersion)
    except Exception:
        pass

    try:
        from pyannote.audio.core.task import Problem, Resolution, Specifications

        safe.extend([Specifications, Problem, Resolution])
    except Exception:
        pass

    if safe:
        torch.serialization.add_safe_globals(safe)


def try_add_unsupported_global(exc: Exception) -> bool:
    try:
        import torch
    except Exception:
        return False

    match = UNSUPPORTED_GLOBAL_RE.search(str(exc))
    if not match:
        return False

    qualname = match.group(1)
    if "." not in qualname:
        return False
    module_name, object_name = qualname.rsplit(".", 1)
    try:
        obj = getattr(importlib.import_module(module_name), object_name)
    except Exception:
        return False

    torch.serialization.add_safe_globals([obj])
    return True


def run_whisperx_module(argv: list[str], max_retries: int = 12) -> int:
    add_common_safe_globals()

    for _ in range(max_retries):
        saved_argv = sys.argv[:]
        sys.argv = ["whisperx", *argv]
        try:
            runpy.run_module("whisperx", run_name="__main__")
            return 0
        except SystemExit as exc:
            return int(exc.code) if isinstance(exc.code, int) else 0
        except Exception as exc:
            if try_add_unsupported_global(exc):
                continue
            raise
        finally:
            sys.argv = saved_argv
    return 1


def run_whisperx_logged(argv: list[str], log_path: Path, append: bool = False) -> int:
    mode = "a" if append else "w"
    with log_path.open(mode, encoding="utf-8", errors="ignore") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                return run_whisperx_module(argv)
            except Exception:
                import traceback

                traceback.print_exc()
                return 1


def build_whisperx_args(
    cfg: RunConfig,
    input_path: Path,
    output_dir: Path,
    hf_token: str | None,
    diarize: bool,
    no_align: bool = False,
) -> list[str]:
    args = [
        str(input_path),
        "--device",
        cfg.device,
        "--compute_type",
        cfg.compute_type,
        "--model",
        cfg.model,
        "--batch_size",
        str(cfg.batch_size),
        "--temperature",
        str(cfg.temperature),
        "--beam_size",
        str(cfg.beam_size),
        "--patience",
        str(cfg.patience),
        "--task",
        cfg.task,
        "--language",
        cfg.language,
        "--vad_method",
        "silero",
        "--output_dir",
        str(output_dir),
        "--output_format",
        "srt",
    ]
    if no_align:
        args.append("--no_align")
    if diarize:
        args.extend(["--diarize", "--hf_token", hf_token or ""])
    return args


def should_fallback_without_diarization(log_path: Path) -> bool:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False
    return any(pattern in text for pattern in DIARIZATION_FALLBACK_PATTERNS)


def should_fallback_without_alignment(log_path: Path) -> bool:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore").lower().replace("\\", "/")
    except Exception:
        return False
    if ALIGNMENT_FALLBACK_PATTERNS[0] not in text:
        return False
    return any(pattern in text for pattern in ALIGNMENT_FALLBACK_PATTERNS[1:])


def extract_srt_text_lines(srt_path: Path) -> list[str]:
    lines = srt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    text_lines: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.isdigit():
            continue
        if "-->" in s and "," in s:
            continue
        text_lines.append(s)
    return text_lines


def build_llm_file(srt_path: Path, llm_path: Path) -> None:
    if not srt_path.exists():
        return

    text_lines = extract_srt_text_lines(srt_path)
    preface = (
        "You are given an automatic transcript.\n"
        "Preserve speaker labels if present.\n"
        "Tasks: (1) concise summary, (2) key claims and evidence, (3) action items, "
        "(4) names and affiliations, (5) open questions, (6) glossary of technical terms.\n"
        "If something sounds uncertain, mark it as uncertain.\n\n"
        "TRANSCRIPT:\n"
    )
    llm_path.write_text(preface + "\n".join(text_lines), encoding="utf-8")


def print_summary(
    cfg: RunConfig,
    input_path: Path,
    outputs: OutputPaths,
    report: Reporter = print,
) -> None:
    report("")
    report(f'Input:  "{input_path}"')
    report(f'Output: "{outputs.srt_path}"')
    report(f'LLM:    "{outputs.llm_path}"')
    report(f'Log:    "{outputs.log_path}"')
    report(f'OutDir: "{input_path.parent}"')
    report(f"Lang:   {cfg.language}")
    report(f"Task:   {cfg.task}")
    report(f"Mode:   {cfg.mode}")
    report(f"Model:  {cfg.model}")
    report(f'Diarize: {"on" if cfg.diarize else "off"}')
    report("")


def transcribe_file(cfg: RunConfig, input_path: Path, report: Reporter = print) -> int:
    if not input_path.exists():
        report("")
        report(f'File not found: "{input_path}"')
        report("")
        return 1

    outputs = output_paths_for_input(input_path)
    print_summary(cfg, input_path, outputs, report=report)

    hf_token: str | None = None
    if cfg.diarize:
        hf_token = load_hf_token(project_dir())
        if not hf_token:
            report("")
            report("Missing Hugging Face token.")
            report(
                f'Create "{project_dir() / "hf_token.txt"}" (or "HF_TOKEN.txt") or set HF_TOKEN.'
            )
            report("")
            return 1

    fallback_no_diarize = False
    fallback_no_align = False
    current_diarize = cfg.diarize
    current_no_align = False
    attempt = 0

    while True:
        run_args = build_whisperx_args(
            cfg,
            input_path,
            outputs.srt_path.parent,
            hf_token,
            diarize=current_diarize,
            no_align=current_no_align,
        )
        rc = run_whisperx_logged(run_args, outputs.log_path, append=attempt > 0)
        if rc == 0:
            break

        if current_diarize and should_fallback_without_diarization(outputs.log_path):
            fallback_no_diarize = True
            current_diarize = False
            report("")
            report("Diarization unavailable or blocked. Retrying without diarization...")
            with outputs.log_path.open("a", encoding="utf-8", errors="ignore") as log:
                log.write(
                    "\n[transcriber] Diarization unavailable or blocked; retrying without --diarize.\n"
                )
            attempt += 1
            continue

        if not current_no_align and should_fallback_without_alignment(outputs.log_path):
            fallback_no_align = True
            current_no_align = True
            report("")
            report("WhisperX alignment failed. Retrying without alignment...")
            with outputs.log_path.open("a", encoding="utf-8", errors="ignore") as log:
                log.write(
                    "\n[transcriber] Alignment failed; retrying with --no_align.\n"
                )
            attempt += 1
            continue

        break

    if outputs.srt_path.exists():
        try:
            build_llm_file(outputs.srt_path, outputs.llm_path)
        except Exception:
            pass

    if rc != 0:
        report("")
        report(f"WhisperX failed (exit code {rc}).")
        report(f'See the log: "{outputs.log_path}"')
        report("")
        return rc

    if outputs.srt_path.exists():
        report("")
        report("Done.")
        report(f'SRT: "{outputs.srt_path}"')
        report(f'LLM: "{outputs.llm_path}"')
        if cfg.mode == "fast" and not cfg.diarize:
            report("Note: fast mode used (speaker diarization disabled).")
        if fallback_no_diarize:
            report("Note: completed without speaker diarization.")
            report("To enable diarization, accept terms with the SAME HF account at:")
            report("  https://hf.co/pyannote/speaker-diarization-3.1")
            report("  https://hf.co/pyannote/segmentation-3.0")
        if fallback_no_align:
            report("Note: completed without WhisperX alignment.")
            report("Timestamps may be less precise than the default aligned output.")
        report("")
        return 0

    report("")
    report("Done, but SRT not found where expected:")
    report(f'  "{outputs.srt_path}"')
    report("Check the log:")
    report(f'  "{outputs.log_path}"')
    report("")
    return 0


def watcher_log(log_path: Path, message: str) -> None:
    stamped = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    try:
        print(stamped, flush=True)
    except Exception:
        pass
    try:
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


def iter_watch_candidates(watch_dir: Path) -> Iterable[Path]:
    try:
        candidates = sorted(watch_dir.iterdir(), key=lambda path: path.name.lower())
    except FileNotFoundError:
        return
    for path in candidates:
        if is_watchable_media(path):
            yield path


def needs_transcription(input_path: Path) -> bool:
    outputs = output_paths_for_input(input_path)
    if not outputs.srt_path.exists():
        return True
    try:
        input_mtime_ns = input_path.stat().st_mtime_ns
        output_mtime_ns = outputs.srt_path.stat().st_mtime_ns
    except OSError:
        return True
    return input_mtime_ns > output_mtime_ns


def file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


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
    watcher_log_path = watch_dir / WATCHER_LOG_NAME
    report = make_watch_reporter(watcher_log_path)
    report(f'Watcher started for "{watch_dir}".')
    report(
        "Defaults: "
        f"lang={cfg.language}, task={cfg.task}, mode={cfg.mode}, model={cfg.model}, "
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
            current_paths: set[str] = set()
            now = time.monotonic()

            for path in iter_watch_candidates(watch_dir):
                key = str(path.resolve())
                current_paths.add(key)

                if not needs_transcription(path):
                    pending.pop(key, None)
                    continue

                try:
                    size, mtime_ns = file_signature(path)
                except OSError:
                    pending.pop(key, None)
                    continue

                pending_file = pending.get(key)
                if pending_file is None:
                    pending[key] = PendingWatchFile(
                        size=size,
                        mtime_ns=mtime_ns,
                        stable_since=now,
                    )
                    report(f'Detected "{path.name}". Waiting for the file to settle.')
                    continue

                if pending_file.size != size or pending_file.mtime_ns != mtime_ns:
                    pending_file.size = size
                    pending_file.mtime_ns = mtime_ns
                    pending_file.stable_since = now
                    pending_file.last_attempt_at = None
                    continue

                if now - pending_file.stable_since < settle_seconds:
                    continue

                if pending_file.last_attempt_at is not None:
                    if now - pending_file.last_attempt_at < WATCH_RETRY_COOLDOWN_SECONDS:
                        continue

                pending_file.last_attempt_at = now
                report(f'Starting transcription for "{path.name}".')
                rc = transcribe_file(cfg, path, report=report)
                outputs = output_paths_for_input(path)
                if rc == 0 and outputs.srt_path.exists():
                    pending.pop(key, None)
                    report(f'Finished "{path.name}" -> "{outputs.srt_path.name}".')
                else:
                    report(
                        f'Failed "{path.name}". Will retry in '
                        f"{WATCH_RETRY_COOLDOWN_SECONDS:g}s if the transcript is still missing."
                    )

            for key in list(pending):
                if key not in current_paths:
                    pending.pop(key, None)

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
