from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import re
import runpy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


MEDIA_FILTER = (
    "Audio/Video",
    "*.wav *.mp3 *.m4a *.flac *.aac *.ogg *.wma *.mp4 *.mov *.mkv *.webm",
)


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


UNSUPPORTED_GLOBAL_RE = re.compile(r"Unsupported global: GLOBAL ([A-Za-z0-9_\.]+)")
DIARIZATION_FALLBACK_PATTERNS = (
    "could not download 'pyannote/speaker-diarization-3.1' pipeline.",
    "visit https://hf.co/pyannote/speaker-diarization-3.1 to accept the user conditions.",
    "attributeerror: 'nonetype' object has no attribute 'to'",
    "unpicklingerror",
    "unsupported global: global",
)


def project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


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


def build_config(args: argparse.Namespace) -> RunConfig:
    legacy = parse_legacy(args.legacy)

    language = args.lang or legacy.language or "en"
    language_locked = bool(args.lang) or legacy.language_locked

    task = args.task or legacy.task or "transcribe"
    task_locked = bool(args.task) or legacy.task_locked

    mode = args.mode or legacy.mode or "quality"
    mode_locked = bool(args.mode) or legacy.mode_locked

    model = args.model or legacy.model
    model_locked = bool(args.model) or legacy.model_locked

    if not language_locked:
        language = prompt_language(language)
    if language == "es" and not task_locked:
        task = prompt_spanish_task(task)
    if not mode_locked:
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
    cfg: RunConfig, input_path: Path, output_dir: Path, hf_token: str | None, diarize: bool
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
    if diarize:
        args.extend(["--diarize", "--hf_token", hf_token or ""])
    return args


def should_fallback_without_diarization(log_path: Path) -> bool:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False
    return any(pattern in text for pattern in DIARIZATION_FALLBACK_PATTERNS)


def build_llm_file(srt_path: Path, llm_path: Path) -> None:
    if not srt_path.exists():
        return

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
    srt_path: Path,
    llm_path: Path,
    log_path: Path,
) -> None:
    print()
    print(f'Input:  "{input_path}"')
    print(f'Output: "{srt_path}"')
    print(f'LLM:    "{llm_path}"')
    print(f'Log:    "{log_path}"')
    print(f'OutDir: "{input_path.parent}"')
    print(f"Lang:   {cfg.language}")
    print(f"Task:   {cfg.task}")
    print(f"Mode:   {cfg.mode}")
    print(f"Model:  {cfg.model}")
    print(f'Diarize: {"on" if cfg.diarize else "off"}')
    print()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(list(argv) if argv is not None else sys.argv[1:])
    cfg = build_config(args)

    input_path = resolve_input_path(args.input)
    if input_path is None:
        print("\nNo file selected.\n")
        return 0
    if not input_path.exists():
        print(f'\nFile not found:\n  "{input_path}"\n')
        return 1

    output_dir = input_path.parent
    base = input_path.stem
    srt_path = output_dir / f"{base}.srt"
    llm_path = output_dir / f"{base}_llm.txt"
    log_path = output_dir / f"{base}_whisperx.log"

    print_summary(cfg, input_path, srt_path, llm_path, log_path)

    hf_token: str | None = None
    if cfg.diarize:
        hf_token = load_hf_token(project_dir())
        if not hf_token:
            print(
                "\nMissing Hugging Face token.\n"
                f'Create "{project_dir() / "hf_token.txt"}" (or "HF_TOKEN.txt") '
                "or set HF_TOKEN in your environment.\n"
            )
            return 1

    run_args = build_whisperx_args(cfg, input_path, output_dir, hf_token, cfg.diarize)
    rc = run_whisperx_logged(run_args, log_path, append=False)

    fallback_no_diarize = False
    if cfg.diarize and rc != 0 and should_fallback_without_diarization(log_path):
        fallback_no_diarize = True
        print("\nDiarization unavailable or blocked. Retrying without diarization...")
        with log_path.open("a", encoding="utf-8", errors="ignore") as log:
            log.write(
                "\n[transcriber] Diarization unavailable or blocked; retrying without --diarize.\n"
            )
        fallback_args = build_whisperx_args(cfg, input_path, output_dir, hf_token, diarize=False)
        rc = run_whisperx_logged(fallback_args, log_path, append=True)

    if srt_path.exists():
        try:
            build_llm_file(srt_path, llm_path)
        except Exception:
            pass

    if rc != 0:
        print(f"\nWhisperX failed (exit code {rc}).")
        print(f'See the log:\n  "{log_path}"\n')
        return rc

    if srt_path.exists():
        print("\nDone.")
        print(f'SRT: "{srt_path}"')
        print(f'LLM: "{llm_path}"')
        if cfg.mode == "fast" and not cfg.diarize:
            print("Note: fast mode used (speaker diarization disabled).")
        if fallback_no_diarize:
            print("Note: completed without speaker diarization.")
            print("To enable diarization, accept terms with the SAME HF account at:")
            print("  https://hf.co/pyannote/speaker-diarization-3.1")
            print("  https://hf.co/pyannote/segmentation-3.0")
        print()
        return 0

    print("\nDone, but SRT not found where expected:")
    print(f'  "{srt_path}"')
    print("Check the log:")
    print(f'  "{log_path}"\n')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

