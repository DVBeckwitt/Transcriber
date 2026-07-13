"""Command-line parsing and transcription settings."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .subtitles import (
    DEFAULT_HIGH_NO_SPEECH_PROB,
    DEFAULT_LOW_CONFIDENCE_LOGPROB,
    DEFAULT_LOW_CONFIDENCE_WORD_PROB,
    DEFAULT_MIN_SPEAKER_TURN_MS,
    DEFAULT_MIN_SPEAKER_TURN_TOKENS,
    normalize_subtitle_whitespace,
)

MEDIA_FILTER = (
    "Audio/Video",
    "*.wav *.mp3 *.m4a *.flac *.aac *.ogg *.opus *.wma *.mp4 *.mov *.mkv *.webm *.weba",
)

DEFAULT_POLL_INTERVAL = 2.0

DEFAULT_SETTLE_SECONDS = 5.0

DEFAULT_WATCH_DIR = Path.home() / "OneDrive" / "recordings"

DEFAULT_STALE_LOCK_SECONDS = 12 * 60 * 60

SPANISH_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-es-en"

TRANSLATION_CONTEXT_WINDOW = 2

TRANSLATION_NUM_BEAMS = 4

TRANSLATION_LENGTH_PENALTY = 1.0

TRANSLATION_NO_REPEAT_NGRAM_SIZE = 3

TRANSLATION_BATCH_SIZE = 4

TRANSLATION_MAX_NEW_TOKENS = 256

TRANSLATION_MARKER_START = "__CUR_START__"

TRANSLATION_MARKER_END = "__CUR_END__"


# Presets and configuration model


@dataclass
class LegacyOptions:
    language: str | None = None
    language_locked: bool = False
    mode: str | None = None
    mode_locked: bool = False
    model: str | None = None
    model_locked: bool = False


@dataclass(frozen=True)
class ModePreset:
    model: str
    batch_size: int
    beam_size: int
    patience: float
    temperature_schedule: tuple[float, ...]
    best_of: int
    condition_on_previous_text: bool
    diarize: bool


MODE_PRESETS = {
    "fast": ModePreset(
        model="medium",
        batch_size=16,
        beam_size=2,
        patience=1.0,
        temperature_schedule=(0.0,),
        best_of=1,
        condition_on_previous_text=False,
        diarize=False,
    ),
    "quality": ModePreset(
        model="large-v3",
        batch_size=8,
        beam_size=8,
        patience=1.2,
        temperature_schedule=(0.0, 0.2, 0.4, 0.6, 0.8),
        best_of=5,
        condition_on_previous_text=True,
        diarize=True,
    ),
}

LEGACY_ALIASES = {
    "e": ("language", "en"),
    "en": ("language", "en"),
    "s": ("language", "es"),
    "es": ("language", "es"),
    "t": ("language", "es"),
    "tr": ("language", "es"),
    "translate": ("language", "es"),
    "f": ("mode", "fast"),
    "fast": ("mode", "fast"),
    "q": ("mode", "quality"),
    "quality": ("mode", "quality"),
}


@dataclass
class RunConfig:
    """Resolved settings for one transcription run."""

    language: str
    translate_to_english: bool
    mode: str
    model: str
    batch_size: int
    beam_size: int
    patience: float
    temperature: float
    temperature_schedule: tuple[float, ...]
    best_of: int | None
    compression_ratio_threshold: float | None
    logprob_threshold: float | None
    no_speech_threshold: float | None
    condition_on_previous_text: bool
    diarize: bool
    diarize_smoothing: bool
    min_speaker_turn_ms: int
    min_speaker_turn_tokens: int
    include_speaker_labels: bool
    confidence_cleanup: bool
    confidence_cleanup_mode: str
    low_confidence_logprob: float
    high_no_speech_prob: float
    low_confidence_word_prob: float
    device: str
    compute_type: str
    translation_context_window: int
    translation_batch_size: int
    translation_num_beams: int
    translation_max_new_tokens: int
    translation_no_repeat_ngram_size: int
    glossary: dict[str, str]
    glossary_path: str | None
    asr_prompt: str | None
    warm_vram: bool
    dry_run: bool


# Prompt and glossary parsing


def parse_glossary_entries(raw_items: Sequence[str]) -> dict[str, str]:
    glossary: dict[str, str] = {}
    for raw in raw_items:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        line = line.strip().strip('"').strip("'")
        if not line:
            continue
        sep = None
        for candidate in ("=>", "->", "=", "|", "\t"):
            if candidate in line:
                sep = candidate
                break
        if sep is None:
            source = line
            target = line
        else:
            source, target = line.split(sep, 1)
            source = source.strip().strip('"').strip("'")
            target = target.strip().strip('"').strip("'")
            if not target:
                target = source
        if source:
            glossary[source] = target
    return glossary


def load_glossary_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    return list(text.splitlines())


def load_text_lines_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8-sig", errors="ignore")
    except OSError:
        return []

    lines: list[str] = []
    for line in raw.splitlines():
        cleaned = normalize_subtitle_whitespace(line)
        if cleaned and not cleaned.startswith("#"):
            lines.append(cleaned)
    return lines


def looks_like_glossary_file(value: str) -> bool:
    if any(sep in value for sep in ("=>", "->", "=", "|", "\t")):
        return False
    candidate = Path(value).expanduser()
    return candidate.exists()


def build_asr_prompt(
    *,
    glossary: dict[str, str],
    prompt_text: str | None = None,
    prompt_file: str | None = None,
    max_glossary_terms: int = 32,
) -> str | None:
    lines: list[str] = []

    if prompt_file:
        lines.extend(load_text_lines_file(Path(prompt_file).expanduser()))

    if prompt_text:
        for line in prompt_text.splitlines():
            cleaned = normalize_subtitle_whitespace(line)
            if cleaned:
                lines.append(cleaned)

    if glossary:
        lines.append("Use these names, product names, and jargon exactly as written:")
        for source, target in sorted(
            glossary.items(), key=lambda item: len(item[0]), reverse=True
        )[:max_glossary_terms]:
            if source == target:
                lines.append(f"- {source}")
            else:
                lines.append(f"- {source} (preferred spelling: {target})")

    prompt = "\n".join(line for line in lines if line.strip()).strip()
    return prompt or None


def parse_temperature_schedule(value: str | None) -> tuple[float, ...]:
    if not value:
        return ()

    temperatures: list[float] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        temperatures.append(float(item))

    if not temperatures:
        raise ValueError("Temperature schedule must contain at least one value.")
    return tuple(temperatures)


# Argument parser


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    add = parser.add_argument
    add("legacy", nargs="*", help="Legacy tokens: [language] [model] [mode]")
    add("--input", "-i", help="Audio/video file path.")
    add("--lang", choices=("auto", "en", "es"), help="Language: auto, en, or es.")
    add(
        "--glossary",
        action="append",
        default=[],
        help="Glossary entry: 'source=target' or 'source' to preserve. Repeatable.",
    )
    add(
        "--glossary-file",
        help="Glossary file with one source, source => target, tab-separated, or source | target entry per line.",
    )
    add(
        "--asr-prompt",
        help="Optional text prompt to bias WhisperX toward names and jargon.",
    )
    add(
        "--asr-prompt-file",
        help="Text file with extra ASR prompt lines for names and jargon.",
    )
    add(
        "--translate-to-english",
        action="store_true",
        help="Use WhisperX translate mode to write English SRT directly.",
    )
    add(
        "--no-speaker-labels",
        dest="include_speaker_labels",
        action="store_false",
        help="Hide diarization speaker labels in rendered transcript outputs.",
    )


def _add_decode_arguments(parser: argparse.ArgumentParser) -> None:
    add = parser.add_argument
    add(
        "--temperature",
        type=float,
        help="Single decoding temperature. Overrides the preset schedule.",
    )
    add(
        "--temperature-schedule",
        help="Comma-separated fallback temperatures, e.g. 0.0,0.2,0.4.",
    )
    add("--best-of", type=int, help="Sampling candidates when temperature is above 0.")
    add(
        "--compression-ratio-threshold",
        type=float,
        help="Fallback threshold for output compression ratio.",
    )
    add(
        "--logprob-threshold",
        type=float,
        help="Fallback threshold for average log probability.",
    )
    add(
        "--no-speech-threshold",
        type=float,
        help="Fallback threshold for no-speech probability.",
    )
    add(
        "--condition-on-previous-text",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Condition each decode window on previous text.",
    )
    add("--mode", choices=("quality", "fast"), help="Run mode preset: quality or fast.")
    add("--model", help="Whisper model name, e.g. large-v3, medium.")
    add("--device", default="cuda", help="Inference device (default: cuda).")
    add(
        "--compute-type",
        default="float16",
        choices=("float16", "float32", "int8"),
        help="Computation dtype.",
    )
    add(
        "--warm-vram",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep compatible WhisperX models loaded between jobs (default: off).",
    )


def _add_watcher_arguments(parser: argparse.ArgumentParser) -> None:
    add = parser.add_argument
    add(
        "--watch",
        action="store_true",
        help="Continuously watch a folder and transcribe new media files.",
    )
    add(
        "--watch-dir",
        default=str(DEFAULT_WATCH_DIR),
        help=f'Watch folder (default: "{DEFAULT_WATCH_DIR}").',
    )
    add(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help="Seconds between watch scans.",
    )
    add(
        "--settle-seconds",
        type=float,
        default=DEFAULT_SETTLE_SECONDS,
        help="Seconds a file must remain unchanged.",
    )
    add(
        "--stale-lock-seconds",
        type=float,
        default=DEFAULT_STALE_LOCK_SECONDS,
        help="Age after which a lock is stale.",
    )
    add(
        "--dry-run",
        action="store_true",
        help="Show planned actions without running or writing outputs.",
    )


def _add_diarization_arguments(parser: argparse.ArgumentParser) -> None:
    diarize = parser.add_mutually_exclusive_group()
    diarize.add_argument(
        "--diarize",
        dest="force_diarize",
        action="store_true",
        help="Force diarization on.",
    )
    diarize.add_argument(
        "--no-diarize",
        dest="force_no_diarize",
        action="store_true",
        help="Force diarization off.",
    )
    parser.add_argument(
        "--no-diarize-smoothing",
        action="store_true",
        help="Disable speaker diarization smoothing.",
    )
    parser.add_argument(
        "--min-speaker-turn-ms",
        type=int,
        default=DEFAULT_MIN_SPEAKER_TURN_MS,
        help=f"Minimum speaker turn duration (default: {DEFAULT_MIN_SPEAKER_TURN_MS}).",
    )
    parser.add_argument(
        "--min-speaker-turn-tokens",
        type=int,
        default=DEFAULT_MIN_SPEAKER_TURN_TOKENS,
        help=f"Minimum speaker turn token count (default: {DEFAULT_MIN_SPEAKER_TURN_TOKENS}).",
    )


def _add_confidence_arguments(parser: argparse.ArgumentParser) -> None:
    add = parser.add_argument
    add(
        "--confidence-cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable low-confidence transcript cleanup (default: on).",
    )
    add(
        "--confidence-cleanup-mode",
        choices=("mark", "redact"),
        default="mark",
        help="Low-confidence handling mode.",
    )
    add(
        "--low-confidence-logprob",
        type=float,
        default=DEFAULT_LOW_CONFIDENCE_LOGPROB,
        help=f"Average logprob threshold (default: {DEFAULT_LOW_CONFIDENCE_LOGPROB}).",
    )
    add(
        "--high-no-speech-prob",
        type=float,
        default=DEFAULT_HIGH_NO_SPEECH_PROB,
        help=f"No-speech probability threshold (default: {DEFAULT_HIGH_NO_SPEECH_PROB}).",
    )
    add(
        "--low-confidence-word-prob",
        type=float,
        default=DEFAULT_LOW_CONFIDENCE_WORD_PROB,
        help=f"Word confidence threshold (default: {DEFAULT_LOW_CONFIDENCE_WORD_PROB}).",
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="WhisperX one-click transcriber (quality + fast modes)."
    )
    _add_input_arguments(parser)
    _add_decode_arguments(parser)
    _add_watcher_arguments(parser)
    _add_diarization_arguments(parser)
    _add_confidence_arguments(parser)
    return parser.parse_args(argv)


# Interactive input and final configuration assembly


def parse_legacy(tokens: Iterable[str]) -> LegacyOptions:
    options = LegacyOptions()
    for token in tokens:
        value = token.strip()
        if not value:
            continue

        alias = LEGACY_ALIASES.get(value.lower())
        if alias is not None:
            field, normalized_value = alias
            setattr(options, field, normalized_value)
            setattr(options, f"{field}_locked", True)
        elif not options.model_locked:
            options.model = value
            options.model_locked = True
    return options


def prompt_input_path() -> str | None:
    if not sys.stdin.isatty():
        return None
    raw = input("Media file path (or Enter to cancel): ").strip()
    return raw or None


def prompt_choice(prompt: str, current: str, choices: dict[str, str]) -> str:
    if not sys.stdin.isatty():
        return current
    response = input(prompt).strip().lower()
    return choices.get(response, current)


def prompt_language(current: str) -> str:
    return prompt_choice(
        "Language [Enter=Auto, e=English, s=Spanish]: ",
        current,
        {
            "": "auto",
            "a": "auto",
            "auto": "auto",
            "e": "en",
            "en": "en",
            "s": "es",
            "es": "es",
        },
    )


def prompt_mode(current: str) -> str:
    return prompt_choice(
        "Run mode [Enter=quality, f=fast]: ",
        current,
        {
            "": "quality",
            "f": "fast",
            "fast": "fast",
            "q": "quality",
            "quality": "quality",
        },
    )


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


def resolve_input_path(arg_input: str | None) -> Path | None:
    if arg_input:
        return Path(arg_input).expanduser()

    chosen = pick_media_file()
    if chosen:
        return Path(chosen)

    manual = prompt_input_path()
    if manual:
        return Path(manual).expanduser()

    return None


def _temperature_settings(
    args: argparse.Namespace, preset: ModePreset
) -> tuple[float, tuple[float, ...]]:
    schedule = preset.temperature_schedule
    if args.temperature is not None:
        schedule = (float(args.temperature),)

    try:
        override = parse_temperature_schedule(args.temperature_schedule)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if override:
        schedule = override
    return schedule[0], schedule


def _glossary_settings(args: argparse.Namespace) -> tuple[dict[str, str], str | None]:
    entries = list(args.glossary or [])
    files = [Path(args.glossary_file).expanduser()] if args.glossary_file else []

    for entry in list(entries):
        if looks_like_glossary_file(entry):
            files.append(Path(entry).expanduser())
            entries.remove(entry)

    file_entries = [
        line for glossary_file in files for line in load_glossary_file(glossary_file)
    ]
    glossary = parse_glossary_entries([*file_entries, *entries])
    return glossary, str(files[0]) if files else None


def _diarization_enabled(args: argparse.Namespace, default: bool) -> bool:
    if args.force_diarize:
        return True
    if args.force_no_diarize:
        return False
    return default


def build_config(args: argparse.Namespace, interactive: bool = True) -> RunConfig:
    legacy = parse_legacy(args.legacy)

    language = args.lang or legacy.language or "auto"
    mode = args.mode or legacy.mode or "quality"
    model = args.model or legacy.model

    if interactive and not (args.lang or legacy.language_locked):
        language = prompt_language(language)
    if interactive and not (args.mode or legacy.mode_locked):
        mode = prompt_mode(mode)

    preset = MODE_PRESETS[mode]
    if not (args.model or legacy.model_locked):
        model = preset.model

    temperature, temperature_schedule = _temperature_settings(args, preset)
    best_of = preset.best_of if args.best_of is None else max(1, int(args.best_of))
    compression_ratio_threshold = (
        2.4
        if args.compression_ratio_threshold is None
        else float(args.compression_ratio_threshold)
    )
    logprob_threshold = (
        -1.0 if args.logprob_threshold is None else float(args.logprob_threshold)
    )
    no_speech_threshold = (
        0.6 if args.no_speech_threshold is None else float(args.no_speech_threshold)
    )
    condition_on_previous_text = (
        preset.condition_on_previous_text
        if args.condition_on_previous_text is None
        else bool(args.condition_on_previous_text)
    )

    glossary, glossary_path = _glossary_settings(args)
    asr_prompt = build_asr_prompt(
        glossary=glossary,
        prompt_text=args.asr_prompt,
        prompt_file=args.asr_prompt_file,
    )

    return RunConfig(
        language=language,
        translate_to_english=bool(args.translate_to_english) or language == "es",
        mode=mode,
        model=model or preset.model,
        batch_size=preset.batch_size,
        beam_size=preset.beam_size,
        patience=preset.patience,
        temperature=temperature,
        temperature_schedule=temperature_schedule,
        best_of=best_of,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold,
        condition_on_previous_text=condition_on_previous_text,
        diarize=_diarization_enabled(args, preset.diarize),
        diarize_smoothing=not args.no_diarize_smoothing,
        min_speaker_turn_ms=max(0, int(args.min_speaker_turn_ms)),
        min_speaker_turn_tokens=max(0, int(args.min_speaker_turn_tokens)),
        include_speaker_labels=bool(args.include_speaker_labels),
        confidence_cleanup=bool(args.confidence_cleanup),
        confidence_cleanup_mode=args.confidence_cleanup_mode,
        low_confidence_logprob=float(args.low_confidence_logprob),
        high_no_speech_prob=float(args.high_no_speech_prob),
        low_confidence_word_prob=float(args.low_confidence_word_prob),
        device=args.device,
        compute_type=args.compute_type,
        translation_context_window=TRANSLATION_CONTEXT_WINDOW,
        translation_batch_size=TRANSLATION_BATCH_SIZE,
        translation_num_beams=TRANSLATION_NUM_BEAMS,
        translation_max_new_tokens=TRANSLATION_MAX_NEW_TOKENS,
        translation_no_repeat_ngram_size=TRANSLATION_NO_REPEAT_NGRAM_SIZE,
        glossary=glossary,
        glossary_path=glossary_path,
        asr_prompt=asr_prompt,
        warm_vram=bool(args.warm_vram),
        dry_run=bool(args.dry_run),
    )
