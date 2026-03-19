from __future__ import annotations

import argparse
import contextlib
import functools
import importlib
import inspect
import math
import os
import re
import socket
import sys
import tempfile
import textwrap
import time
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


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

LOG_DIR_NAME = "logs"
WATCHER_LOG_NAME = "transcriber-watcher.log"
DEFAULT_POLL_INTERVAL = 2.0
DEFAULT_SETTLE_SECONDS = 5.0
DEFAULT_WATCH_DIR = Path.home() / "OneDrive" / "recordings"
WATCH_RETRY_COOLDOWN_SECONDS = 300.0
FALLBACK_TEMP_DIR_NAME = ".tmp_transcriber_temp"
LOCK_SUFFIX = ".transcribing.lock"
DEFAULT_STALE_LOCK_SECONDS = 12 * 60 * 60

SUBTITLE_MAX_LINES = 2
SUBTITLE_MAX_CHARS_PER_LINE = 42
SUBTITLE_MAX_DURATION_SECONDS = 6.0
SUBTITLE_TARGET_CPS = 17.0
SUBTITLE_PREFERRED_BREAK_CHARS = ".?!,:;"

SPANISH_TRANSLATION_MODEL = "Helsinki-NLP/opus-mt-es-en"
TRANSLATION_CONTEXT_WINDOW = 2
TRANSLATION_NUM_BEAMS = 4
TRANSLATION_LENGTH_PENALTY = 1.0
TRANSLATION_NO_REPEAT_NGRAM_SIZE = 3
TRANSLATION_MARKER_START = "__CUR_START__"
TRANSLATION_MARKER_END = "__CUR_END__"
UNCERTAIN_MARKER_RE = re.compile(
    r"__UNCERTAIN(?:_(\d+))?__(.*?)__UNCERTAIN_END__",
    re.DOTALL,
)

DEFAULT_MIN_SPEAKER_TURN_MS = 900
DEFAULT_MIN_SPEAKER_TURN_TOKENS = 2
DEFAULT_LOW_CONFIDENCE_LOGPROB = -1.0
DEFAULT_HIGH_NO_SPEECH_PROB = 0.6
DEFAULT_LOW_CONFIDENCE_WORD_PROB = 0.5

Reporter = Callable[[str], None]


@dataclass
class LegacyOptions:
    language: str | None = None
    language_locked: bool = False
    mode: str | None = None
    mode_locked: bool = False
    model: str | None = None
    model_locked: bool = False


@dataclass
class RunConfig:
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
    confidence_cleanup: bool
    confidence_cleanup_mode: str
    low_confidence_logprob: float
    high_no_speech_prob: float
    low_confidence_word_prob: float
    device: str
    compute_type: str
    glossary: dict[str, str]
    glossary_path: str | None
    asr_prompt: str | None
    dry_run: bool


@dataclass
class OutputPaths:
    output_dir: Path
    srt_path: Path
    llm_path: Path
    log_path: Path
    lock_path: Path


@dataclass
class SRTCue:
    index: int
    start_ms: int
    end_ms: int
    text: str


@dataclass
class TimedToken:
    text: str
    start_ms: int
    end_ms: int
    speaker: str = ""
    low_confidence: bool = False
    confidence: float | None = None


@dataclass
class PendingWatchFile:
    size: int
    mtime_ns: int
    stable_since: float
    last_attempt_at: float | None = None


UNSUPPORTED_GLOBAL_RE = re.compile(r"Unsupported global: GLOBAL ([A-Za-z0-9_\.]+)")
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


def project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def timestamp_to_ms(value: str) -> int:
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return (((int(hours) * 60) + int(minutes)) * 60 + int(seconds)) * 1000 + int(millis)


def ms_to_timestamp(total_ms: int) -> str:
    total_ms = max(0, int(total_ms))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def normalize_subtitle_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    text = re.sub(r'([("])\s+', r"\1", text)
    text = re.sub(r'\s+([)")])', r"\1", text)
    return text


def split_speaker_prefix(text: str) -> tuple[str, str]:
    match = re.match(r"^([A-Z][A-Z0-9_ ]{1,31}:\s+)(.+)$", text.strip())
    if not match:
        return "", normalize_subtitle_whitespace(text)
    return match.group(1), normalize_subtitle_whitespace(match.group(2))


def normalize_speaker_label(label: str | None) -> str:
    if not label:
        return ""
    return normalize_subtitle_whitespace(str(label)).rstrip(":")


def speaker_prefix(label: str | None) -> str:
    cleaned = normalize_speaker_label(label)
    return f"{cleaned}: " if cleaned else ""


def parse_srt_cues(srt_text: str) -> list[SRTCue]:
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    cues: list[SRTCue] = []
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        line_index = 0
        if lines[0].strip().isdigit():
            line_index = 1
        if line_index >= len(lines) or "-->" not in lines[line_index]:
            continue
        start_raw, end_raw = [part.strip() for part in lines[line_index].split("-->", 1)]
        text = " ".join(line.strip() for line in lines[line_index + 1 :] if line.strip())
        if not text:
            continue
        cues.append(
            SRTCue(
                index=len(cues) + 1,
                start_ms=timestamp_to_ms(start_raw),
                end_ms=timestamp_to_ms(end_raw),
                text=text,
            )
        )
    return cues


def render_srt_cues(cues: Sequence[SRTCue]) -> str:
    rendered: list[str] = []
    for idx, cue in enumerate(cues, start=1):
        rendered.append(str(idx))
        rendered.append(f"{ms_to_timestamp(cue.start_ms)} --> {ms_to_timestamp(cue.end_ms)}")
        rendered.append(cue.text)
        rendered.append("")
    return "\n".join(rendered).rstrip() + "\n"


def wrap_subtitle_lines(
    text: str,
    max_chars_per_line: int = SUBTITLE_MAX_CHARS_PER_LINE,
    max_lines: int = SUBTITLE_MAX_LINES,
) -> str:
    normalized = normalize_subtitle_whitespace(text)
    if not normalized:
        return normalized

    wrapped = textwrap.wrap(
        normalized,
        width=max_chars_per_line,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if len(wrapped) <= max_lines:
        return "\n".join(wrapped)

    wrapped = textwrap.wrap(
        normalized,
        width=max_chars_per_line,
        break_long_words=True,
        break_on_hyphens=False,
    )
    if len(wrapped) <= max_lines:
        return "\n".join(wrapped)

    kept = wrapped[: max_lines - 1]
    remainder = normalize_subtitle_whitespace(" ".join(wrapped[max_lines - 1 :]))
    kept.append(remainder)
    return "\n".join(kept)


def wrap_subtitle_lines_exact(
    text: str,
    max_chars_per_line: int = SUBTITLE_MAX_CHARS_PER_LINE,
    max_lines: int = SUBTITLE_MAX_LINES,
) -> list[str]:
    normalized = normalize_subtitle_whitespace(text)
    if not normalized:
        return []

    for break_long_words in (False, True):
        wrapped = textwrap.wrap(
            normalized,
            width=max_chars_per_line,
            break_long_words=break_long_words,
            break_on_hyphens=False,
        )
        if wrapped and len(wrapped) <= max_lines and max(len(line) for line in wrapped) <= max_chars_per_line:
            return wrapped
    return []


def split_text_into_chunks(text: str, soft_limit: int, hard_limit: int) -> list[str]:
    normalized = normalize_subtitle_whitespace(text)
    if not normalized:
        return []

    words = normalized.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []

    def flush_current() -> None:
        if current:
            chunks.append(" ".join(current))
            current.clear()

    for idx, word in enumerate(words):
        tentative_words = [*current, word]
        tentative = " ".join(tentative_words)

        if current and len(tentative) > hard_limit:
            flush_current()
            current.append(word)
            continue

        current.append(word)
        current_text = " ".join(current)
        next_word = words[idx + 1] if idx + 1 < len(words) else ""
        next_tentative = (current_text + " " + next_word).strip() if next_word else current_text

        if len(current_text) >= soft_limit and word.endswith(tuple(SUBTITLE_PREFERRED_BREAK_CHARS)):
            flush_current()
            continue

        if next_word and len(next_tentative) > hard_limit:
            flush_current()

    flush_current()

    merged: list[str] = []
    for chunk in chunks:
        if merged and len(chunk) < max(12, soft_limit // 3) and len(merged[-1]) + 1 + len(chunk) <= hard_limit:
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)

    while len(merged) > 1 and len(merged[-1]) < max(12, hard_limit // 5):
        combined = f"{merged[-2]} {merged[-1]}"
        if len(combined) > hard_limit:
            break
        merged[-2] = combined
        merged.pop()
    return merged


def split_cue_for_subtitles(cue: SRTCue) -> list[SRTCue]:
    cleaned_text = normalize_subtitle_whitespace(cue.text)
    if not cleaned_text:
        return []

    prefix, content = split_speaker_prefix(cleaned_text)
    base_hard_limit = SUBTITLE_MAX_LINES * SUBTITLE_MAX_CHARS_PER_LINE
    hard_limit = max(SUBTITLE_MAX_CHARS_PER_LINE, base_hard_limit - len(prefix))
    duration_ms = max(1, cue.end_ms - cue.start_ms)
    duration_seconds = duration_ms / 1000.0
    char_count = max(1, len(content))

    segment_count = max(
        1,
        math.ceil(duration_seconds / SUBTITLE_MAX_DURATION_SECONDS),
        math.ceil(char_count / hard_limit),
        math.ceil(char_count / (SUBTITLE_MAX_DURATION_SECONDS * SUBTITLE_TARGET_CPS)),
    )
    soft_limit = max(SUBTITLE_MAX_CHARS_PER_LINE, math.ceil(char_count / segment_count))
    chunks = split_text_into_chunks(content, soft_limit=min(soft_limit, hard_limit), hard_limit=hard_limit)

    if not chunks:
        return []

    weighted_lengths = [max(1, len(chunk)) for chunk in chunks]
    total_weight = sum(weighted_lengths)
    split_points = [cue.start_ms]
    running = 0
    for weight in weighted_lengths[:-1]:
        running += weight
        split_points.append(cue.start_ms + round(duration_ms * running / total_weight))
    split_points.append(cue.end_ms)

    new_cues: list[SRTCue] = []
    for idx, chunk in enumerate(chunks):
        start_ms = split_points[idx]
        end_ms = split_points[idx + 1]
        wrapped_text = wrap_subtitle_lines(prefix + chunk)
        new_cues.append(SRTCue(index=idx + 1, start_ms=start_ms, end_ms=end_ms, text=wrapped_text))
    return new_cues


def seconds_to_ms(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return max(0, int(round(float(value) * 1000.0)))
    except (TypeError, ValueError):
        return None


def probability_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        prob = float(value)
    except (TypeError, ValueError):
        return None
    if prob < 0.0:
        return 0.0
    if prob > 1.0:
        return 1.0
    return prob


def call_with_supported_kwargs(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)

    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return func(*args, **kwargs)

    filtered = {key: value for key, value in kwargs.items() if key in parameters}
    return func(*args, **filtered)


def resolve_whisperx_symbol(whisperx_module: Any, symbol_name: str) -> Any:
    if hasattr(whisperx_module, symbol_name):
        return getattr(whisperx_module, symbol_name)

    for module_name in ("diarize",):
        with contextlib.suppress(Exception):
            module = importlib.import_module(f"{whisperx_module.__name__}.{module_name}")
            if hasattr(module, symbol_name):
                return getattr(module, symbol_name)

    raise AttributeError(
        f"Could not find whisperx symbol {symbol_name!r} in {whisperx_module.__name__} or its supported compatibility modules."
    )


def segment_is_low_confidence(
    segment: dict[str, Any],
    *,
    low_logprob: float,
    high_no_speech: float,
) -> bool:
    avg_logprob = segment.get("avg_logprob")
    if isinstance(avg_logprob, (int, float)) and avg_logprob < low_logprob:
        return True
    no_speech_prob = segment.get("no_speech_prob")
    if isinstance(no_speech_prob, (int, float)) and no_speech_prob > high_no_speech:
        return True
    return False


def word_is_low_confidence(word: dict[str, Any], *, low_prob: float) -> bool:
    for key in ("probability", "confidence", "score"):
        value = word.get(key)
        if isinstance(value, (int, float)) and value < low_prob:
            return True
    return False


def apply_confidence_cleanup(result: dict[str, Any], cfg: RunConfig) -> None:
    if not cfg.confidence_cleanup:
        return

    segments = result.get("segments", [])
    if not isinstance(segments, list):
        return

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        seg_text = str(segment.get("text") or "")
        seg_low = segment_is_low_confidence(
            segment,
            low_logprob=cfg.low_confidence_logprob,
            high_no_speech=cfg.high_no_speech_prob,
        )
        if seg_low:
            segment["_low_confidence"] = True

        words = segment.get("words") or []
        if isinstance(words, list):
            for word in words:
                if not isinstance(word, dict):
                    continue
                word_low = word_is_low_confidence(word, low_prob=cfg.low_confidence_word_prob) or seg_low
                if not word_low:
                    continue
                word["_low_confidence"] = True

        if seg_low:
            segment["_low_confidence"] = True


def segment_to_timed_tokens(segment: dict[str, Any]) -> list[TimedToken]:
    seg_speaker = normalize_speaker_label(segment.get("speaker"))
    seg_low_confidence = bool(segment.get("_low_confidence"))
    words = segment.get("words") or []
    timed_words: list[TimedToken] = []

    for raw_word in words:
        if not isinstance(raw_word, dict):
            continue
        word_text = normalize_subtitle_whitespace(str(raw_word.get("word") or ""))
        start_ms = seconds_to_ms(raw_word.get("start"))
        end_ms = seconds_to_ms(raw_word.get("end"))
        if not word_text or start_ms is None or end_ms is None:
            continue
        end_ms = max(end_ms, start_ms + 1)
        word_speaker = normalize_speaker_label(raw_word.get("speaker") or seg_speaker)
        timed_words.append(
            TimedToken(
                text=word_text,
                start_ms=start_ms,
                end_ms=end_ms,
                speaker=word_speaker,
                low_confidence=bool(raw_word.get("_low_confidence")) or seg_low_confidence,
                confidence=probability_value(raw_word.get("probability") or raw_word.get("score")),
            )
        )

    if timed_words:
        return timed_words

    seg_text = normalize_subtitle_whitespace(str(segment.get("text") or ""))
    seg_start_ms = seconds_to_ms(segment.get("start"))
    seg_end_ms = seconds_to_ms(segment.get("end"))
    if not seg_text or seg_start_ms is None or seg_end_ms is None or seg_end_ms <= seg_start_ms:
        return []

    raw_words = seg_text.split()
    if not raw_words:
        return []

    total_weight = sum(max(1, len(word)) for word in raw_words)
    duration_ms = seg_end_ms - seg_start_ms
    cursor_ms = seg_start_ms
    distributed: list[TimedToken] = []
    running_weight = 0
    for idx, word in enumerate(raw_words):
        running_weight += max(1, len(word))
        if idx == len(raw_words) - 1:
            next_ms = seg_end_ms
        else:
            next_ms = seg_start_ms + round(duration_ms * running_weight / total_weight)
        next_ms = max(next_ms, cursor_ms + 1)
        distributed.append(
            TimedToken(
                text=word,
                start_ms=cursor_ms,
                end_ms=next_ms,
                speaker=seg_speaker,
                low_confidence=seg_low_confidence,
                confidence=probability_value(segment.get("probability") or segment.get("score")),
            )
        )
        cursor_ms = next_ms
    return distributed


def token_uncertain_marker(token: TimedToken) -> str:
    if token.confidence is None:
        return f"__UNCERTAIN__{token.text}__UNCERTAIN_END__"
    percent = max(0, min(100, int(round(float(token.confidence) * 100.0))))
    return f"__UNCERTAIN_{percent}__{token.text}__UNCERTAIN_END__"


def format_token_text(token: TimedToken, style: str = "plain") -> str:
    if not token.text:
        return ""
    if style == "marker" and token.low_confidence:
        return token_uncertain_marker(token)
    return token.text


def format_cue_text(prefix: str, tokens: Sequence[TimedToken], style: str = "plain") -> str:
    body = normalize_subtitle_whitespace(
        " ".join(format_token_text(token, style=style) for token in tokens if token.text)
    )
    return normalize_subtitle_whitespace(f"{prefix}{body}") if prefix else body


def render_uncertain_markup(text: str, style: str) -> str:
    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        pct = match.group(1)
        inner = normalize_subtitle_whitespace(match.group(2))
        if style == "srt":
            return f"<i>{inner}</i>"
        if style == "llm":
            if pct:
                return f"[{inner}] [{pct}% confidence]"
            return f"[{inner}]"
        return inner

    return UNCERTAIN_MARKER_RE.sub(replace, text)


def smooth_timed_tokens(
    tokens: Sequence[TimedToken],
    min_run_duration_ms: int = DEFAULT_MIN_SPEAKER_TURN_MS,
    min_run_words: int = DEFAULT_MIN_SPEAKER_TURN_TOKENS,
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_WORD_PROB,
) -> list[TimedToken]:
    if not tokens:
        return []

    smoothed = [
        TimedToken(
            text=token.text,
            start_ms=token.start_ms,
            end_ms=token.end_ms,
            speaker=token.speaker,
            low_confidence=token.low_confidence,
            confidence=token.confidence,
        )
        for token in tokens
    ]

    runs: list[dict[str, Any]] = []
    current_run: list[TimedToken] = []
    current_speaker = ""

    def flush_run() -> None:
        nonlocal current_run, current_speaker
        if not current_run:
            return
        start_ms = current_run[0].start_ms
        end_ms = current_run[-1].end_ms
        confidences = [token.confidence for token in current_run if token.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None
        has_low_confidence = any(token.low_confidence for token in current_run)
        runs.append(
            {
                "speaker": current_speaker,
                "tokens": current_run[:],
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": max(1, end_ms - start_ms),
                "word_count": len(current_run),
                "avg_confidence": avg_confidence,
                "has_low_confidence": has_low_confidence,
            }
        )
        current_run = []
        current_speaker = ""

    for token in smoothed:
        token_speaker = token.speaker
        if not current_run:
            current_run = [token]
            current_speaker = token_speaker
            continue
        if token_speaker != current_speaker:
            flush_run()
            current_run = [token]
            current_speaker = token_speaker
            continue
        current_run.append(token)
    flush_run()

    if len(runs) < 3:
        return smoothed

    for idx in range(1, len(runs) - 1):
        run = runs[idx]
        speaker = str(run["speaker"] or "")
        prev_run = runs[idx - 1]
        next_run = runs[idx + 1]
        is_short = int(run["duration_ms"]) < min_run_duration_ms or int(run["word_count"]) <= min_run_words
        low_confidence = (
            run["avg_confidence"] is not None and float(run["avg_confidence"]) < low_confidence_threshold
        )
        low_confidence = low_confidence or bool(run.get("has_low_confidence"))
        if not speaker or not (is_short or low_confidence):
            continue

        prev_speaker = str(prev_run["speaker"] or "")
        next_speaker = str(next_run["speaker"] or "")
        if prev_speaker and prev_speaker == next_speaker:
            replacement = prev_speaker
        elif prev_speaker and next_speaker:
            prev_score = int(prev_run["duration_ms"]) + int(prev_run["word_count"]) * 250
            next_score = int(next_run["duration_ms"]) + int(next_run["word_count"]) * 250
            replacement = prev_speaker if prev_score >= next_score else next_speaker
        else:
            replacement = prev_speaker or next_speaker

        if not replacement or replacement == speaker:
            continue
        for token in run["tokens"]:
            token.speaker = replacement

    smoothed_tokens: list[TimedToken] = []
    for run in runs:
        smoothed_tokens.extend(run["tokens"])
    return smoothed_tokens


def extract_timed_tokens(result: dict[str, Any], cfg: RunConfig | None = None) -> list[TimedToken]:
    tokens: list[TimedToken] = []
    for raw_segment in result.get("segments", []):
        if not isinstance(raw_segment, dict):
            continue
        tokens.extend(segment_to_timed_tokens(raw_segment))

    if not tokens:
        return []

    tokens.sort(key=lambda token: (token.start_ms, token.end_ms))
    previous_start = 0
    for token in tokens:
        token.start_ms = max(token.start_ms, previous_start)
        token.end_ms = max(token.end_ms, token.start_ms + 1)
        previous_start = token.start_ms

    if cfg is not None and not cfg.diarize_smoothing:
        return tokens

    min_run_ms = cfg.min_speaker_turn_ms if cfg is not None else DEFAULT_MIN_SPEAKER_TURN_MS
    min_run_tokens = cfg.min_speaker_turn_tokens if cfg is not None else DEFAULT_MIN_SPEAKER_TURN_TOKENS
    low_confidence_threshold = cfg.low_confidence_word_prob if cfg is not None else DEFAULT_LOW_CONFIDENCE_WORD_PROB
    low_confidence_threshold = max(0.0, min(1.0, float(low_confidence_threshold)))
    return smooth_timed_tokens(
        tokens,
        min_run_duration_ms=max(0, int(min_run_ms)),
        min_run_words=max(0, int(min_run_tokens)),
        low_confidence_threshold=low_confidence_threshold,
    )


def reading_speed_cps(text: str, duration_ms: int) -> float:
    if duration_ms <= 0:
        return float("inf")
    return len(text.replace("\n", " ")) / (duration_ms / 1000.0)


def cue_candidate_is_valid(tokens: Sequence[TimedToken], prefix: str) -> bool:
    if not tokens:
        return False

    start_ms = tokens[0].start_ms
    end_ms = max(tokens[-1].end_ms, start_ms + 1)
    duration_ms = end_ms - start_ms
    if duration_ms > int(round(SUBTITLE_MAX_DURATION_SECONDS * 1000.0)):
        return False

    display_text = format_cue_text(prefix, tokens)
    wrapped = wrap_subtitle_lines_exact(display_text)
    return bool(wrapped)


def should_soft_break(tokens: Sequence[TimedToken], prefix: str) -> bool:
    if not tokens:
        return False

    display_text = format_cue_text(prefix, tokens)
    duration_ms = max(1, tokens[-1].end_ms - tokens[0].start_ms)
    last_text = tokens[-1].text.strip()
    punctuation_break = last_text.endswith(tuple(SUBTITLE_PREFERRED_BREAK_CHARS))
    enough_text = len(display_text) >= max(26, SUBTITLE_MAX_CHARS_PER_LINE)
    enough_time = duration_ms >= 2200
    near_limit = duration_ms >= 4200 or reading_speed_cps(display_text, duration_ms) >= SUBTITLE_TARGET_CPS * 0.95
    return punctuation_break and (enough_text or enough_time or near_limit)


def finalize_timed_cue(index: int, prefix: str, tokens: Sequence[TimedToken]) -> SRTCue:
    if not tokens:
        raise ValueError("Cannot finalize an empty subtitle cue.")

    text = format_cue_text(prefix, tokens, style="marker")
    wrapped = wrap_subtitle_lines_exact(text)
    if not wrapped:
        wrapped_text = wrap_subtitle_lines(text)
    else:
        wrapped_text = "\n".join(wrapped)

    start_ms = tokens[0].start_ms
    end_ms = max(tokens[-1].end_ms, start_ms + 1)
    return SRTCue(index=index, start_ms=start_ms, end_ms=end_ms, text=wrapped_text)


def build_segment_fallback_cues(result: dict[str, Any]) -> list[SRTCue]:
    cues: list[SRTCue] = []
    for raw_segment in result.get("segments", []):
        if not isinstance(raw_segment, dict):
            continue
        start_ms = seconds_to_ms(raw_segment.get("start"))
        end_ms = seconds_to_ms(raw_segment.get("end"))
        if start_ms is None or end_ms is None or end_ms <= start_ms:
            continue
        text = normalize_subtitle_whitespace(str(raw_segment.get("text") or ""))
        if not text:
            continue
        prefix = speaker_prefix(raw_segment.get("speaker"))
        cues.extend(split_cue_for_subtitles(SRTCue(index=0, start_ms=start_ms, end_ms=end_ms, text=f"{prefix}{text}")))

    for idx, cue in enumerate(cues, start=1):
        cue.index = idx
    return cues


def build_srt_cues_from_result(result: dict[str, Any], cfg: RunConfig | None = None) -> list[SRTCue]:
    tokens = extract_timed_tokens(result, cfg)
    if not tokens:
        return build_segment_fallback_cues(result)

    cues: list[SRTCue] = []
    current_tokens: list[TimedToken] = []
    current_prefix = ""

    def flush_current() -> None:
        nonlocal current_tokens, current_prefix
        if not current_tokens:
            return
        cues.append(finalize_timed_cue(len(cues) + 1, current_prefix, current_tokens))
        current_tokens = []
        current_prefix = ""

    for token in tokens:
        token_prefix = speaker_prefix(token.speaker)
        if not current_tokens:
            current_tokens = [token]
            current_prefix = token_prefix
            continue

        if token_prefix and current_prefix and token_prefix != current_prefix:
            flush_current()
            current_tokens = [token]
            current_prefix = token_prefix
            continue

        candidate_tokens = [*current_tokens, token]
        if cue_candidate_is_valid(candidate_tokens, current_prefix):
            current_tokens = candidate_tokens
            if should_soft_break(current_tokens, current_prefix):
                flush_current()
            continue

        flush_current()
        current_tokens = [token]
        current_prefix = token_prefix

        if not cue_candidate_is_valid(current_tokens, current_prefix):
            flush_current()

    flush_current()

    if not cues:
        return build_segment_fallback_cues(result)

    for idx, cue in enumerate(cues, start=1):
        cue.index = idx
    return cues


def write_direct_srt_from_result(result: dict[str, Any], srt_path: Path, cfg: RunConfig | None = None) -> None:
    cues = build_srt_cues_from_result(result, cfg)
    if not cues:
        raise RuntimeError("WhisperX returned no subtitle cues.")
    srt_path.write_text(render_srt_cues(cues), encoding="utf-8")


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
    return [line for line in text.splitlines()]


def load_translation_glossary(glossary_spec: str | None) -> dict[str, str]:
    if glossary_spec:
        candidates = [Path(glossary_spec).expanduser()]
    else:
        candidates = [project_dir() / "transcriber_glossary.txt"]

    for candidate in candidates:
        if not candidate.exists():
            if glossary_spec:
                return {}
            continue
        try:
            raw = candidate.read_text(encoding="utf-8-sig", errors="ignore")
        except OSError:
            if glossary_spec:
                return {}
            continue
        return parse_glossary_entries(raw.splitlines())
    return {}


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


def apply_glossary_placeholders(text: str, glossary: dict[str, str]) -> tuple[str, dict[str, str]]:
    if not glossary or not text:
        return text, {}

    placeholder_map: dict[str, str] = {}
    terms = sorted(glossary.items(), key=lambda item: len(item[0]), reverse=True)
    updated = text
    placeholder_index = 0
    for source, target in terms:
        if not source:
            continue
        escaped = re.escape(source)
        if re.fullmatch(r"[A-Za-z0-9_]+", source):
            pattern = rf"\b{escaped}\b"
        else:
            pattern = escaped
        if not re.search(pattern, updated):
            continue
        placeholder = f"__GLOSSARY_{placeholder_index}__"
        placeholder_index += 1
        updated = re.sub(pattern, placeholder, updated)
        placeholder_map[placeholder] = target
    return updated, placeholder_map


def replace_glossary_placeholders(text: str, placeholder_map: dict[str, str]) -> str:
    updated = text
    for placeholder, target in placeholder_map.items():
        updated = updated.replace(placeholder, target)
    return updated


def extract_between_markers(text: str, start: str, end: str) -> str | None:
    if start not in text or end not in text:
        return None
    _, after_start = text.split(start, 1)
    middle, _ = after_start.split(end, 1)
    return middle.strip()


def translation_context_for_cue(cues: Sequence[SRTCue], index: int, window: int) -> list[tuple[int, str]]:
    context: list[tuple[int, str]] = []
    for offset in range(window, 0, -1):
        prev_index = index - offset
        if prev_index < 0:
            continue
        _, prev_text = split_speaker_prefix(cues[prev_index].text)
        prev_text = normalize_subtitle_whitespace(prev_text)
        if prev_text:
            context.append((-offset, prev_text))
    for offset in range(1, window + 1):
        next_index = index + offset
        if next_index >= len(cues):
            break
        _, next_text = split_speaker_prefix(cues[next_index].text)
        next_text = normalize_subtitle_whitespace(next_text)
        if next_text:
            context.append((offset, next_text))
    return context


def build_translation_prompt(*, model_name: str, context_window: int, glossary: dict[str, str]) -> str:
    lines = [
        "Translate Spanish subtitle text to natural, accurate English.",
        f"Model: {model_name}",
        f"Context window: {context_window}",
        f"Preserve markers: {TRANSLATION_MARKER_START} ... {TRANSLATION_MARKER_END}",
        "Preserve speaker labels, names, numbers, and uncertain markers exactly.",
        "Use surrounding context only to disambiguate the current cue.",
        "Prefer faithful meaning over literal phrasing.",
    ]
    if glossary:
        lines.append("Glossary:")
        for source, target in glossary.items():
            if source == target:
                lines.append(f"- preserve: {source}")
            else:
                lines.append(f"- {source} => {target}")
    else:
        lines.append("Glossary: (none)")
    return "\n".join(lines)


def log_translation_prompt(log_path: Path, prompt: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", errors="ignore") as log:
        log.write("\n[transcriber] Translation prompt:\n")
        for line in prompt.splitlines():
            log.write(f"[transcriber] {line}\n")


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
        for source, target in sorted(glossary.items(), key=lambda item: len(item[0]), reverse=True)[:max_glossary_terms]:
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


@functools.lru_cache(maxsize=1)
def load_spanish_to_english_translator() -> tuple[Any, Any]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(SPANISH_TRANSLATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(SPANISH_TRANSLATION_MODEL)
    return tokenizer, model


def chunked_text(items: Sequence[str], size: int) -> Iterable[list[str]]:
    for idx in range(0, len(items), size):
        yield list(items[idx : idx + size])


def translate_spanish_texts(texts: Sequence[str], device: str, batch_size: int = 4) -> list[str]:
    if not texts:
        return []

    tokenizer, model = load_spanish_to_english_translator()

    try:
        import torch
    except Exception:
        torch = None

    use_cuda = bool(torch is not None and device.startswith("cuda") and getattr(torch.cuda, "is_available", lambda: False)())
    if torch is not None:
        target_device = torch.device("cuda" if use_cuda else "cpu")
        model.to(target_device)
        model.eval()
    else:
        target_device = None

    translated: list[str] = []
    for batch in chunked_text([text.strip() for text in texts], batch_size):
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        if target_device is not None:
            inputs = {key: value.to(target_device) for key, value in inputs.items()}
        with torch.no_grad() if torch is not None else contextlib.nullcontext():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=TRANSLATION_NUM_BEAMS,
                length_penalty=TRANSLATION_LENGTH_PENALTY,
                no_repeat_ngram_size=TRANSLATION_NO_REPEAT_NGRAM_SIZE,
                early_stopping=True,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated.extend(text.strip() for text in decoded)

    return translated


def translate_srt_to_english(
    srt_path: Path,
    device: str,
    glossary: dict[str, str] | None = None,
    glossary_spec: str | None = None,
    context_window: int = TRANSLATION_CONTEXT_WINDOW,
    log_path: Path | None = None,
) -> None:
    if not srt_path.exists():
        return

    srt_text = srt_path.read_text(encoding="utf-8", errors="ignore")
    cues = parse_srt_cues(srt_text)
    if not cues:
        return

    glossary = dict(glossary or {})
    file_glossary = load_translation_glossary(glossary_spec)
    if file_glossary:
        merged = dict(file_glossary)
        merged.update(glossary)
        glossary = merged

    if log_path is not None:
        prompt = build_translation_prompt(
            model_name=SPANISH_TRANSLATION_MODEL,
            context_window=context_window,
            glossary=glossary,
        )
        log_translation_prompt(log_path, prompt)

    source_blocks: list[str] = []
    block_meta: list[tuple[str, dict[str, str], str]] = []

    for idx, cue in enumerate(cues):
        prefix, current_text = split_speaker_prefix(cue.text)
        current_text = normalize_subtitle_whitespace(current_text)
        context_entries = translation_context_for_cue(cues, idx, context_window)
        block_lines = [f"[context {offset:+d}] {line}" for offset, line in context_entries]
        block_lines.append(f"{TRANSLATION_MARKER_START} {current_text} {TRANSLATION_MARKER_END}")
        block_text = "\n".join(block_lines)
        protected_text, placeholders = apply_glossary_placeholders(block_text, glossary)
        source_blocks.append(protected_text)
        block_meta.append((prefix, placeholders, current_text))

    translated_blocks = translate_spanish_texts(source_blocks, device=device)
    if len(translated_blocks) != len(cues):
        raise RuntimeError("Translation produced an unexpected cue count.")

    translated_cues: list[SRTCue] = []
    for cue, translated_block, meta in zip(cues, translated_blocks, block_meta):
        prefix, placeholders, original = meta
        translated_text = replace_glossary_placeholders(translated_block, placeholders)
        extracted = extract_between_markers(translated_text, TRANSLATION_MARKER_START, TRANSLATION_MARKER_END)
        if not extracted:
            fallback = translate_spanish_texts([original], device=device, batch_size=1)[0]
            extracted = replace_glossary_placeholders(fallback, placeholders)
        translated_text = normalize_subtitle_whitespace(extracted)
        if prefix:
            translated_text = f"{prefix}{translated_text}".strip()
        translated_cues.append(
            SRTCue(index=cue.index, start_ms=cue.start_ms, end_ms=cue.end_ms, text=translated_text)
        )

    rewrapped: list[SRTCue] = []
    for cue in translated_cues:
        rewrapped.extend(split_cue_for_subtitles(cue))

    srt_path.write_text(render_srt_cues(rewrapped or translated_cues), encoding="utf-8")


def build_llm_file(srt_path: Path, llm_path: Path) -> None:
    if not srt_path.exists():
        return

    text_lines = []
    for line in srt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.isdigit() or ("-->" in s and "," in s):
            continue
        text_lines.append(render_uncertain_markup(s, "llm"))

    preface = (
        "You are given an automatic transcript.\n"
        "Refine it into a cleaner transcript in the same language.\n"
        "Preserve speaker labels if present.\n"
        "Do not translate, summarize, or rewrite more than necessary.\n"
        "Fix obvious punctuation, capitalization, spacing, and clear recognition mistakes.\n"
        "Keep the meaning and cadence close to the source.\n"
        "Use square brackets for brief editorial notes such as [inaudible], [crosstalk], or [name unclear].\n"
        "If a word or short phrase is uncertain, italicize it and add a confidence percentage in square brackets, for example *word* [65% confidence].\n"
        "If you are not confident enough to refine a passage cleanly, keep it cautious instead of guessing.\n"
        "Output only the refined transcript.\n\n"
        "TRANSCRIPT:\n"
    )
    llm_path.write_text(preface + "\n".join(text_lines), encoding="utf-8")


def finalize_srt_file(srt_path: Path) -> None:
    if not srt_path.exists():
        return

    text = srt_path.read_text(encoding="utf-8", errors="ignore")
    updated_lines: list[str] = []
    for line in text.splitlines():
        updated_lines.append(render_uncertain_markup(line, "srt"))
    srt_path.write_text("\n".join(updated_lines).rstrip() + "\n", encoding="utf-8")


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


def build_audio_preprocess_command(input_path: Path, output_path: Path) -> list[str]:
    return [
        "ffmpeg",
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


def preprocess_audio_for_whisperx(input_path: Path, temp_dir: Path, report: Reporter = print) -> Path:
    output_path = temp_dir / f"{input_path.stem}.preprocessed.wav"
    command = build_audio_preprocess_command(input_path, output_path)

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        report("[transcriber] ffmpeg not found; using the original input audio.")
        return input_path
    except subprocess.CalledProcessError as exc:
        detail = f": {exc.stderr.strip()}" if exc.stderr and exc.stderr.strip() else ""
        report(f"[transcriber] Audio preprocessing failed; using the original input audio{detail}")
        with contextlib.suppress(OSError):
            output_path.unlink()
        return input_path
    except Exception as exc:
        report(f"[transcriber] Audio preprocessing failed; using the original input audio: {exc}")
        with contextlib.suppress(OSError):
            output_path.unlink()
        return input_path

    if not output_path.exists() or output_path.stat().st_size == 0:
        report("[transcriber] Audio preprocessing produced no output; using the original input audio.")
        with contextlib.suppress(OSError):
            output_path.unlink()
        return input_path

    return output_path


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WhisperX one-click transcriber (quality + fast modes).")
    parser.add_argument("legacy", nargs="*", help="Legacy tokens: [language] [model] [mode]")
    parser.add_argument("--input", "-i", help="Audio/video file path.")
    parser.add_argument("--lang", choices=("auto", "en", "es"), help="Language: auto, en, or es.")
    parser.add_argument(
        "--glossary",
        action="append",
        default=[],
        help="Glossary entry: 'source=target' or 'source' to preserve. Repeatable.",
    )
    parser.add_argument(
        "--glossary-file",
        help=(
            "Glossary text file (one entry per line). Lines can use 'source => target', tab-separated pairs, or 'source | target'."
        ),
    )
    parser.add_argument(
        "--asr-prompt",
        help="Optional text prompt to bias WhisperX toward names and jargon.",
    )
    parser.add_argument(
        "--asr-prompt-file",
        help="Text file with extra ASR prompt lines for names and jargon.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Single decoding temperature. Overrides the preset temperature schedule.",
    )
    parser.add_argument(
        "--temperature-schedule",
        help="Comma-separated fallback temperatures, e.g. 0.0,0.2,0.4. Overrides --temperature.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        help="Sampling candidates when temperature is above 0.",
    )
    parser.add_argument(
        "--compression-ratio-threshold",
        type=float,
        help="Fallback when output compression ratio exceeds this value.",
    )
    parser.add_argument(
        "--logprob-threshold",
        type=float,
        help="Fallback when average log probability drops below this value.",
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        help="Fallback when no-speech probability exceeds this value.",
    )
    parser.add_argument(
        "--condition-on-previous-text",
        dest="condition_on_previous_text",
        action="store_true",
        default=None,
        help="Condition each decode window on the previous text.",
    )
    parser.add_argument(
        "--no-condition-on-previous-text",
        dest="condition_on_previous_text",
        action="store_false",
        help="Decode each window without conditioning on previous text.",
    )
    parser.add_argument("--mode", choices=("quality", "fast"), help="Run mode preset: quality or fast.")
    parser.add_argument("--model", help="Whisper model name, e.g. large-v3, medium.")
    parser.add_argument("--device", default="cuda", help="Inference device (default: cuda).")
    parser.add_argument(
        "--compute-type",
        default="float16",
        choices=("float16", "float32", "int8"),
        help="Computation dtype (default: float16).",
    )
    parser.add_argument("--watch", action="store_true", help="Continuously watch a folder and transcribe new media files.")
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
    parser.add_argument(
        "--stale-lock-seconds",
        type=float,
        default=DEFAULT_STALE_LOCK_SECONDS,
        help=(
            "Age after which a lock file is treated as stale and may be cleared before processing "
            f"(default: {DEFAULT_STALE_LOCK_SECONDS:g})."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Show planned actions without running WhisperX or writing outputs.")
    diarize_group = parser.add_mutually_exclusive_group()
    diarize_group.add_argument("--diarize", dest="force_diarize", action="store_true", help="Force diarization on.")
    diarize_group.add_argument("--no-diarize", dest="force_no_diarize", action="store_true", help="Force diarization off.")
    parser.add_argument("--no-diarize-smoothing", action="store_true", help="Disable speaker diarization smoothing.")
    parser.add_argument(
        "--min-speaker-turn-ms",
        type=int,
        default=DEFAULT_MIN_SPEAKER_TURN_MS,
        help=f"Minimum speaker turn duration for smoothing (default: {DEFAULT_MIN_SPEAKER_TURN_MS}).",
    )
    parser.add_argument(
        "--min-speaker-turn-tokens",
        type=int,
        default=DEFAULT_MIN_SPEAKER_TURN_TOKENS,
        help=f"Minimum speaker turn token count for smoothing (default: {DEFAULT_MIN_SPEAKER_TURN_TOKENS}).",
    )
    parser.add_argument(
        "--confidence-cleanup",
        dest="confidence_cleanup",
        action="store_true",
        default=True,
        help="Enable low-confidence transcript cleanup (default: on).",
    )
    parser.add_argument(
        "--no-confidence-cleanup",
        dest="confidence_cleanup",
        action="store_false",
        help="Disable low-confidence transcript cleanup.",
    )
    parser.add_argument(
        "--confidence-cleanup-mode",
        choices=("mark", "redact"),
        default="mark",
        help="How to handle low-confidence regions (default: mark).",
    )
    parser.add_argument(
        "--low-confidence-logprob",
        type=float,
        default=DEFAULT_LOW_CONFIDENCE_LOGPROB,
        help=f"Avg logprob threshold for low confidence (default: {DEFAULT_LOW_CONFIDENCE_LOGPROB}).",
    )
    parser.add_argument(
        "--high-no-speech-prob",
        type=float,
        default=DEFAULT_HIGH_NO_SPEECH_PROB,
        help=f"No-speech probability threshold (default: {DEFAULT_HIGH_NO_SPEECH_PROB}).",
    )
    parser.add_argument(
        "--low-confidence-word-prob",
        type=float,
        default=DEFAULT_LOW_CONFIDENCE_WORD_PROB,
        help=f"Word confidence threshold (default: {DEFAULT_LOW_CONFIDENCE_WORD_PROB}).",
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
            opt.language = "es"
            opt.language_locked = True
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


def prompt_input_path() -> str | None:
    if not sys.stdin.isatty():
        return None
    raw = input("Media file path (or Enter to cancel): ").strip()
    return raw or None


def prompt_language(current: str) -> str:
    if not sys.stdin.isatty():
        return current
    choice = input("Language [Enter=Auto, e=English, s=Spanish]: ").strip().lower()
    if choice in {"", "a", "auto"}:
        return "auto"
    if choice in {"s", "es"}:
        return "es"
    if choice in {"e", "en"}:
        return "en"
    return current


def prompt_mode(current: str) -> str:
    if not sys.stdin.isatty():
        return current
    choice = input("Run mode [Enter=quality, f=fast]: ").strip().lower()
    if choice in {"f", "fast"}:
        return "fast"
    if choice in {"q", "quality", ""}:
        return "quality"
    return current


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
        p = Path(arg_input).expanduser()
        return p if p.exists() else p

    chosen = pick_media_file()
    if chosen:
        return Path(chosen)

    manual = prompt_input_path()
    if manual:
        return Path(manual).expanduser()

    return None


def build_config(args: argparse.Namespace, interactive: bool = True) -> RunConfig:
    legacy = parse_legacy(args.legacy)

    language = args.lang or legacy.language or "auto"
    language_locked = bool(args.lang) or legacy.language_locked

    mode = args.mode or legacy.mode or "quality"
    mode_locked = bool(args.mode) or legacy.mode_locked

    model = args.model or legacy.model
    model_locked = bool(args.model) or legacy.model_locked

    if interactive and not language_locked:
        language = prompt_language(language)
    if interactive and not mode_locked:
        mode = prompt_mode(mode)

    if mode == "fast":
        if not model_locked:
            model = "medium"
        batch_size = 16
        beam_size = 2
        patience = 1.0
        temperature = 0.0
        temperature_schedule = (0.0,)
        best_of = 1
        compression_ratio_threshold = 2.4
        logprob_threshold = -1.0
        no_speech_threshold = 0.6
        condition_on_previous_text = False
        diarize_default = False
    else:
        if not model_locked:
            model = "large-v3"
        batch_size = 8
        beam_size = 8
        patience = 1.2
        temperature = 0.0
        temperature_schedule = (0.0, 0.2, 0.4, 0.6, 0.8)
        best_of = 5
        compression_ratio_threshold = 2.4
        logprob_threshold = -1.0
        no_speech_threshold = 0.6
        condition_on_previous_text = True
        diarize_default = True

    if args.temperature is not None:
        temperature = float(args.temperature)
        temperature_schedule = (temperature,)

    try:
        override_schedule = parse_temperature_schedule(args.temperature_schedule)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if override_schedule:
        temperature_schedule = override_schedule
        temperature = override_schedule[0]

    if args.best_of is not None:
        best_of = max(1, int(args.best_of))
    if args.compression_ratio_threshold is not None:
        compression_ratio_threshold = float(args.compression_ratio_threshold)
    if args.logprob_threshold is not None:
        logprob_threshold = float(args.logprob_threshold)
    if args.no_speech_threshold is not None:
        no_speech_threshold = float(args.no_speech_threshold)
    if args.condition_on_previous_text is not None:
        condition_on_previous_text = bool(args.condition_on_previous_text)

    if args.force_diarize:
        diarize = True
    elif args.force_no_diarize:
        diarize = False
    else:
        diarize = diarize_default

    diarize_smoothing = not args.no_diarize_smoothing

    glossary_entries: list[str] = list(args.glossary or [])
    glossary_files: list[Path] = []
    if args.glossary_file:
        glossary_files.append(Path(args.glossary_file).expanduser())
    for entry in list(glossary_entries):
        if looks_like_glossary_file(entry):
            glossary_files.append(Path(entry).expanduser())
            glossary_entries.remove(entry)

    file_entries: list[str] = []
    for glossary_file in glossary_files:
        file_entries.extend(load_glossary_file(glossary_file))
    glossary = parse_glossary_entries([*file_entries, *glossary_entries])
    glossary_path = str(glossary_files[0]) if glossary_files else None
    asr_prompt = build_asr_prompt(
        glossary=glossary,
        prompt_text=args.asr_prompt,
        prompt_file=args.asr_prompt_file,
    )

    return RunConfig(
        language=language,
        translate_to_english=language == "es",
        mode=mode,
        model=model or "large-v3",
        batch_size=batch_size,
        beam_size=beam_size,
        patience=patience,
        temperature=temperature,
        temperature_schedule=temperature_schedule,
        best_of=best_of,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold,
        condition_on_previous_text=condition_on_previous_text,
        diarize=diarize,
        diarize_smoothing=diarize_smoothing,
        min_speaker_turn_ms=max(0, int(args.min_speaker_turn_ms)),
        min_speaker_turn_tokens=max(0, int(args.min_speaker_turn_tokens)),
        confidence_cleanup=bool(args.confidence_cleanup),
        confidence_cleanup_mode=args.confidence_cleanup_mode,
        low_confidence_logprob=float(args.low_confidence_logprob),
        high_no_speech_prob=float(args.high_no_speech_prob),
        low_confidence_word_prob=float(args.low_confidence_word_prob),
        device=args.device,
        compute_type=args.compute_type,
        glossary=glossary,
        glossary_path=glossary_path,
        asr_prompt=asr_prompt,
        dry_run=bool(args.dry_run),
    )


def output_paths_for_input(input_path: Path, cfg: RunConfig, create_dirs: bool = False) -> OutputPaths:
    output_dir = input_path.parent
    log_dir = project_dir() / LOG_DIR_NAME
    if create_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
    base = input_path.stem
    return OutputPaths(
        output_dir=output_dir,
        srt_path=output_dir / f"{base}.srt",
        llm_path=output_dir / f"{base}_llm.txt",
        log_path=log_dir / f"{base}_whisperx.log",
        lock_path=output_dir / f"{base}{LOCK_SUFFIX}",
    )


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


def should_fallback_without_diarization(log_path: Path) -> bool:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        return False
    return any(pattern in text for pattern in DIARIZATION_FALLBACK_PATTERNS)


def parse_detected_language_from_log(log_path: Path) -> str | None:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None

    matches = re.findall(r"Detected language:\s*([A-Za-z-]+)\s*\(", text)
    if not matches:
        return None
    return matches[-1].strip().lower() or None


def run_whisperx_direct(
    cfg: RunConfig,
    input_path: Path,
    srt_path: Path,
    hf_token: str | None,
    diarize: bool,
) -> str | None:
    import whisperx

    try:
        import torch
    except Exception:
        torch = None

    whisper_language = None if cfg.language == "auto" else cfg.language

    print(f"[transcriber] Loading model {cfg.model} on {cfg.device}...")
    asr_options = {"beam_size": cfg.beam_size, "patience": cfg.patience}
    load_model_kwargs: dict[str, Any] = {
        "device": cfg.device,
        "compute_type": cfg.compute_type,
        "task": "transcribe",
        "asr_options": asr_options,
        "vad_method": "silero",
    }
    if whisper_language:
        load_model_kwargs["language"] = whisper_language
    model = call_with_supported_kwargs(whisperx.load_model, cfg.model, **load_model_kwargs)

    print(f"[transcriber] Loading audio: {input_path}")
    audio = whisperx.load_audio(str(input_path))

    print("[transcriber] Transcribing audio...")
    transcribe_kwargs: dict[str, Any] = {
        "batch_size": cfg.batch_size,
        "task": "transcribe",
        "temperature": cfg.temperature_schedule or cfg.temperature,
        "print_progress": False,
        "condition_on_previous_text": cfg.condition_on_previous_text,
    }
    if cfg.asr_prompt:
        transcribe_kwargs["initial_prompt"] = cfg.asr_prompt
    if cfg.best_of is not None:
        transcribe_kwargs["best_of"] = cfg.best_of
    if cfg.compression_ratio_threshold is not None:
        transcribe_kwargs["compression_ratio_threshold"] = cfg.compression_ratio_threshold
    if cfg.logprob_threshold is not None:
        transcribe_kwargs["logprob_threshold"] = cfg.logprob_threshold
    if cfg.no_speech_threshold is not None:
        transcribe_kwargs["no_speech_threshold"] = cfg.no_speech_threshold
    if whisper_language:
        transcribe_kwargs["language"] = whisper_language
    result = call_with_supported_kwargs(model.transcribe, audio, **transcribe_kwargs)
    if not isinstance(result, dict):
        raise RuntimeError(f"Unexpected WhisperX transcription result type: {type(result)!r}")

    detected_language = str(result.get("language") or "").strip().lower() or None

    print(f"[transcriber] Aligning words for language={(detected_language or whisper_language or 'en')}...")
    try:
        language_code = detected_language or whisper_language or "en"
        align_model, metadata = call_with_supported_kwargs(
            whisperx.load_align_model,
            language_code=language_code,
            device=cfg.device,
        )
        result = call_with_supported_kwargs(
            whisperx.align,
            result.get("segments", []),
            align_model,
            metadata,
            audio,
            cfg.device,
            return_char_alignments=False,
        )
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected WhisperX alignment result type: {type(result)!r}")
    except Exception as exc:
        print(f"[transcriber] Alignment failed; continuing without alignment: {exc}")

    if diarize:
        print("[transcriber] Running diarization...")
        diarization_pipeline = resolve_whisperx_symbol(whisperx, "DiarizationPipeline")
        with allow_trusted_checkpoint_loads():
            diarize_model = call_with_supported_kwargs(
                diarization_pipeline,
                use_auth_token=hf_token or "",
                device=cfg.device,
            )
            diarize_segments = call_with_supported_kwargs(diarize_model, audio)
        assign_word_speakers = resolve_whisperx_symbol(whisperx, "assign_word_speakers")
        result = assign_word_speakers(diarize_segments, result)

    apply_confidence_cleanup(result, cfg)

    print("[transcriber] Writing subtitle-sized SRT from in-memory timings...")
    write_direct_srt_from_result(result, srt_path, cfg)

    if torch is not None and getattr(torch, "cuda", None) is not None:
        with contextlib.suppress(Exception):
            torch.cuda.empty_cache()

    return detected_language


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
    with log_path.open(mode, encoding="utf-8", errors="ignore") as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            try:
                detected_language = run_whisperx_direct(cfg, input_path, srt_path, hf_token, diarize)
                if detected_language is None:
                    detected_language = parse_detected_language_from_log(log_path)
                return 0, detected_language
            except Exception:
                import traceback

                traceback.print_exc()
                return 1, None


def print_summary(cfg: RunConfig, input_path: Path, outputs: OutputPaths, report: Reporter = print) -> None:
    report("")
    report(f'Input:     "{input_path}"')
    report(f'Output:    "{outputs.srt_path}"')
    report(f'LLM:       "{outputs.llm_path}"')
    report(f'Log:       "{outputs.log_path}"')
    report(f'Lock:      "{outputs.lock_path}"')
    report(f'OutDir:    "{outputs.output_dir}"')
    report(f"Lang:      {cfg.language}")
    if cfg.language == "auto":
        report("Translate: auto (Spanish -> English when detected)")
    else:
        report(f"Translate: {'on' if cfg.translate_to_english else 'off'}")
    report(f"Mode:      {cfg.mode}")
    report(f"Model:     {cfg.model}")
    report(f'Diarize:   {"on" if cfg.diarize else "off"}')
    report(f'Smooth:    {"on" if (cfg.diarize and cfg.diarize_smoothing) else "off"}')
    report(
        f"Cleanup:   {'on' if cfg.confidence_cleanup else 'off'}"
        f"{' (' + cfg.confidence_cleanup_mode + ')' if cfg.confidence_cleanup else ''}"
    )
    if cfg.glossary_path:
        report(f'Glossary:  "{cfg.glossary_path}"')
    if cfg.asr_prompt:
        report("ASRPrompt: on")
    report(
        "Decode:    "
        f"temps={','.join(f'{temp:g}' for temp in cfg.temperature_schedule) if cfg.temperature_schedule else f'{cfg.temperature:g}'} "
        f"best_of={cfg.best_of if cfg.best_of is not None else 'auto'} "
        f"cr={cfg.compression_ratio_threshold if cfg.compression_ratio_threshold is not None else 'auto'} "
        f"logprob={cfg.logprob_threshold if cfg.logprob_threshold is not None else 'auto'} "
        f"no_speech={cfg.no_speech_threshold if cfg.no_speech_threshold is not None else 'auto'} "
        f"prev_text={'on' if cfg.condition_on_previous_text else 'off'}"
    )
    if cfg.dry_run:
        report("DryRun:    on")
    report("")


def describe_dry_run_plan(cfg: RunConfig, input_path: Path, outputs: OutputPaths, report: Reporter) -> None:
    report("Dry run. No files will be changed.")
    report(f'Would acquire lock: "{outputs.lock_path}"')
    report(f'Would run WhisperX with output dir: "{outputs.output_dir}"')
    report(f'Would write transcript: "{outputs.srt_path}"')
    report(f'Would write LLM prompt file: "{outputs.llm_path}"')
    report(f'Would write log: "{outputs.log_path}"')
    report(f'Would leave the source file in place: "{input_path}"')


def transcribe_file(
    cfg: RunConfig,
    input_path: Path,
    report: Reporter = print,
    stale_lock_seconds: float = DEFAULT_STALE_LOCK_SECONDS,
) -> int:
    if not input_path.exists():
        report("")
        report(f'File not found: "{input_path}"')
        report("")
        return 1

    outputs = output_paths_for_input(input_path, cfg, create_dirs=not cfg.dry_run)
    print_summary(cfg, input_path, outputs, report=report)

    if cfg.dry_run:
        describe_dry_run_plan(cfg, input_path, outputs, report)
        return 0

    if not acquire_lock(input_path, outputs.lock_path, stale_lock_seconds, report):
        return 0

    hf_token: str | None = None
    fallback_no_diarize = False
    current_diarize = cfg.diarize
    detected_language: str | None = None
    rc = 1

    try:
        if cfg.diarize:
            hf_token = load_hf_token(project_dir())
            if not hf_token:
                report("")
                report("Missing Hugging Face token.")
                report(f'Create "{project_dir() / "hf_token.txt"}" (or "HF_TOKEN.txt") or set HF_TOKEN.')
                report("")
                return 1

        with tempfile.TemporaryDirectory(prefix="transcriber-audio-") as temp_audio_dir:
            prepared_input = preprocess_audio_for_whisperx(input_path, Path(temp_audio_dir), report=report)
            if prepared_input != input_path:
                report(f'Audio preprocessing: "{prepared_input}"')

            attempt = 0
            while True:
                rc, detected_language = run_whisperx_direct_logged(
                    cfg,
                    prepared_input,
                    outputs.srt_path,
                    hf_token,
                    diarize=current_diarize,
                    log_path=outputs.log_path,
                    append=attempt > 0,
                )
                if rc == 0:
                    break

                if current_diarize and should_fallback_without_diarization(outputs.log_path):
                    fallback_no_diarize = True
                    current_diarize = False
                    report("")
                    report("Diarization unavailable or blocked. Retrying without diarization...")
                    with outputs.log_path.open("a", encoding="utf-8", errors="ignore") as log:
                        log.write("\n[transcriber] Diarization unavailable or blocked; retrying without --diarize.\n")
                    attempt += 1
                    continue

                break

        if outputs.srt_path.exists():
            should_translate = cfg.translate_to_english or (cfg.language == "auto" and detected_language == "es")
            if cfg.language == "auto":
                report(f"Detected language: {detected_language or 'unknown'}.")
            if should_translate:
                if cfg.glossary_path and not Path(cfg.glossary_path).expanduser().exists():
                    report(f'Glossary file not found: "{cfg.glossary_path}"')
                report("Translating Spanish transcript to English text...")
                try:
                    translate_srt_to_english(
                        outputs.srt_path,
                        cfg.device,
                        glossary=cfg.glossary,
                        glossary_spec=cfg.glossary_path,
                        context_window=TRANSLATION_CONTEXT_WINDOW,
                        log_path=outputs.log_path,
                    )
                except Exception as exc:
                    report("")
                    report(f"Translation failed: {exc}")
                    report("Keeping the original transcript text.")
                    report("")
            try:
                build_llm_file(outputs.srt_path, outputs.llm_path)
                finalize_srt_file(outputs.srt_path)
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
            report("")
            return 0

        report("")
        report("Done, but SRT not found where expected:")
        report(f'  "{outputs.srt_path}"')
        report("Check the log:")
        report(f'  "{outputs.log_path}"')
        report("")
        return 0
    finally:
        release_lock(outputs.lock_path, report=report)


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

    watcher_log_path = project_dir() / LOG_DIR_NAME / WATCHER_LOG_NAME
    report = make_watch_reporter(watcher_log_path)
    report(f'Watcher started for "{watch_dir}".')
    report(
        "Defaults: "
        f"lang={cfg.language}, mode={cfg.mode}, model={cfg.model}, "
        f'diarize={"on" if cfg.diarize else "off"}, device={cfg.device}, '
        f"compute_type={cfg.compute_type}."
    )
    report(
        f"Polling every {poll_interval:g}s. A file must stay unchanged for {settle_seconds:g}s before transcription starts."
    )

    pending: dict[str, PendingWatchFile] = {}
    while True:
        try:
            current_paths: set[str] = set()
            now = time.monotonic()

            for path in iter_watch_candidates(watch_dir):
                key = str(path.resolve())
                current_paths.add(key)

                if not needs_transcription(path, cfg):
                    pending.pop(key, None)
                    continue

                try:
                    size, mtime_ns = file_signature(path)
                except OSError:
                    pending.pop(key, None)
                    continue

                pending_file = pending.get(key)
                if pending_file is None:
                    pending[key] = PendingWatchFile(size=size, mtime_ns=mtime_ns, stable_since=now)
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
                outputs = output_paths_for_input(path, cfg)
                if rc == 0 and outputs.srt_path.exists():
                    pending.pop(key, None)
                    report(f'Finished "{path.name}" -> "{outputs.srt_path.name}".')
                else:
                    report(
                        f'Failed "{path.name}". Will retry in {WATCH_RETRY_COOLDOWN_SECONDS:g}s if the transcript is still missing.'
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
