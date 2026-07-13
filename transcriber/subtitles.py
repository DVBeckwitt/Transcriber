"""Subtitle parsing, confidence cleanup, speaker smoothing, and SRT rendering."""

from __future__ import annotations

import itertools
import math
import re
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol


class TranscriptConfig(Protocol):
    """Configuration fields required by subtitle processing."""

    confidence_cleanup: bool
    low_confidence_logprob: float
    high_no_speech_prob: float
    low_confidence_word_prob: float
    diarize_smoothing: bool
    min_speaker_turn_ms: int
    min_speaker_turn_tokens: int
    include_speaker_labels: bool


SUBTITLE_MAX_LINES = 2
SUBTITLE_MAX_CHARS_PER_LINE = 42
SUBTITLE_MAX_DURATION_SECONDS = 6.0
SUBTITLE_TARGET_CPS = 17.0
SUBTITLE_PREFERRED_BREAK_CHARS = ".?!,:;"
LEGACY_LOW_CONFIDENCE_MARKER = "UNC" + "ERTAIN"
LOW_CONFIDENCE_MARKER_RE = re.compile(
    rf"__(?:LOWCONF|{LEGACY_LOW_CONFIDENCE_MARKER})(?:_(\d+))?__(.*?)__(?:LOWCONF|{LEGACY_LOW_CONFIDENCE_MARKER})_END__",
    re.DOTALL,
)
LOW_CONFIDENCE_MARKER_NOISE_HINTS = (
    "unc" + "ert",
    "cerain",
    "ciert",
    "certain",
    "lowconf",
    "end",
)
LOW_CONFIDENCE_MARKER_NOISE_WORDS = {
    "unc" + "ertain",
    "uncerain",
    "unc" + "ertaint",
    "uncierta",
    "certain",
    "certaint",
    "cierta",
    "ccertain",
    "end",
    "un",
}
DEFAULT_MIN_SPEAKER_TURN_MS = 900
DEFAULT_MIN_SPEAKER_TURN_TOKENS = 2
DEFAULT_LOW_CONFIDENCE_LOGPROB = -1.0
DEFAULT_HIGH_NO_SPEECH_PROB = 0.6
DEFAULT_LOW_CONFIDENCE_WORD_PROB = 0.10
LOW_CONFIDENCE_PLACEHOLDER = "—"


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
class SpeakerRun:
    speaker: str
    tokens: list[TimedToken]
    duration_ms: int
    average_confidence: float | None
    has_low_confidence: bool

    @property
    def word_count(self) -> int:
        return len(self.tokens)

    @property
    def score(self) -> int:
        return self.duration_ms + self.word_count * 250


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
        start_raw, end_raw = [
            part.strip() for part in lines[line_index].split("-->", 1)
        ]
        text = " ".join(
            line.strip() for line in lines[line_index + 1 :] if line.strip()
        )
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
        rendered.append(
            f"{ms_to_timestamp(cue.start_ms)} --> {ms_to_timestamp(cue.end_ms)}"
        )
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
        if (
            wrapped
            and len(wrapped) <= max_lines
            and max(len(line) for line in wrapped) <= max_chars_per_line
        ):
            return wrapped
    return []


def _initial_text_chunks(
    words: Sequence[str], soft_limit: int, hard_limit: int
) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            chunks.append(" ".join(current))
            current.clear()

    for index, word in enumerate(words):
        tentative = " ".join((*current, word))
        if current and len(tentative) > hard_limit:
            flush()
            current.append(word)
            continue

        current.append(word)
        current_text = " ".join(current)
        next_word = words[index + 1] if index + 1 < len(words) else ""
        ends_sentence = word.endswith(tuple(SUBTITLE_PREFERRED_BREAK_CHARS))

        if len(current_text) >= soft_limit and ends_sentence:
            flush()
        elif next_word and len(f"{current_text} {next_word}") > hard_limit:
            flush()

    flush()
    return chunks


def _merge_short_text_chunks(
    chunks: Sequence[str], soft_limit: int, hard_limit: int
) -> list[str]:
    merged: list[str] = []
    minimum_middle_length = max(12, soft_limit // 3)

    for chunk in chunks:
        can_merge = (
            merged
            and len(chunk) < minimum_middle_length
            and len(merged[-1]) + 1 + len(chunk) <= hard_limit
        )
        if can_merge:
            merged[-1] = f"{merged[-1]} {chunk}"
        else:
            merged.append(chunk)

    minimum_final_length = max(12, hard_limit // 5)
    while len(merged) > 1 and len(merged[-1]) < minimum_final_length:
        combined = f"{merged[-2]} {merged[-1]}"
        if len(combined) > hard_limit:
            break
        merged[-2:] = [combined]
    return merged


def split_text_into_chunks(text: str, soft_limit: int, hard_limit: int) -> list[str]:
    words = normalize_subtitle_whitespace(text).split()
    if not words:
        return []

    chunks = _initial_text_chunks(words, soft_limit, hard_limit)
    return _merge_short_text_chunks(chunks, soft_limit, hard_limit)


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
    chunks = split_text_into_chunks(
        content, soft_limit=min(soft_limit, hard_limit), hard_limit=hard_limit
    )

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
        new_cues.append(
            SRTCue(index=idx + 1, start_ms=start_ms, end_ms=end_ms, text=wrapped_text)
        )
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


def segment_is_low_confidence(
    segment: dict[str, Any],
    *,
    low_logprob: float,
    high_no_speech: float,
) -> bool:
    avg_logprob = segment.get("avg_logprob")
    if isinstance(avg_logprob, int | float) and avg_logprob < low_logprob:
        return True
    no_speech_prob = segment.get("no_speech_prob")
    if isinstance(no_speech_prob, int | float) and no_speech_prob > high_no_speech:
        return True
    return False


def word_is_low_confidence(word: dict[str, Any], *, low_prob: float) -> bool:
    for key in ("probability", "confidence", "score"):
        value = word.get(key)
        if isinstance(value, int | float) and value < low_prob:
            return True
    return False


def apply_confidence_cleanup(result: dict[str, Any], cfg: TranscriptConfig) -> None:
    if not cfg.confidence_cleanup:
        return

    segments = result.get("segments", [])
    if not isinstance(segments, list):
        return

    for segment in segments:
        if not isinstance(segment, dict):
            continue
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
                word_low = (
                    word_is_low_confidence(word, low_prob=cfg.low_confidence_word_prob)
                    or seg_low
                )
                if not word_low:
                    continue
                word["_low_confidence"] = True


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
                low_confidence=bool(raw_word.get("_low_confidence"))
                or seg_low_confidence,
                confidence=probability_value(
                    raw_word.get("probability") or raw_word.get("score")
                ),
            )
        )

    if timed_words:
        return timed_words

    seg_text = normalize_subtitle_whitespace(str(segment.get("text") or ""))
    seg_start_ms = seconds_to_ms(segment.get("start"))
    seg_end_ms = seconds_to_ms(segment.get("end"))
    if (
        not seg_text
        or seg_start_ms is None
        or seg_end_ms is None
        or seg_end_ms <= seg_start_ms
    ):
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
                confidence=probability_value(
                    segment.get("probability") or segment.get("score")
                ),
            )
        )
        cursor_ms = next_ms
    return distributed


def token_low_confidence_marker(token: TimedToken) -> str:
    if token.confidence is None:
        return f"__LOWCONF__{token.text}__LOWCONF_END__"
    percent = max(0, min(100, int(round(float(token.confidence) * 100.0))))
    return f"__LOWCONF_{percent}__{token.text}__LOWCONF_END__"


def format_token_text(token: TimedToken, style: str = "plain") -> str:
    if not token.text:
        return ""
    if style == "marker" and token.low_confidence:
        return token_low_confidence_marker(token)
    return token.text


def format_cue_text(
    prefix: str, tokens: Sequence[TimedToken], style: str = "plain"
) -> str:
    body = normalize_subtitle_whitespace(
        " ".join(
            format_token_text(token, style=style) for token in tokens if token.text
        )
    )
    return normalize_subtitle_whitespace(f"{prefix}{body}") if prefix else body


def render_low_confidence_markup(text: str, style: str) -> str:
    if not text:
        return text

    def replace(match: re.Match[str]) -> str:
        return LOW_CONFIDENCE_PLACEHOLDER

    updated = LOW_CONFIDENCE_MARKER_RE.sub(replace, text)
    if style == "srt":
        updated = strip_low_confidence_marker_noise(updated)
        return normalize_subtitle_whitespace(updated)
    return updated


def strip_low_confidence_marker_noise(text: str) -> str:
    if not text:
        return text

    def clean_token(token: str) -> str:
        low = token.lower()
        if not (
            "__" in token
            or (
                "_" in token
                and any(hint in low for hint in LOW_CONFIDENCE_MARKER_NOISE_HINTS)
            )
        ):
            return token

        parts = re.split(r"_+", token)
        kept: list[str] = []
        for part in parts:
            cleaned = re.sub(r"[^a-z]+", "", part.lower())
            if not cleaned:
                continue
            if cleaned.isdigit() or cleaned in LOW_CONFIDENCE_MARKER_NOISE_WORDS:
                continue
            kept.append(part)
        return " ".join(kept)

    cleaned = " ".join(filter(None, (clean_token(token) for token in text.split())))
    return normalize_subtitle_whitespace(cleaned)


def _speaker_run(tokens: list[TimedToken]) -> SpeakerRun:
    confidences = [token.confidence for token in tokens if token.confidence is not None]
    average_confidence = sum(confidences) / len(confidences) if confidences else None
    return SpeakerRun(
        speaker=tokens[0].speaker,
        tokens=tokens,
        duration_ms=max(1, tokens[-1].end_ms - tokens[0].start_ms),
        average_confidence=average_confidence,
        has_low_confidence=any(token.low_confidence for token in tokens),
    )


def _neighbor_speaker(previous: SpeakerRun, following: SpeakerRun) -> str:
    if previous.speaker and previous.speaker == following.speaker:
        return previous.speaker
    if previous.speaker and following.speaker:
        return (
            previous.speaker if previous.score >= following.score else following.speaker
        )
    return previous.speaker or following.speaker


def smooth_timed_tokens(
    tokens: Sequence[TimedToken],
    min_run_duration_ms: int = DEFAULT_MIN_SPEAKER_TURN_MS,
    min_run_words: int = DEFAULT_MIN_SPEAKER_TURN_TOKENS,
    low_confidence_threshold: float = DEFAULT_LOW_CONFIDENCE_WORD_PROB,
) -> list[TimedToken]:
    """Replace brief or uncertain speaker flips with the stronger neighbor."""
    if not tokens:
        return []

    copied_tokens = [replace(token) for token in tokens]
    runs = [
        _speaker_run(list(group))
        for _, group in itertools.groupby(
            copied_tokens, key=lambda token: token.speaker
        )
    ]
    if len(runs) < 3:
        return copied_tokens

    for previous, current, following in zip(runs, runs[1:], runs[2:], strict=False):
        is_short = (
            current.duration_ms < min_run_duration_ms
            or current.word_count <= min_run_words
        )
        is_uncertain = current.has_low_confidence or (
            current.average_confidence is not None
            and current.average_confidence < low_confidence_threshold
        )
        if not current.speaker or not (is_short or is_uncertain):
            continue

        replacement_speaker = _neighbor_speaker(previous, following)
        if not replacement_speaker or replacement_speaker == current.speaker:
            continue
        for token in current.tokens:
            token.speaker = replacement_speaker

    return [token for run in runs for token in run.tokens]


def extract_timed_tokens(
    result: dict[str, Any], cfg: TranscriptConfig | None = None
) -> list[TimedToken]:
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

    min_run_ms = (
        cfg.min_speaker_turn_ms if cfg is not None else DEFAULT_MIN_SPEAKER_TURN_MS
    )
    min_run_tokens = (
        cfg.min_speaker_turn_tokens
        if cfg is not None
        else DEFAULT_MIN_SPEAKER_TURN_TOKENS
    )
    low_confidence_threshold = (
        cfg.low_confidence_word_prob
        if cfg is not None
        else DEFAULT_LOW_CONFIDENCE_WORD_PROB
    )
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
    near_limit = (
        duration_ms >= 4200
        or reading_speed_cps(display_text, duration_ms) >= SUBTITLE_TARGET_CPS * 0.95
    )
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


def build_segment_fallback_cues(
    result: dict[str, Any], include_speaker_labels: bool = True
) -> list[SRTCue]:
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
        prefix = (
            speaker_prefix(raw_segment.get("speaker")) if include_speaker_labels else ""
        )
        cues.extend(
            split_cue_for_subtitles(
                SRTCue(
                    index=0, start_ms=start_ms, end_ms=end_ms, text=f"{prefix}{text}"
                )
            )
        )

    for idx, cue in enumerate(cues, start=1):
        cue.index = idx
    return cues


@dataclass
class TimedCueAccumulator:
    include_speaker_labels: bool
    cues: list[SRTCue]
    current_tokens: list[TimedToken]
    current_speaker: str = ""

    @classmethod
    def create(cls, include_speaker_labels: bool) -> TimedCueAccumulator:
        return cls(include_speaker_labels, [], [])

    @property
    def prefix(self) -> str:
        if not self.include_speaker_labels:
            return ""
        return speaker_prefix(self.current_speaker)

    def start(self, token: TimedToken, speaker: str) -> None:
        self.current_tokens = [token]
        self.current_speaker = speaker

    def flush(self) -> None:
        if not self.current_tokens:
            return
        cue = finalize_timed_cue(
            len(self.cues) + 1,
            self.prefix,
            self.current_tokens,
        )
        self.cues.append(cue)
        self.current_tokens = []
        self.current_speaker = ""

    def add(self, token: TimedToken) -> None:
        token_speaker = normalize_speaker_label(token.speaker)
        if not self.current_tokens:
            self.start(token, token_speaker)
            return

        speaker_changed = (
            token_speaker
            and self.current_speaker
            and token_speaker != self.current_speaker
        )
        if speaker_changed:
            self.flush()
            self.start(token, token_speaker)
            return

        candidate = [*self.current_tokens, token]
        if cue_candidate_is_valid(candidate, self.prefix):
            self.current_tokens = candidate
            if should_soft_break(candidate, self.prefix):
                self.flush()
            return

        self.flush()
        self.start(token, token_speaker)
        if not cue_candidate_is_valid(self.current_tokens, self.prefix):
            self.flush()


def build_srt_cues_from_result(
    result: dict[str, Any], cfg: TranscriptConfig | None = None
) -> list[SRTCue]:
    include_speaker_labels = True if cfg is None else cfg.include_speaker_labels
    tokens = extract_timed_tokens(result, cfg)
    if not tokens:
        return build_segment_fallback_cues(
            result, include_speaker_labels=include_speaker_labels
        )

    accumulator = TimedCueAccumulator.create(include_speaker_labels)
    for token in tokens:
        accumulator.add(token)
    accumulator.flush()

    if accumulator.cues:
        return accumulator.cues
    return build_segment_fallback_cues(
        result, include_speaker_labels=include_speaker_labels
    )


def write_direct_srt_from_result(
    result: dict[str, Any], srt_path: Path, cfg: TranscriptConfig | None = None
) -> None:
    cues = build_srt_cues_from_result(result, cfg)
    if not cues:
        srt_path.write_text("", encoding="utf-8")
        return
    srt_path.write_text(render_srt_cues(cues), encoding="utf-8")
