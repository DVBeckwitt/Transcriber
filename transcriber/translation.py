"""Spanish-to-English subtitle translation support."""

from __future__ import annotations

import contextlib
import functools
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    SPANISH_TRANSLATION_MODEL,
    TRANSLATION_BATCH_SIZE,
    TRANSLATION_CONTEXT_WINDOW,
    TRANSLATION_LENGTH_PENALTY,
    TRANSLATION_MARKER_END,
    TRANSLATION_MARKER_START,
    TRANSLATION_MAX_NEW_TOKENS,
    TRANSLATION_NO_REPEAT_NGRAM_SIZE,
    TRANSLATION_NUM_BEAMS,
    parse_glossary_entries,
)
from .subtitles import (
    SRTCue,
    normalize_subtitle_whitespace,
    parse_srt_cues,
    render_srt_cues,
    split_cue_for_subtitles,
    split_speaker_prefix,
)
from .utils import project_dir

TranslatorLoader = Callable[[], tuple[Any, Any]]
GPUCleanup = Callable[[str], None]
TextTranslator = Callable[..., list[str]]


@dataclass
class TranslationCue:
    cue: SRTCue
    prefix: str
    original_text: str
    protected_text: str
    placeholders: dict[str, str]


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


def apply_glossary_placeholders(
    text: str, glossary: dict[str, str]
) -> tuple[str, dict[str, str]]:
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


def translation_context_for_cue(
    cues: Sequence[SRTCue], index: int, window: int
) -> list[tuple[int, str]]:
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


def build_translation_prompt(
    *, model_name: str, context_window: int, glossary: dict[str, str]
) -> str:
    lines = [
        "Translate Spanish subtitle text to natural, accurate English.",
        f"Model: {model_name}",
        f"Context window: {context_window}",
        f"Preserve markers: {TRANSLATION_MARKER_START} ... {TRANSLATION_MARKER_END}",
        "Preserve speaker labels, names, numbers, and low-confidence markers exactly.",
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


@functools.lru_cache(maxsize=1)
def load_spanish_to_english_translator() -> tuple[Any, Any]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(SPANISH_TRANSLATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(SPANISH_TRANSLATION_MODEL)
    return tokenizer, model


def chunked_text(items: Sequence[str], size: int) -> Iterable[list[str]]:
    for idx in range(0, len(items), size):
        yield list(items[idx : idx + size])


def translate_spanish_texts(
    texts: Sequence[str],
    device: str,
    batch_size: int = 4,
    num_beams: int = TRANSLATION_NUM_BEAMS,
    max_new_tokens: int = 256,
    no_repeat_ngram_size: int = TRANSLATION_NO_REPEAT_NGRAM_SIZE,
    *,
    load_translator: TranslatorLoader,
    cleanup_gpu: GPUCleanup,
) -> list[str]:
    if not texts:
        return []

    tokenizer, model = load_translator()

    try:
        import torch
    except Exception:
        torch = None

    use_cuda = False
    target_device: Any = None
    translated: list[str] = []
    inputs: dict[str, Any] | None = None
    outputs: Any = None
    try:
        use_cuda = bool(
            torch is not None
            and device.startswith("cuda")
            and getattr(torch.cuda, "is_available", lambda: False)()
        )
        if torch is not None:
            target_device = torch.device("cuda" if use_cuda else "cpu")
            model.to(target_device)
            model.eval()

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
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    length_penalty=TRANSLATION_LENGTH_PENALTY,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    early_stopping=True,
                )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            translated.extend(text.strip() for text in decoded)
        return translated
    finally:
        inputs = None
        outputs = None
        if use_cuda and torch is not None:
            with contextlib.suppress(Exception):
                model.to(torch.device("cpu"))
        model = None
        tokenizer = None
        cleanup_gpu(device)


def _translation_glossary(
    glossary: dict[str, str] | None, glossary_spec: str | None
) -> dict[str, str]:
    merged = load_translation_glossary(glossary_spec)
    merged.update(glossary or {})
    return merged


def _build_translation_cues(
    cues: Sequence[SRTCue],
    context_window: int,
    glossary: dict[str, str],
) -> list[TranslationCue]:
    requests: list[TranslationCue] = []
    for index, cue in enumerate(cues):
        prefix, current_text = split_speaker_prefix(cue.text)
        current_text = normalize_subtitle_whitespace(current_text)
        context = translation_context_for_cue(cues, index, context_window)
        block_lines = [f"[context {offset:+d}] {text}" for offset, text in context]
        block_lines.append(
            f"{TRANSLATION_MARKER_START} {current_text} {TRANSLATION_MARKER_END}"
        )
        protected_text, placeholders = apply_glossary_placeholders(
            "\n".join(block_lines), glossary
        )
        requests.append(
            TranslationCue(
                cue=cue,
                prefix=prefix,
                original_text=current_text,
                protected_text=protected_text,
                placeholders=placeholders,
            )
        )
    return requests


def _translate_with_settings(
    texts: Sequence[str],
    *,
    translate_texts: TextTranslator,
    device: str,
    batch_size: int,
    num_beams: int,
    max_new_tokens: int,
    no_repeat_ngram_size: int,
) -> list[str]:
    return translate_texts(
        texts,
        device=device,
        batch_size=batch_size,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )


def _resolve_translation_texts(
    requests: Sequence[TranslationCue],
    translated_blocks: Sequence[str],
    *,
    translate_texts: TextTranslator,
    device: str,
    batch_size: int,
    num_beams: int,
    max_new_tokens: int,
    no_repeat_ngram_size: int,
) -> list[str]:
    if len(translated_blocks) != len(requests):
        raise RuntimeError("Translation produced an unexpected cue count.")

    resolved: list[str | None] = []
    fallback_indexes: list[int] = []
    for index, (request, translated_block) in enumerate(
        zip(requests, translated_blocks, strict=False)
    ):
        restored = replace_glossary_placeholders(translated_block, request.placeholders)
        current_text = extract_between_markers(
            restored, TRANSLATION_MARKER_START, TRANSLATION_MARKER_END
        )
        resolved.append(current_text)
        if not current_text:
            fallback_indexes.append(index)

    if fallback_indexes:
        fallback_texts = _translate_with_settings(
            [requests[index].original_text for index in fallback_indexes],
            translate_texts=translate_texts,
            device=device,
            batch_size=batch_size,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        if len(fallback_texts) != len(fallback_indexes):
            raise RuntimeError("Fallback translation produced an unexpected cue count.")
        for index, fallback_text in zip(fallback_indexes, fallback_texts, strict=False):
            resolved[index] = replace_glossary_placeholders(
                fallback_text, requests[index].placeholders
            )

    return [
        translated or request.original_text
        for request, translated in zip(requests, resolved, strict=False)
    ]


def _render_translated_cues(
    requests: Sequence[TranslationCue], translated_texts: Sequence[str]
) -> list[SRTCue]:
    translated_cues = []
    for request, translated_text in zip(requests, translated_texts, strict=False):
        text = normalize_subtitle_whitespace(translated_text)
        if request.prefix:
            text = f"{request.prefix}{text}".strip()
        translated_cues.append(
            SRTCue(
                index=request.cue.index,
                start_ms=request.cue.start_ms,
                end_ms=request.cue.end_ms,
                text=text,
            )
        )

    rewrapped = [
        split_cue
        for cue in translated_cues
        for split_cue in split_cue_for_subtitles(cue)
    ]
    return rewrapped or translated_cues


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
    *,
    translate_texts: TextTranslator,
) -> None:
    if not srt_path.exists():
        return

    cues = parse_srt_cues(srt_path.read_text(encoding="utf-8", errors="ignore"))
    if not cues:
        return

    merged_glossary = _translation_glossary(glossary, glossary_spec)
    if log_path is not None:
        log_translation_prompt(
            log_path,
            build_translation_prompt(
                model_name=SPANISH_TRANSLATION_MODEL,
                context_window=context_window,
                glossary=merged_glossary,
            ),
        )

    requests = _build_translation_cues(cues, context_window, merged_glossary)
    translated_blocks = _translate_with_settings(
        [request.protected_text for request in requests],
        translate_texts=translate_texts,
        device=device,
        batch_size=batch_size,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    translated_texts = _resolve_translation_texts(
        requests,
        translated_blocks,
        translate_texts=translate_texts,
        device=device,
        batch_size=batch_size,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    srt_path.write_text(
        render_srt_cues(_render_translated_cues(requests, translated_texts)),
        encoding="utf-8",
    )
