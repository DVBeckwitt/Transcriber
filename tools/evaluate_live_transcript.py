from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BilingualSegment:
    source_text: str
    translated_text: str


def parse_bilingual_transcript(text: str) -> list[BilingualSegment]:
    segments: list[BilingualSegment] = []
    source_text: str | None = None
    translated_text: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or (line.endswith(".") and line[:-1].isdigit()):
            _append_segment(segments, source_text, translated_text)
            source_text = None
            translated_text = None
        elif line.startswith("ES:"):
            source_text = line[3:].strip()
        elif line.startswith("EN:"):
            translated_text = line[3:].strip()

    _append_segment(segments, source_text, translated_text)
    return segments


def _append_segment(segments: list[BilingualSegment], source_text: str | None, translated_text: str | None) -> None:
    if source_text is not None or translated_text is not None:
        segments.append(BilingualSegment(source_text=source_text or "", translated_text=translated_text or ""))


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _levenshtein_distance(reference: Sequence[object], candidate: Sequence[object]) -> int:
    if not reference:
        return len(candidate)
    if not candidate:
        return len(reference)

    previous = list(range(len(candidate) + 1))
    for reference_index, reference_item in enumerate(reference, start=1):
        current = [reference_index]
        for candidate_index, candidate_item in enumerate(candidate, start=1):
            insertion = current[candidate_index - 1] + 1
            deletion = previous[candidate_index] + 1
            substitution = previous[candidate_index - 1] + (reference_item != candidate_item)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def _error_rate(reference: Sequence[object], candidate: Sequence[object]) -> float:
    if not reference:
        return 0.0 if not candidate else 1.0
    return _levenshtein_distance(reference, candidate) / len(reference)


def character_error_rate(reference: str, candidate: str) -> float:
    normalized_reference = _normalize_text(reference)
    normalized_candidate = _normalize_text(candidate)
    return _error_rate(normalized_reference, normalized_candidate)


def word_error_rate(reference: str, candidate: str) -> float:
    normalized_reference = _normalize_text(reference).split()
    normalized_candidate = _normalize_text(candidate).split()
    return _error_rate(normalized_reference, normalized_candidate)


def evaluate_transcript(
    *, reference_es: str, candidate_bilingual: str, reference_en: str | None = None
) -> dict[str, float]:
    segments = parse_bilingual_transcript(candidate_bilingual)
    candidate_es = " ".join(segment.source_text for segment in segments if segment.source_text)
    candidate_en = " ".join(segment.translated_text for segment in segments if segment.translated_text)

    metrics = {
        "ES CER": character_error_rate(reference_es, candidate_es),
        "ES WER": word_error_rate(reference_es, candidate_es),
    }
    if reference_en is not None:
        metrics["EN CER"] = character_error_rate(reference_en, candidate_en)
        metrics["EN WER"] = word_error_rate(reference_en, candidate_en)
    return metrics


def evaluate_files(
    *, reference_es_path: Path, candidate_bilingual_path: Path, reference_en_path: Path | None = None
) -> dict[str, float]:
    reference_es = reference_es_path.read_text(encoding="utf-8")
    candidate_bilingual = candidate_bilingual_path.read_text(encoding="utf-8")
    reference_en = reference_en_path.read_text(encoding="utf-8") if reference_en_path else None
    return evaluate_transcript(
        reference_es=reference_es,
        candidate_bilingual=candidate_bilingual,
        reference_en=reference_en,
    )


def format_metrics(metrics: dict[str, float]) -> str:
    return "".join(f"{name}: {value:.3f}\n" for name, value in metrics.items())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a live bilingual transcript against reference text.")
    parser.add_argument("--reference-es", required=True, type=Path, help="Spanish reference transcript path.")
    parser.add_argument("--candidate-bilingual", required=True, type=Path, help="Candidate bilingual transcript path.")
    parser.add_argument("--reference-en", type=Path, help="Optional English reference transcript path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    metrics = evaluate_files(
        reference_es_path=args.reference_es,
        candidate_bilingual_path=args.candidate_bilingual,
        reference_en_path=args.reference_en,
    )
    print(format_metrics(metrics), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
