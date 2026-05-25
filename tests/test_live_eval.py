from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def _load_evaluator_module() -> Any:
    module_path = Path(__file__).resolve().parents[1] / "tools" / "evaluate_live_transcript.py"
    spec = importlib.util.spec_from_file_location("evaluate_live_transcript", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluator module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evaluator = _load_evaluator_module()
BilingualSegment = evaluator.BilingualSegment
character_error_rate = evaluator.character_error_rate
evaluate_transcript = evaluator.evaluate_transcript
main = evaluator.main
parse_bilingual_transcript = evaluator.parse_bilingual_transcript
word_error_rate = evaluator.word_error_rate


class LiveTranscriptEvaluatorTests(unittest.TestCase):
    def test_parse_bilingual_transcript_reads_spanish_and_english_pairs(self) -> None:
        segments = parse_bilingual_transcript("1.\nES: hola mundo\nEN: hello world\n\n2.\nES: gracias\nEN: thanks\n")

        self.assertEqual(
            segments,
            [
                BilingualSegment(source_text="hola mundo", translated_text="hello world"),
                BilingualSegment(source_text="gracias", translated_text="thanks"),
            ],
        )

    def test_error_rates_are_zero_for_exact_normalized_match(self) -> None:
        self.assertEqual(character_error_rate("hola\nmundo", "hola mundo"), 0.0)
        self.assertEqual(word_error_rate("hola\nmundo", "hola mundo"), 0.0)

    def test_evaluate_transcript_reports_spanish_and_optional_english_metrics(self) -> None:
        metrics = evaluate_transcript(
            reference_es="hola mundo",
            candidate_bilingual="1.\nES: hola muno\nEN: hello word\n",
            reference_en="hello world",
        )

        self.assertAlmostEqual(metrics["ES CER"], 0.1)
        self.assertAlmostEqual(metrics["ES WER"], 0.5)
        self.assertAlmostEqual(metrics["EN CER"], 1 / 11)
        self.assertAlmostEqual(metrics["EN WER"], 0.5)

    def test_main_prints_compact_metrics(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            reference_es = root / "reference_es.txt"
            reference_en = root / "reference_en.txt"
            candidate = root / "candidate.txt"
            reference_es.write_text("hola mundo", encoding="utf-8")
            reference_en.write_text("hello world", encoding="utf-8")
            candidate.write_text("1.\nES: hola muno\nEN: hello word\n", encoding="utf-8")

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = main(
                    [
                        "--reference-es",
                        str(reference_es),
                        "--candidate-bilingual",
                        str(candidate),
                        "--reference-en",
                        str(reference_en),
                    ]
                )

        self.assertEqual(rc, 0)
        self.assertEqual(
            stdout.getvalue(),
            "ES CER: 0.100\nES WER: 0.500\nEN CER: 0.091\nEN WER: 0.500\n",
        )


if __name__ == "__main__":
    unittest.main()
