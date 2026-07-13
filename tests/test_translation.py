from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import transcriber.__main__ as transcriber_main
from tests.support import make_cuda_translation_mocks
from transcriber.__main__ import (
    SRTCue,
    translate_spanish_texts,
    translation_context_for_cue,
)


class TranslationTests(unittest.TestCase):
    def test_translation_context_includes_neighbors(self) -> None:
        cues = [
            SRTCue(index=1, start_ms=0, end_ms=1000, text="SPEAKER_1: Hola"),
            SRTCue(index=2, start_ms=1000, end_ms=2000, text="SPEAKER_2: Que tal"),
            SRTCue(index=3, start_ms=2000, end_ms=3000, text="SPEAKER_1: Bien"),
        ]
        context = translation_context_for_cue(cues, index=1, window=1)
        self.assertEqual(context, [(-1, "Hola"), (1, "Bien")])

    @patch("transcriber.__main__.load_spanish_to_english_translator")
    def test_translation_uses_beam_search(self, load_translator: MagicMock) -> None:
        import torch

        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        tokenizer.batch_decode.return_value = ["Hello there"]

        model = MagicMock()
        model.generate.return_value = torch.tensor([[1, 2, 3]])
        load_translator.return_value = (tokenizer, model)

        result = translate_spanish_texts(["Hola"], device="cpu", batch_size=1)

        self.assertEqual(result, ["Hello there"])
        model.to.assert_called_once()
        self.assertEqual(str(model.to.call_args.args[0]), "cpu")
        model.generate.assert_called_once()
        kwargs = model.generate.call_args.kwargs
        self.assertEqual(kwargs["num_beams"], 4)
        self.assertEqual(kwargs["length_penalty"], 1.0)
        self.assertEqual(kwargs["no_repeat_ngram_size"], 3)
        self.assertTrue(kwargs["early_stopping"])

    @patch("transcriber.__main__.translate_spanish_texts")
    def test_translation_marker_fallbacks_are_batched(
        self, translate_texts: MagicMock
    ) -> None:
        calls: list[list[str]] = []

        def fake_translate(
            texts: list[str], *args: object, **kwargs: object
        ) -> list[str]:
            calls.append(list(texts))
            if len(calls) == 1:
                return ["missing markers", "also missing markers"]
            return ["Hello.", "Goodbye."]

        translate_texts.side_effect = fake_translate

        with TemporaryDirectory() as tmpdir:
            srt_path = Path(tmpdir) / "spanish.srt"
            srt_path.write_text(
                (
                    "1\n"
                    "00:00:00,000 --> 00:00:01,000\n"
                    "Hola.\n\n"
                    "2\n"
                    "00:00:01,000 --> 00:00:02,000\n"
                    "Adios.\n"
                ),
                encoding="utf-8",
            )

            transcriber_main.translate_srt_to_english(srt_path, "cpu", context_window=0)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1], ["Hola.", "Adios."])

    @patch("transcriber.__main__.flush_gpu_memory")
    @patch("transcriber.__main__.load_spanish_to_english_translator")
    def test_translation_flushes_gpu_after_cuda_success(
        self, load_translator: MagicMock, flush_gpu: MagicMock
    ) -> None:
        events, tokenizer, model, fake_torch = make_cuda_translation_mocks(
            load_translator, flush_gpu
        )
        tokenizer.batch_decode.return_value = ["Hello there"]
        model.generate.return_value = object()

        with patch.dict(sys.modules, {"torch": fake_torch}):
            result = translate_spanish_texts(["Hola"], device="cuda", batch_size=1)

        self.assertEqual(result, ["Hello there"])
        self.assertEqual(
            events,
            [
                ("model.to", "cuda-device"),
                ("model.to", "cpu-device"),
                ("flush", "cuda"),
            ],
        )

    @patch("transcriber.__main__.flush_gpu_memory")
    @patch("transcriber.__main__.load_spanish_to_english_translator")
    def test_translation_flushes_gpu_after_cuda_failure(
        self, load_translator: MagicMock, flush_gpu: MagicMock
    ) -> None:
        events, _, model, fake_torch = make_cuda_translation_mocks(
            load_translator, flush_gpu
        )
        model.generate.side_effect = RuntimeError("boom")

        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaises(RuntimeError):
                translate_spanish_texts(["Hola"], device="cuda", batch_size=1)

        self.assertEqual(
            events,
            [
                ("model.to", "cuda-device"),
                ("model.to", "cpu-device"),
                ("flush", "cuda"),
            ],
        )

    @patch("transcriber.__main__.flush_gpu_memory")
    @patch("transcriber.__main__.load_spanish_to_english_translator")
    def test_translation_flushes_gpu_when_cuda_model_move_fails(
        self, load_translator: MagicMock, flush_gpu: MagicMock
    ) -> None:
        events, _, model, fake_torch = make_cuda_translation_mocks(
            load_translator, flush_gpu
        )

        def move_model(device: object) -> None:
            events.append(("model.to", device))
            if device == "cuda-device":
                raise RuntimeError("cuda oom")

        model.to.side_effect = move_model

        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaises(RuntimeError):
                translate_spanish_texts(["Hola"], device="cuda", batch_size=1)

        self.assertEqual(
            events,
            [
                ("model.to", "cuda-device"),
                ("model.to", "cpu-device"),
                ("flush", "cuda"),
            ],
        )
