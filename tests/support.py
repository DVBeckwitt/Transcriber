"""Shared test fixtures for transcription runtime tests."""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock

from transcriber.__main__ import RunConfig


class FakeTensor:
    def to(self, device: object) -> FakeTensor:
        return self


def make_cuda_translation_mocks(
    load_translator: MagicMock,
    flush_gpu: MagicMock,
) -> tuple[list[tuple[str, object]], MagicMock, MagicMock, SimpleNamespace]:
    events: list[tuple[str, object]] = []
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": FakeTensor(),
        "attention_mask": FakeTensor(),
    }

    model = MagicMock()
    model.to.side_effect = lambda device: events.append(("model.to", device))
    load_translator.return_value = (tokenizer, model)
    flush_gpu.side_effect = lambda device: events.append(("flush", device))

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=MagicMock(return_value=True)),
        device=MagicMock(side_effect=lambda name: f"{name}-device"),
        no_grad=MagicMock(return_value=contextlib.nullcontext()),
    )
    return events, tokenizer, model, fake_torch


def make_cfg(**overrides: object) -> RunConfig:
    cfg = RunConfig(
        language="auto",
        translate_to_english=False,
        mode="quality",
        model="large-v3",
        batch_size=8,
        beam_size=8,
        patience=1.2,
        temperature=0.0,
        temperature_schedule=(0.0, 0.2, 0.4, 0.6, 0.8),
        best_of=5,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        condition_on_previous_text=True,
        diarize=True,
        diarize_smoothing=True,
        min_speaker_turn_ms=900,
        min_speaker_turn_tokens=2,
        include_speaker_labels=True,
        confidence_cleanup=True,
        confidence_cleanup_mode="mark",
        low_confidence_logprob=-1.0,
        high_no_speech_prob=0.6,
        low_confidence_word_prob=0.10,
        device="cpu",
        compute_type="float32",
        translation_context_window=2,
        translation_batch_size=4,
        translation_num_beams=4,
        translation_max_new_tokens=256,
        translation_no_repeat_ngram_size=3,
        glossary={},
        glossary_path=None,
        asr_prompt=None,
        warm_vram=False,
        dry_run=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg
