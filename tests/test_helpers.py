from __future__ import annotations

import contextlib
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from transcriber.__main__ import (
    build_audio_preprocess_command,
    build_asr_prompt,
    build_config,
    RunConfig,
    SRTCue,
    TimedToken,
    apply_confidence_cleanup,
    build_llm_file,
    build_srt_cues_from_result,
    build_translation_prompt,
    flush_gpu_memory,
    load_translation_glossary,
    is_watchable_media,
    translate_spanish_texts,
    output_paths_for_input,
    parse_args,
    parse_detected_language_from_log,
    parse_glossary_entries,
    parse_temperature_schedule,
    preprocess_audio_for_whisperx,
    project_dir,
    read_text_tail,
    render_low_confidence_markup,
    run_whisperx_direct_logged,
    should_fallback_without_diarization,
    smooth_timed_tokens,
    transcribe_file,
    translation_context_for_cue,
    write_direct_srt_from_result,
)


class FakeTensor:
    def to(self, device: object) -> "FakeTensor":
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
        dry_run=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class HelperTests(unittest.TestCase):
    def test_build_config_defaults_to_auto(self) -> None:
        cfg = build_config(parse_args([]), interactive=False)
        self.assertEqual(cfg.language, "auto")
        self.assertFalse(cfg.translate_to_english)

    def test_build_config_uses_less_aggressive_word_confidence_default(self) -> None:
        cfg = build_config(parse_args([]), interactive=False)
        self.assertEqual(cfg.low_confidence_word_prob, 0.10)

    def test_translate_flag_enables_direct_whisperx_output(self) -> None:
        cfg = build_config(parse_args(["--translate-to-english"]), interactive=False)
        self.assertTrue(cfg.translate_to_english)

    def test_speaker_labels_are_enabled_by_default(self) -> None:
        cfg = build_config(parse_args([]), interactive=False)
        self.assertTrue(cfg.include_speaker_labels)

    def test_no_speaker_labels_flag_disables_rendered_labels_only(self) -> None:
        cfg = build_config(parse_args(["--no-speaker-labels", "--diarize"]), interactive=False)
        self.assertFalse(cfg.include_speaker_labels)
        self.assertTrue(cfg.diarize)

    def test_spanish_language_defaults_to_translation(self) -> None:
        cfg = build_config(parse_args(["--lang", "es"]), interactive=False)
        self.assertTrue(cfg.translate_to_english)

    def test_temperature_schedule_parser(self) -> None:
        self.assertEqual(parse_temperature_schedule("0.0, 0.2,0.4"), (0.0, 0.2, 0.4))

    def test_audio_preprocess_command_targets_mono_wav(self) -> None:
        command = build_audio_preprocess_command(Path("in.mp4"), Path("out.wav"))

        self.assertEqual(command[0], "ffmpeg")
        self.assertIn("0:a:0?", command)
        self.assertIn("-vn", command)
        self.assertIn("-ac", command)
        self.assertIn("1", command)
        self.assertIn("-ar", command)
        self.assertIn("16000", command)
        self.assertIn("-af", command)
        self.assertIn("highpass=f=60,lowpass=f=8000", command)
        self.assertEqual(command[-1], "out.wav")

    @patch("transcriber.__main__.subprocess.run")
    def test_audio_preprocess_uses_timeout(self, run: MagicMock) -> None:
        run.side_effect = FileNotFoundError()
        with TemporaryDirectory() as tmpdir:
            reports: list[str] = []

            result = preprocess_audio_for_whisperx(Path("in.mp4"), Path(tmpdir), report=reports.append)

            self.assertEqual(result, Path("in.mp4"))
            self.assertIn("timeout", run.call_args.kwargs)
            self.assertGreater(run.call_args.kwargs["timeout"], 0)

    @patch("transcriber.__main__.subprocess.run")
    def test_audio_preprocess_truncates_long_stderr(self, run: MagicMock) -> None:
        run.side_effect = subprocess.CalledProcessError(
            1,
            ["ffmpeg"],
            stderr="x" * 1200,
        )
        with TemporaryDirectory() as tmpdir:
            reports: list[str] = []

            result = preprocess_audio_for_whisperx(Path("in.mp4"), Path(tmpdir), report=reports.append)

            self.assertEqual(result, Path("in.mp4"))
            self.assertTrue(any("[truncated]" in line for line in reports))
            self.assertLess(max(len(line) for line in reports), 700)

    def test_read_text_tail_returns_recent_content_only(self) -> None:
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "large.log"
            log_path.write_text("old-" + ("x" * 100) + "-tail", encoding="utf-8")

            text = read_text_tail(log_path, max_chars=10)

            self.assertEqual(text, "xxxxx-tail")

    def test_log_parsers_use_recent_tail_content(self) -> None:
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "large.log"
            log_path.write_text(
                ("older noise\n" * 1000)
                + "Could not download 'pyannote/speaker-diarization-3.1' pipeline.\n"
                + "Detected language: Spanish (0.98)\n",
                encoding="utf-8",
            )

            self.assertTrue(should_fallback_without_diarization(log_path))
            self.assertEqual(parse_detected_language_from_log(log_path), "spanish")

    def test_opus_files_are_watchable(self) -> None:
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "clip.opus"
            source.write_bytes(b"data")
            self.assertTrue(is_watchable_media(source))

    def test_output_paths_stay_next_to_source(self) -> None:
        cfg = make_cfg()
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "meeting.mp4"
            source.write_bytes(b"data")
            outputs = output_paths_for_input(source, cfg, create_dirs=False)

            self.assertEqual(outputs.output_dir, source.parent)
            self.assertEqual(outputs.srt_path.parent, source.parent)
            self.assertEqual(outputs.llm_path.parent, source.parent)
            self.assertEqual(outputs.lock_path.parent, source.parent)
            self.assertEqual(outputs.log_path.parent, project_dir() / "logs")

    @patch("transcriber.__main__.run_whisperx_direct_logged")
    def test_transcribe_file_rejects_unsupported_input_before_work(self, run_logged: MagicMock) -> None:
        cfg = make_cfg(diarize=False)
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "notes.txt"
            source.write_text("not media", encoding="utf-8")
            outputs = output_paths_for_input(source, cfg, create_dirs=False)
            reports: list[str] = []

            rc = transcribe_file(cfg, source, report=reports.append)

            self.assertEqual(rc, 1)
            self.assertFalse(outputs.srt_path.exists())
            self.assertFalse(outputs.llm_path.exists())
            self.assertFalse(outputs.lock_path.exists())
            self.assertTrue(any("Unsupported media file" in line for line in reports))
            run_logged.assert_not_called()

    @patch("transcriber.__main__.run_whisperx_direct_logged")
    def test_transcribe_file_rejects_directory_before_work(self, run_logged: MagicMock) -> None:
        cfg = make_cfg(diarize=False)
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "recordings.mp4"
            source.mkdir()
            outputs = output_paths_for_input(source, cfg, create_dirs=False)
            reports: list[str] = []

            rc = transcribe_file(cfg, source, report=reports.append)

            self.assertEqual(rc, 1)
            self.assertFalse(outputs.srt_path.exists())
            self.assertFalse(outputs.llm_path.exists())
            self.assertFalse(outputs.lock_path.exists())
            self.assertTrue(any("Input is not a file" in line for line in reports))
            run_logged.assert_not_called()

    @patch("transcriber.__main__.flush_gpu_memory")
    @patch("transcriber.__main__.run_whisperx_direct")
    def test_run_whisperx_direct_logged_flushes_gpu_after_success(
        self, run_direct: MagicMock, flush_gpu: MagicMock
    ) -> None:
        run_direct.return_value = "en"
        cfg = make_cfg(device="cuda")

        with TemporaryDirectory() as tmpdir:
            rc, detected_language = run_whisperx_direct_logged(
                cfg,
                Path("input.wav"),
                Path("output.srt"),
                hf_token=None,
                diarize=False,
                log_path=Path(tmpdir) / "whisperx.log",
            )

        self.assertEqual(rc, 0)
        self.assertEqual(detected_language, "en")
        flush_gpu.assert_called_once_with("cuda")

    @patch("transcriber.__main__.flush_gpu_memory")
    @patch("transcriber.__main__.run_whisperx_direct")
    def test_run_whisperx_direct_logged_flushes_gpu_after_failure(
        self, run_direct: MagicMock, flush_gpu: MagicMock
    ) -> None:
        run_direct.side_effect = RuntimeError("boom")
        cfg = make_cfg(device="cuda")

        with TemporaryDirectory() as tmpdir:
            rc, detected_language = run_whisperx_direct_logged(
                cfg,
                Path("input.wav"),
                Path("output.srt"),
                hf_token=None,
                diarize=False,
                log_path=Path(tmpdir) / "whisperx.log",
            )

        self.assertEqual(rc, 1)
        self.assertIsNone(detected_language)
        flush_gpu.assert_called_once_with("cuda")

    def test_glossary_parsing_and_prompt(self) -> None:
        glossary = parse_glossary_entries(["OpenAI => OpenAI", "esfuerzo|effort", "termino"])
        prompt = build_translation_prompt(model_name="model", context_window=1, glossary=glossary)

        self.assertEqual(glossary["OpenAI"], "OpenAI")
        self.assertEqual(glossary["esfuerzo"], "effort")
        self.assertEqual(glossary["termino"], "termino")
        self.assertIn("__CUR_START__", prompt)
        self.assertIn("Glossary:", prompt)

    def test_asr_prompt_includes_glossary_and_file_terms(self) -> None:
        with TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "asr.txt"
            prompt_file.write_text("Project Falcon\n# comment\nAcmeOS\n", encoding="utf-8")

            glossary = parse_glossary_entries(["OpenAI => OpenAI", "WhisperX => WhisperX"])
            prompt = build_asr_prompt(
                glossary=glossary,
                prompt_text="Use exact spellings.",
                prompt_file=str(prompt_file),
            )

            self.assertIsNotNone(prompt)
            self.assertIn("Project Falcon", prompt or "")
            self.assertIn("AcmeOS", prompt or "")
            self.assertIn("Use exact spellings.", prompt or "")
            self.assertIn("OpenAI", prompt or "")
            self.assertIn("WhisperX", prompt or "")

    def test_temperature_override_sets_single_value(self) -> None:
        cfg = build_config(parse_args(["--temperature", "0.3"]), interactive=False)
        self.assertEqual(cfg.temperature, 0.3)
        self.assertEqual(cfg.temperature_schedule, (0.3,))

    def test_glossary_file_loader(self) -> None:
        with TemporaryDirectory() as tmpdir:
            glossary_path = Path(tmpdir) / "glossary.txt"
            glossary_path.write_text("OpenAI => OpenAI\n", encoding="utf-8")

            glossary = load_translation_glossary(str(glossary_path))

            self.assertEqual(glossary["OpenAI"], "OpenAI")

    def test_flush_gpu_memory_skips_cpu_devices(self) -> None:
        cuda = SimpleNamespace(
            is_available=MagicMock(return_value=True),
            empty_cache=MagicMock(),
            ipc_collect=MagicMock(),
        )
        fake_torch = SimpleNamespace(cuda=cuda)

        with patch.dict(sys.modules, {"torch": fake_torch}):
            flush_gpu_memory("cpu")

        cuda.is_available.assert_not_called()
        cuda.empty_cache.assert_not_called()
        cuda.ipc_collect.assert_not_called()

    def test_flush_gpu_memory_releases_cuda_cache(self) -> None:
        cuda = SimpleNamespace(
            is_available=MagicMock(return_value=True),
            empty_cache=MagicMock(),
            ipc_collect=MagicMock(),
        )
        fake_torch = SimpleNamespace(cuda=cuda)

        with (
            patch("transcriber.__main__.gc.collect") as collect,
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            flush_gpu_memory("cuda")

        collect.assert_called_once()
        cuda.is_available.assert_called_once()
        cuda.empty_cache.assert_called_once()
        cuda.ipc_collect.assert_called_once()

    def test_flush_gpu_memory_suppresses_cleanup_errors(self) -> None:
        cuda = SimpleNamespace(
            is_available=MagicMock(return_value=True),
            empty_cache=MagicMock(side_effect=RuntimeError("boom")),
            ipc_collect=MagicMock(side_effect=RuntimeError("boom")),
        )
        fake_torch = SimpleNamespace(cuda=cuda)

        with (
            patch("transcriber.__main__.gc.collect"),
            patch.dict(sys.modules, {"torch": fake_torch}),
        ):
            flush_gpu_memory("cuda")

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
            [("model.to", "cuda-device"), ("model.to", "cpu-device"), ("flush", "cuda")],
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
            [("model.to", "cuda-device"), ("model.to", "cpu-device"), ("flush", "cuda")],
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
            [("model.to", "cuda-device"), ("model.to", "cpu-device"), ("flush", "cuda")],
        )

    def test_confidence_cleanup_marks_low_confidence(self) -> None:
        cfg = make_cfg()
        result = {
            "segments": [
                {
                    "text": "hola",
                    "avg_logprob": -2.0,
                    "no_speech_prob": 0.0,
                    "words": [
                        {"word": "hola", "start": 0.0, "end": 0.2, "probability": 0.1},
                    ],
                }
            ]
        }

        apply_confidence_cleanup(result, cfg)

        segment = result["segments"][0]
        self.assertTrue(segment.get("_low_confidence"))
        self.assertEqual(segment["text"], "hola")
        word = segment["words"][0]
        self.assertTrue(word.get("_low_confidence"))
        self.assertEqual(word["word"], "hola")

    def test_low_confidence_markup_renders_as_dash_for_srt_and_llm(self) -> None:
        marker = "__LOWCONF_65__hola__LOWCONF_END__"
        self.assertEqual(render_low_confidence_markup(marker, "srt"), "—")
        self.assertEqual(render_low_confidence_markup(marker, "llm"), "—")

    def test_legacy_low_confidence_markup_renders_as_dash(self) -> None:
        legacy_marker_name = "UNC" + "ERTAIN"
        marker = f"__{legacy_marker_name}_65__hola__{legacy_marker_name}_END__"
        self.assertEqual(render_low_confidence_markup(marker, "srt"), "—")

    def test_timed_srt_keeps_speaker_labels_by_default(self) -> None:
        result = {
            "segments": [
                {
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
                        {"word": "there.", "start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"},
                        {"word": "Come", "start": 1.0, "end": 1.5, "speaker": "SPEAKER_01"},
                        {"word": "in.", "start": 1.5, "end": 2.0, "speaker": "SPEAKER_01"},
                    ]
                }
            ]
        }

        cues = build_srt_cues_from_result(result, make_cfg(diarize_smoothing=False))

        self.assertEqual([cue.text for cue in cues], ["SPEAKER_00: Hello there.", "SPEAKER_01: Come in."])

    def test_timed_srt_hides_speaker_labels_without_losing_speaker_splits(self) -> None:
        result = {
            "segments": [
                {
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
                        {"word": "there.", "start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"},
                        {"word": "Come", "start": 1.0, "end": 1.5, "speaker": "SPEAKER_01"},
                        {"word": "in.", "start": 1.5, "end": 2.0, "speaker": "SPEAKER_01"},
                    ]
                }
            ]
        }

        cues = build_srt_cues_from_result(
            result,
            make_cfg(diarize_smoothing=False, include_speaker_labels=False),
        )

        self.assertEqual([cue.text for cue in cues], ["Hello there.", "Come in."])

    def test_segment_fallback_srt_hides_speaker_labels(self) -> None:
        result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Hello there.",
                    "speaker": "SPEAKER_00",
                }
            ]
        }

        cues = build_srt_cues_from_result(result, make_cfg(include_speaker_labels=False))

        self.assertEqual([cue.text for cue in cues], ["Hello there."])

    def test_llm_file_uses_no_label_srt_body(self) -> None:
        result = {
            "segments": [
                {
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
                        {"word": "there.", "start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"},
                    ]
                }
            ]
        }
        with TemporaryDirectory() as tmpdir:
            srt_path = Path(tmpdir) / "movie.srt"
            llm_path = Path(tmpdir) / "movie_llm.txt"

            write_direct_srt_from_result(result, srt_path, make_cfg(include_speaker_labels=False))
            build_llm_file(srt_path, llm_path)

            transcript_body = llm_path.read_text(encoding="utf-8").split("TRANSCRIPT:\n", 1)[1]
            self.assertEqual(transcript_body, "Hello there.")
            self.assertNotIn("SPEAKER_", transcript_body)

    def test_speaker_smoothing_merges_short_blips(self) -> None:
        tokens = [
            TimedToken(text="Hello", start_ms=0, end_ms=300, speaker="SPEAKER_00"),
            TimedToken(text="yes", start_ms=300, end_ms=500, speaker="SPEAKER_01", confidence=0.2),
            TimedToken(text="there", start_ms=500, end_ms=1100, speaker="SPEAKER_00"),
        ]

        smoothed = smooth_timed_tokens(tokens)

        self.assertEqual([token.speaker for token in smoothed], ["SPEAKER_00", "SPEAKER_00", "SPEAKER_00"])


if __name__ == "__main__":
    unittest.main()
