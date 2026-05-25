from __future__ import annotations

import contextlib
import io
import queue
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from transcriber.__main__ import main, parse_args
from transcriber.live import (
    LiveAudioQueueStats,
    _live_audio_queue_maxsize,
    _put_live_audio,
    _state_handler,
    _write_bilingual_transcript,
    build_live_config,
    run_live_mode,
)
from transcriber.live_audio import LoopbackDevice
from transcriber.live_wlk import CaptionPair, CaptionState, LiveTranslationMode


class LiveCliTests(unittest.TestCase):
    def run_main(self, argv: list[str]) -> tuple[int, str]:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = main(argv)
        return rc, stdout.getvalue()

    def test_live_defaults_to_spanish_small_system_source(self) -> None:
        cfg = build_live_config(parse_args(["--live"]))

        self.assertEqual(cfg.language, "es")
        self.assertEqual(cfg.model, "small")
        self.assertEqual(cfg.engine, "whisperlivekit")
        self.assertEqual(cfg.source, "system")
        self.assertEqual(cfg.translation_mode, LiveTranslationMode.DIRECT)
        self.assertEqual(cfg.preset, "latency")
        self.assertEqual(cfg.chunk_ms, 250)
        self.assertTrue(cfg.show_window)
        self.assertTrue(cfg.translate_to_english)
        self.assertFalse(cfg.speaker_labels)
        self.assertFalse(cfg.diarize)
        self.assertIsNone(cfg.save_bilingual_transcript_path)
        self.assertEqual(cfg.backend, "auto")
        self.assertEqual(cfg.backend_policy, "localagreement")
        self.assertEqual(cfg.frame_threshold, 25)
        self.assertEqual(cfg.beams, 1)
        self.assertEqual(cfg.decoder, "auto")
        self.assertEqual(cfg.audio_min_len, 0.0)
        self.assertEqual(cfg.audio_max_len, 30.0)
        self.assertEqual(cfg.nllb_backend, "transformers")
        self.assertEqual(cfg.nllb_size, "600M")
        self.assertIsNone(cfg.static_prompt)
        self.assertFalse(cfg.audio_diagnostics)

    def test_live_cascade_bilingual_transcript_path_reaches_config(self) -> None:
        cfg = build_live_config(
            parse_args(
                [
                    "--live",
                    "--live-translation-mode",
                    "cascade",
                    "--live-save-bilingual-transcript",
                    "logs/live_bilingual_transcript.txt",
                ]
            )
        )

        self.assertEqual(cfg.translation_mode, LiveTranslationMode.CASCADE)
        self.assertEqual(cfg.save_bilingual_transcript_path, "logs/live_bilingual_transcript.txt")

    def test_live_quality_preset_defaults_to_cascade(self) -> None:
        cfg = build_live_config(parse_args(["--live", "--live-preset", "quality"]))

        self.assertEqual(cfg.preset, "quality")
        self.assertEqual(cfg.translation_mode, LiveTranslationMode.CASCADE)
        self.assertEqual(cfg.chunk_ms, 500)
        self.assertEqual(cfg.model, "medium")
        self.assertEqual(cfg.backend, "faster-whisper")
        self.assertEqual(cfg.backend_policy, "localagreement")
        self.assertEqual(cfg.frame_threshold, 35)
        self.assertEqual(cfg.beams, 3)
        self.assertEqual(cfg.decoder, "beam")
        self.assertEqual(cfg.audio_min_len, 0.0)
        self.assertEqual(cfg.audio_max_len, 30.0)
        self.assertEqual(cfg.nllb_backend, "ctranslate2")
        self.assertEqual(cfg.nllb_size, "600M")

    def test_live_quality_preset_uses_unbounded_audio_queue(self) -> None:
        cfg = build_live_config(parse_args(["--live", "--live-preset", "quality"]))

        self.assertEqual(_live_audio_queue_maxsize(cfg), 0)

    def test_live_latency_preset_uses_bounded_audio_queue(self) -> None:
        cfg = build_live_config(parse_args(["--live"]))

        self.assertEqual(_live_audio_queue_maxsize(cfg), 8)

    def test_live_quality_preset_keeps_explicit_overrides(self) -> None:
        cfg = build_live_config(
            parse_args(
                [
                    "--live",
                    "--live-preset",
                    "quality",
                    "--model",
                    "small",
                    "--live-backend",
                    "whisper",
                    "--live-frame-threshold",
                    "50",
                    "--live-beams",
                    "5",
                    "--live-decoder",
                    "greedy",
                    "--live-audio-max-len",
                    "45",
                    "--live-nllb-backend",
                    "transformers",
                    "--live-nllb-size",
                    "1.3B",
                    "--live-audio-diagnostics",
                ]
            )
        )

        self.assertEqual(cfg.model, "small")
        self.assertEqual(cfg.backend, "whisper")
        self.assertEqual(cfg.frame_threshold, 50)
        self.assertEqual(cfg.beams, 5)
        self.assertEqual(cfg.decoder, "greedy")
        self.assertEqual(cfg.audio_max_len, 45.0)
        self.assertEqual(cfg.nllb_backend, "transformers")
        self.assertEqual(cfg.nllb_size, "1.3B")
        self.assertTrue(cfg.audio_diagnostics)

    def test_live_quality_spanish_cascade_uses_asr_static_prompt(self) -> None:
        cfg = build_live_config(parse_args(["--live", "--live-preset", "quality", "--lang", "es"]))

        self.assertEqual(cfg.translation_mode, LiveTranslationMode.CASCADE)
        self.assertIsNotNone(cfg.static_prompt)
        self.assertIn("Transcribe", cfg.static_prompt or "")
        self.assertIn("Do not translate", cfg.static_prompt or "")
        self.assertIn("casarse", cfg.static_prompt or "")
        self.assertNotIn("Translate into natural English", cfg.static_prompt or "")

    def test_live_quality_spanish_direct_uses_translation_static_prompt(self) -> None:
        cfg = build_live_config(
            parse_args(["--live", "--live-preset", "quality", "--live-translation-mode", "direct", "--lang", "es"])
        )

        self.assertEqual(cfg.translation_mode, LiveTranslationMode.DIRECT)
        self.assertIsNotNone(cfg.static_prompt)
        self.assertIn("Translate into natural English", cfg.static_prompt or "")
        self.assertIn("casarse", cfg.static_prompt or "")

    def test_live_static_prompt_overrides_default_prompt(self) -> None:
        cfg = build_live_config(
            parse_args(["--live", "--live-preset", "quality", "--lang", "es", "--live-static-prompt", "Custom"])
        )

        self.assertEqual(cfg.static_prompt, "Custom")

    def test_live_explicit_translation_mode_overrides_quality_preset(self) -> None:
        cfg = build_live_config(parse_args(["--live", "--live-preset", "quality", "--live-translation-mode", "direct"]))

        self.assertEqual(cfg.translation_mode, LiveTranslationMode.DIRECT)

    def test_live_direct_mode_rejects_bilingual_transcript_path(self) -> None:
        stdout = io.StringIO()
        with (
            contextlib.redirect_stdout(stdout),
            patch("transcriber.live.run_live_session", return_value=0) as run_session,
        ):
            rc = run_live_mode(
                parse_args(["--live", "--live-save-bilingual-transcript", "logs/live_bilingual_transcript.txt"])
            )

        self.assertEqual(rc, 1)
        self.assertIn("requires --live-translation-mode cascade", stdout.getvalue())
        run_session.assert_not_called()

    def test_live_rejects_zero_audio_max_len(self) -> None:
        stdout = io.StringIO()
        with (
            contextlib.redirect_stdout(stdout),
            patch("transcriber.live.run_live_session", return_value=0) as run_session,
        ):
            rc = run_live_mode(parse_args(["--live", "--live-audio-max-len", "0"]))

        self.assertEqual(rc, 1)
        self.assertIn("--live-audio-max-len must be greater than 0", stdout.getvalue())
        run_session.assert_not_called()

    def test_live_rejects_audio_min_len_above_max_len(self) -> None:
        stdout = io.StringIO()
        with (
            contextlib.redirect_stdout(stdout),
            patch("transcriber.live.run_live_session", return_value=0) as run_session,
        ):
            rc = run_live_mode(parse_args(["--live", "--live-audio-min-len", "45", "--live-audio-max-len", "5"]))

        self.assertEqual(rc, 1)
        self.assertIn("--live-audio-min-len cannot exceed --live-audio-max-len", stdout.getvalue())
        run_session.assert_not_called()

    def test_live_direct_mode_allows_english_transcript_path(self) -> None:
        with patch("transcriber.live.run_live_session", return_value=0) as run_session:
            rc = run_live_mode(parse_args(["--live", "--live-save-transcript", "logs/live_english_transcript.txt"]))

        self.assertEqual(rc, 0)
        run_session.assert_called_once()

    def test_live_translate_launcher_uses_direct_english_log(self) -> None:
        launcher = Path(__file__).resolve().parents[1] / "live_translate.bat"
        text = launcher.read_text(encoding="utf-8")

        self.assertIn("--live-preset latency", text)
        self.assertIn("--live-translation-mode direct", text)
        self.assertIn("--live-save-transcript", text)
        self.assertIn("logs\\live_english_transcript.txt", text)
        self.assertNotIn("--live-save-bilingual-transcript", text)

    def test_live_translate_quality_launcher_uses_cascade_quality_profile(self) -> None:
        launcher = Path(__file__).resolve().parents[1] / "live_translate_quality.bat"
        text = launcher.read_text(encoding="utf-8")

        self.assertIn("--live-preset quality", text)
        self.assertIn("--live-translation-mode cascade", text)
        self.assertIn("--live-save-bilingual-transcript", text)
        self.assertIn("logs\\live_bilingual_transcript.txt", text)
        self.assertIn("--live-beams", text)
        self.assertIn("--live-frame-threshold", text)
        self.assertIn("--live-audio-diagnostics", text)
        self.assertIn("LIVE_MODEL", text)
        self.assertIn("LIVE_NLLB_SIZE", text)
        self.assertIn("LIVE_MODEL=medium", text)
        self.assertIn("LIVE_NLLB_SIZE=600M", text)
        self.assertIn("LIVE_CHUNK_MS=750", text)
        self.assertIn("LIVE_FRAME_THRESHOLD=45", text)
        self.assertIn("LIVE_BEAMS=5", text)
        self.assertIn("LIVE_AUDIO_MIN_LEN=1.0", text)
        self.assertIn("LIVE_AUDIO_MAX_LEN=45", text)
        self.assertIn('if exist "transcriber_glossary.txt"', text)
        self.assertIn("--glossary-file", text)

    def test_write_bilingual_transcript_formats_spanish_and_english_pairs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "live_bilingual.txt"
            state = CaptionState(
                committed_lines=("hello world", "thanks"),
                partial_line="",
                committed_pairs=(
                    CaptionPair(source_text="hola mundo", translated_text="hello world"),
                    CaptionPair(source_text="gracias", translated_text="thanks"),
                ),
            )

            _write_bilingual_transcript(output_path, state)

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "1.\nES: hola mundo\nEN: hello world\n\n2.\nES: gracias\nEN: thanks\n",
            )

    def test_state_handler_writes_english_and_bilingual_transcripts_independently(self) -> None:
        with TemporaryDirectory() as tmpdir:
            english_path = Path(tmpdir) / "english.txt"
            bilingual_path = Path(tmpdir) / "bilingual.txt"
            state = CaptionState(
                committed_lines=("hello",),
                partial_line="",
                committed_pairs=(CaptionPair(source_text="hola", translated_text="hello"),),
            )
            handle_state = _state_handler(
                state_queue=None,
                save_transcript_path=str(english_path),
                save_bilingual_transcript_path=str(bilingual_path),
            )

            handle_state(state)

            self.assertEqual(english_path.read_text(encoding="utf-8"), "hello\n")
            self.assertEqual(bilingual_path.read_text(encoding="utf-8"), "1.\nES: hola\nEN: hello\n")

    def test_live_audio_latency_queue_drops_oldest_chunk_when_full(self) -> None:
        audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=1)
        stats = LiveAudioQueueStats()
        audio_queue.put_nowait(b"old")

        _put_live_audio(
            audio_queue,
            b"new",
            drop_oldest=True,
            stats=stats,
            stop_event=threading.Event(),
        )

        self.assertEqual(stats.dropped_chunks, 1)
        self.assertEqual(audio_queue.get_nowait(), b"new")

    def test_live_audio_quality_queue_waits_for_space_without_counting_drop(self) -> None:
        audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=1)
        stats = LiveAudioQueueStats()
        stop_event = threading.Event()
        audio_queue.put_nowait(b"old")

        thread = threading.Thread(
            target=_put_live_audio,
            kwargs={
                "audio_queue": audio_queue,
                "chunk": b"new",
                "drop_oldest": False,
                "stats": stats,
                "stop_event": stop_event,
            },
        )
        thread.start()
        time.sleep(0.05)
        self.assertEqual(stats.dropped_chunks, 0)
        self.assertEqual(audio_queue.get_nowait(), b"old")
        thread.join(timeout=1.0)

        self.assertFalse(thread.is_alive())
        self.assertEqual(audio_queue.get_nowait(), b"new")

    def test_live_extra_includes_whisperlivekit_server_dependencies(self) -> None:
        pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
        live_dependency_spec = (
            pyproject.read_text(encoding="utf-8")
            .split("[project.optional-dependencies]", maxsplit=1)[1]
            .split("[project.scripts]", maxsplit=1)[0]
        )

        self.assertIn('"whisperlivekit[translation]', live_dependency_spec)
        self.assertIn('"python-multipart', live_dependency_spec)

    def test_live_reuses_asr_prompt_and_glossary_flags(self) -> None:
        with TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "prompt.txt"
            prompt_file.write_text("Escuela de Nada\n", encoding="utf-8")

            cfg = build_live_config(
                parse_args(
                    [
                        "--live",
                        "--asr-prompt",
                        "OpenAI Codex",
                        "--asr-prompt-file",
                        str(prompt_file),
                        "--glossary",
                        "WhisperLiveKit => WhisperLiveKit",
                    ]
                )
            )

        self.assertIsNotNone(cfg.asr_prompt)
        self.assertIn("OpenAI Codex", cfg.asr_prompt or "")
        self.assertIn("Escuela de Nada", cfg.asr_prompt or "")
        self.assertIn("WhisperLiveKit", cfg.asr_prompt or "")

    def test_live_cannot_combine_with_input(self) -> None:
        rc, output = self.run_main(["--live", "--input", "sample.wav"])

        self.assertEqual(rc, 2)
        self.assertIn("--live cannot be combined with --input", output)

    def test_live_cannot_combine_with_watch(self) -> None:
        rc, output = self.run_main(["--live", "--watch"])

        self.assertEqual(rc, 2)
        self.assertIn("--live cannot be combined with --watch", output)

    def test_live_rejects_speaker_labels(self) -> None:
        rc, output = self.run_main(["--live", "--speaker-labels"])

        self.assertEqual(rc, 2)
        self.assertIn("--live does not support speaker labels", output)

    def test_live_dispatch_does_not_resolve_input_or_import_live_dependencies(self) -> None:
        with (
            patch("transcriber.__main__.resolve_input_path") as resolve_input,
            patch("transcriber.live.run_live_mode", return_value=0) as run_live,
        ):
            rc, _output = self.run_main(["--live", "--no-speaker-labels"])

        self.assertEqual(rc, 0)
        resolve_input.assert_not_called()
        run_live.assert_called_once()

    def test_live_list_devices_dispatches_without_live_flag(self) -> None:
        with patch("transcriber.live.run_live_mode", return_value=0) as run_live:
            rc, _output = self.run_main(["--live-list-devices"])

        self.assertEqual(rc, 0)
        run_live.assert_called_once()

    def test_live_loopback_test_dispatches_without_live_flag(self) -> None:
        with patch("transcriber.live.run_live_mode", return_value=0) as run_live:
            rc, _output = self.run_main(["--live-loopback-test", "--seconds", "1", "--output", "test.wav"])

        self.assertEqual(rc, 0)
        run_live.assert_called_once()

    def test_live_runtime_guard_rejects_python_310_before_importing_live_dependencies(self) -> None:
        stdout = io.StringIO()
        with (
            contextlib.redirect_stdout(stdout),
            patch("transcriber.live.sys.version_info", (3, 10, 9)),
        ):
            rc = run_live_mode(parse_args(["--live"]))

        self.assertEqual(rc, 1)
        self.assertIn("Live mode requires Python 3.11 or newer", stdout.getvalue())

    def test_run_live_mode_lists_loopback_devices(self) -> None:
        stdout = io.StringIO()
        with (
            contextlib.redirect_stdout(stdout),
            patch(
                "transcriber.live_audio.list_loopback_devices",
                return_value=[LoopbackDevice(index=4, name="Speakers", sample_rate=48_000, channels=2)],
            ),
        ):
            rc = run_live_mode(parse_args(["--live-list-devices"]))

        self.assertEqual(rc, 0)
        self.assertIn("4: Speakers", stdout.getvalue())

    def test_run_live_mode_records_loopback_test(self) -> None:
        with patch("transcriber.live_audio.write_loopback_test_wav") as write_wav:
            rc = run_live_mode(parse_args(["--live-loopback-test", "--seconds", "2", "--output", "sample.wav"]))

        self.assertEqual(rc, 0)
        write_wav.assert_called_once()

    def test_run_live_mode_reports_missing_live_dependency_for_device_list(self) -> None:
        stdout = io.StringIO()
        with (
            contextlib.redirect_stdout(stdout),
            patch(
                "transcriber.live_audio.list_loopback_devices",
                side_effect=RuntimeError('Install live extras with: pip install -e ".[live]"'),
            ),
        ):
            rc = run_live_mode(parse_args(["--live-list-devices"]))

        output = stdout.getvalue()
        self.assertEqual(rc, 1)
        self.assertIn('Install live extras with: pip install -e ".[live]"', output)
        self.assertNotIn("Traceback", output)

    def test_run_live_mode_reports_missing_live_dependency_for_loopback_test(self) -> None:
        stdout = io.StringIO()
        with (
            contextlib.redirect_stdout(stdout),
            patch(
                "transcriber.live_audio.write_loopback_test_wav",
                side_effect=RuntimeError('Install live extras with: pip install -e ".[live]"'),
            ),
        ):
            rc = run_live_mode(parse_args(["--live-loopback-test", "--seconds", "1", "--output", "test.wav"]))

        output = stdout.getvalue()
        self.assertEqual(rc, 1)
        self.assertIn('Install live extras with: pip install -e ".[live]"', output)
        self.assertNotIn("Traceback", output)

    def test_run_live_mode_delegates_streaming_to_session_runner(self) -> None:
        with patch("transcriber.live.run_live_session", return_value=0) as run_session:
            rc = run_live_mode(parse_args(["--live", "--no-speaker-labels"]))

        self.assertEqual(rc, 0)
        run_session.assert_called_once()


if __name__ == "__main__":
    unittest.main()
