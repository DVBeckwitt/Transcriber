from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from transcriber.__main__ import main, parse_args
from transcriber.live import build_live_config, run_live_mode
from transcriber.live_audio import LoopbackDevice


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
        self.assertEqual(cfg.chunk_ms, 500)
        self.assertTrue(cfg.show_window)
        self.assertTrue(cfg.translate_to_english)
        self.assertFalse(cfg.speaker_labels)
        self.assertFalse(cfg.diarize)

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
