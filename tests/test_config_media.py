from __future__ import annotations

import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import transcriber.__main__ as transcriber_main
from tests.support import make_cfg
from transcriber.__main__ import (
    build_asr_prompt,
    build_audio_preprocess_command,
    build_config,
    build_translation_prompt,
    is_watchable_media,
    load_translation_glossary,
    output_paths_for_input,
    parse_args,
    parse_detected_language_from_log,
    parse_glossary_entries,
    parse_temperature_schedule,
    preprocess_audio_for_whisperx,
    project_dir,
    read_text_tail,
    run_whisperx_direct_logged,
    should_fallback_without_diarization,
    transcribe_file,
)


class ConfigAndMediaTests(unittest.TestCase):
    def test_build_config_defaults_to_auto(self) -> None:
        cfg = build_config(parse_args([]), interactive=False)
        self.assertEqual(cfg.language, "auto")
        self.assertFalse(cfg.translate_to_english)

    def test_build_config_uses_less_aggressive_word_confidence_default(self) -> None:
        cfg = build_config(parse_args([]), interactive=False)
        self.assertEqual(cfg.low_confidence_word_prob, 0.10)

    def test_warm_vram_defaults_off_and_can_be_enabled(self) -> None:
        self.assertFalse(build_config(parse_args([]), interactive=False).warm_vram)
        self.assertTrue(
            build_config(parse_args(["--warm-vram"]), interactive=False).warm_vram
        )
        self.assertFalse(
            build_config(
                parse_args(["--warm-vram", "--no-warm-vram"]), interactive=False
            ).warm_vram
        )

    def test_translate_flag_enables_direct_whisperx_output(self) -> None:
        cfg = build_config(parse_args(["--translate-to-english"]), interactive=False)
        self.assertTrue(cfg.translate_to_english)

    def test_speaker_labels_are_enabled_by_default(self) -> None:
        cfg = build_config(parse_args([]), interactive=False)
        self.assertTrue(cfg.include_speaker_labels)

    def test_no_speaker_labels_flag_disables_rendered_labels_only(self) -> None:
        cfg = build_config(
            parse_args(["--no-speaker-labels", "--diarize"]), interactive=False
        )
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

    def test_audio_preprocess_command_accepts_resolved_ffmpeg_path(self) -> None:
        command = build_audio_preprocess_command(
            Path("in.mp4"), Path("out.wav"), r"C:\ffmpeg\bin\ffmpeg.exe"
        )

        self.assertEqual(command[0], r"C:\ffmpeg\bin\ffmpeg.exe")

    def test_resolve_ffmpeg_executable_uses_existing_candidate(self) -> None:
        with TemporaryDirectory() as tmpdir:
            ffmpeg_path = Path(tmpdir) / "ffmpeg.exe"
            ffmpeg_path.write_text("", encoding="utf-8")

            with patch(
                "transcriber.__main__.ffmpeg_candidate_paths",
                return_value=[Path(tmpdir) / "missing.exe", ffmpeg_path],
            ):
                self.assertEqual(
                    transcriber_main.resolve_ffmpeg_executable(), str(ffmpeg_path)
                )

    @patch("transcriber.__main__.shutil.which", return_value=None)
    def test_ffmpeg_candidate_paths_strips_quoted_env_path(
        self, which: MagicMock
    ) -> None:
        with patch.dict(
            transcriber_main.os.environ,
            {transcriber_main.FFMPEG_PATH_ENV_VAR: r'"C:\ffmpeg\bin\ffmpeg.exe"'},
        ):
            self.assertEqual(
                transcriber_main.ffmpeg_candidate_paths()[0],
                Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
            )

    @patch("transcriber.__main__.resolve_ffmpeg_executable")
    def test_ensure_ffmpeg_available_for_child_processes_prepends_resolved_parent(
        self, resolve_ffmpeg: MagicMock
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            ffmpeg_path = Path(tmpdir) / "bin" / "ffmpeg.exe"
            ffmpeg_path.parent.mkdir()
            ffmpeg_path.write_text("", encoding="utf-8")
            original_path = str(Path(tmpdir) / "other")
            resolve_ffmpeg.return_value = str(ffmpeg_path)

            with patch.dict(transcriber_main.os.environ, {"PATH": original_path}):
                result = transcriber_main.ensure_ffmpeg_available_for_child_processes()
                path_parts = transcriber_main.os.environ["PATH"].split(
                    transcriber_main.os.pathsep
                )

            self.assertEqual(result, str(ffmpeg_path))
            self.assertEqual(path_parts[0], str(ffmpeg_path.parent.resolve()))
            self.assertEqual(path_parts[1], original_path)

    @patch("transcriber.__main__.resolve_ffmpeg_executable")
    def test_ensure_ffmpeg_available_for_child_processes_is_idempotent(
        self, resolve_ffmpeg: MagicMock
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            ffmpeg_path = Path(tmpdir) / "bin" / "ffmpeg.exe"
            ffmpeg_path.parent.mkdir()
            ffmpeg_path.write_text("", encoding="utf-8")
            ffmpeg_dir = str(ffmpeg_path.parent.resolve())
            resolve_ffmpeg.return_value = str(ffmpeg_path)

            with patch.dict(
                transcriber_main.os.environ,
                {
                    "PATH": ffmpeg_dir
                    + transcriber_main.os.pathsep
                    + str(Path(tmpdir) / "other")
                },
            ):
                transcriber_main.ensure_ffmpeg_available_for_child_processes()
                transcriber_main.ensure_ffmpeg_available_for_child_processes()
                path_parts = transcriber_main.os.environ["PATH"].split(
                    transcriber_main.os.pathsep
                )

            normalized = [
                transcriber_main.os.path.normcase(
                    transcriber_main.os.path.normpath(part)
                )
                for part in path_parts
            ]
            normalized_ffmpeg_dir = transcriber_main.os.path.normcase(
                transcriber_main.os.path.normpath(ffmpeg_dir)
            )
            self.assertEqual(normalized.count(normalized_ffmpeg_dir), 1)

    @patch("transcriber.__main__.resolve_ffmpeg_executable", return_value=None)
    def test_ensure_ffmpeg_available_for_child_processes_requires_ffmpeg(
        self, resolve_ffmpeg: MagicMock
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "ffmpeg.*TRANSCRIBE_FFMPEG"):
            transcriber_main.ensure_ffmpeg_available_for_child_processes()

    @patch("transcriber.__main__.resolve_ffmpeg_executable")
    def test_ensure_ffmpeg_available_for_child_processes_requires_ffmpeg_basename(
        self, resolve_ffmpeg: MagicMock
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            renamed_path = Path(tmpdir) / "custom-decoder.exe"
            renamed_path.write_text("", encoding="utf-8")
            resolve_ffmpeg.return_value = str(renamed_path)

            with self.assertRaisesRegex(RuntimeError, "named ffmpeg"):
                transcriber_main.ensure_ffmpeg_available_for_child_processes()

    @patch("transcriber.__main__.subprocess.run")
    @patch("transcriber.__main__.resolve_ffmpeg_executable", return_value=None)
    def test_audio_preprocess_skips_subprocess_when_ffmpeg_unresolved(
        self,
        resolve_ffmpeg: MagicMock,
        run: MagicMock,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            reports: list[str] = []

            result = preprocess_audio_for_whisperx(
                Path("in.mp4"), Path(tmpdir), report=reports.append
            )

        self.assertEqual(result, Path("in.mp4"))
        run.assert_not_called()
        self.assertTrue(any("ffmpeg not found" in line for line in reports))

    @patch("transcriber.__main__.resolve_ffmpeg_executable", return_value="ffmpeg")
    @patch("transcriber.__main__.subprocess.run")
    def test_audio_preprocess_uses_timeout(
        self, run: MagicMock, resolve_ffmpeg: MagicMock
    ) -> None:
        run.side_effect = FileNotFoundError()
        with TemporaryDirectory() as tmpdir:
            reports: list[str] = []

            result = preprocess_audio_for_whisperx(
                Path("in.mp4"), Path(tmpdir), report=reports.append
            )

            self.assertEqual(result, Path("in.mp4"))
        self.assertIn("timeout", run.call_args.kwargs)
        self.assertGreater(run.call_args.kwargs["timeout"], 0)

    @patch("transcriber.__main__.resolve_ffmpeg_executable", return_value="ffmpeg")
    @patch("transcriber.__main__.subprocess.run")
    def test_audio_preprocess_truncates_long_stderr(
        self, run: MagicMock, resolve_ffmpeg: MagicMock
    ) -> None:
        run.side_effect = subprocess.CalledProcessError(
            1,
            ["ffmpeg"],
            stderr="x" * 1200,
        )
        with TemporaryDirectory() as tmpdir:
            reports: list[str] = []

            result = preprocess_audio_for_whisperx(
                Path("in.mp4"), Path(tmpdir), report=reports.append
            )

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

    def test_diarization_fallback_matches_exception_type(self) -> None:
        self.assertTrue(
            transcriber_main.diarization_error_allows_fallback(
                AttributeError("'NoneType' object has no attribute 'to'")
            )
        )

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

    def test_output_paths_fall_back_when_project_log_dir_is_unusable(self) -> None:
        cfg = make_cfg()
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "meeting.mp4"
            source.write_bytes(b"data")
            preferred_log_dir = project_dir() / "logs"

            def usable(path: Path) -> bool:
                return path != preferred_log_dir

            with patch(
                "transcriber.__main__.ensure_log_dir_usable", side_effect=usable
            ):
                outputs = output_paths_for_input(source, cfg, create_dirs=True)

            self.assertEqual(outputs.log_path.parent, source.parent)
            self.assertEqual(outputs.log_path.name, "meeting_whisperx.log")

    @patch("transcriber.__main__.run_whisperx_direct_logged")
    def test_transcribe_file_rejects_unsupported_input_before_work(
        self, run_logged: MagicMock
    ) -> None:
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
    def test_transcribe_file_rejects_directory_before_work(
        self, run_logged: MagicMock
    ) -> None:
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

    @patch("transcriber.__main__.run_whisperx_direct_logged")
    @patch("transcriber.__main__.preprocess_audio_for_whisperx")
    @patch(
        "transcriber.__main__.ensure_ffmpeg_available_for_child_processes",
        side_effect=RuntimeError(
            "ffmpeg executable not found. Install ffmpeg, set TRANSCRIBE_FFMPEG."
        ),
    )
    def test_transcribe_file_reports_missing_ffmpeg_before_audio_work(
        self,
        ensure_ffmpeg: MagicMock,
        preprocess_audio: MagicMock,
        run_logged: MagicMock,
    ) -> None:
        cfg = make_cfg(diarize=False)
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "meeting.wav"
            source.write_bytes(b"audio")
            outputs = output_paths_for_input(source, cfg, create_dirs=False)
            reports: list[str] = []

            rc = transcribe_file(cfg, source, report=reports.append)

            report_text = "\n".join(reports)
            self.assertEqual(rc, 1)
            self.assertIn("TRANSCRIBE_FFMPEG", report_text)
            self.assertNotIn("using the original input audio", report_text)
            self.assertFalse(outputs.lock_path.exists())
            preprocess_audio.assert_not_called()
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
        glossary = parse_glossary_entries(
            ["OpenAI => OpenAI", "esfuerzo|effort", "termino"]
        )
        prompt = build_translation_prompt(
            model_name="model", context_window=1, glossary=glossary
        )

        self.assertEqual(glossary["OpenAI"], "OpenAI")
        self.assertEqual(glossary["esfuerzo"], "effort")
        self.assertEqual(glossary["termino"], "termino")
        self.assertIn("__CUR_START__", prompt)
        self.assertIn("Glossary:", prompt)

    def test_asr_prompt_includes_glossary_and_file_terms(self) -> None:
        with TemporaryDirectory() as tmpdir:
            prompt_file = Path(tmpdir) / "asr.txt"
            prompt_file.write_text(
                "Project Falcon\n# comment\nAcmeOS\n", encoding="utf-8"
            )

            glossary = parse_glossary_entries(
                ["OpenAI => OpenAI", "WhisperX => WhisperX"]
            )
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
