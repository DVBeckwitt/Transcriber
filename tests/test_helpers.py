from __future__ import annotations

import contextlib
import io
import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import MagicMock, patch

from merge_transcripts import collect_transcript_files, merge_transcript_files
from transcriber import status as status_utils
from transcriber.__main__ import (
    DEFAULT_ESCUELA_DEST_DIR,
    DEFAULT_ESCUELA_LAST_EPISODE,
    DEFAULT_ESCUELA_WATCH_DIR,
    DEFAULT_WATCH_DIR,
    ESCUELA_EPISODE_COUNTER_FILE_NAME,
    ESCUELA_RENAME_STRATEGY,
    VIDEO_EXTENSIONS,
    RunConfig,
    TimedToken,
    apply_confidence_cleanup,
    build_asr_prompt,
    build_audio_preprocess_command,
    build_config,
    build_segment_fallback_cues,
    build_srt_cues_from_result,
    build_watch_targets,
    is_watchable_media,
    load_translation_glossary,
    main,
    move_completed_watch_outputs,
    needs_transcription,
    output_paths_for_input,
    parse_args,
    parse_glossary_entries,
    parse_temperature_schedule,
    print_summary,
    project_dir,
    render_uncertain_markup,
    run_whisperx_direct_logged,
    smooth_timed_tokens,
    transcribe_file,
    translate_spanish_texts,
)
from transcriber.preflight import PreflightReport, check_transcription_preflight
from transcriber.translation import TranslationRequest, TranslationResult


def make_cfg(**overrides: object) -> RunConfig:
    cfg = RunConfig(
        language="auto",
        translate_to_english=False,
        english_output_mode="off",
        direct_whisper_translate=False,
        post_translate_to_english=False,
        translation_backend="server",
        translation_model=None,
        translation_server_url=None,
        save_source_srt=True,
        write_llm_txt=True,
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
        speaker_labels=True,
        diarize=True,
        diarize_smoothing=True,
        min_speaker_turn_ms=900,
        min_speaker_turn_tokens=2,
        confidence_cleanup=True,
        confidence_cleanup_mode="mark",
        low_confidence_logprob=-1.0,
        high_no_speech_prob=0.6,
        low_confidence_word_prob=0.5,
        device="cpu",
        compute_type="float32",
        translation_batch_size=4,
        translation_max_new_tokens=256,
        glossary={},
        glossary_path=None,
        asr_prompt=None,
        dry_run=False,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def speaker_labeled_result() -> dict[str, Any]:
    return {
        "segments": [
            {
                "text": "Hello there",
                "start": 0.0,
                "end": 1.0,
                "speaker": "SPEAKER_00",
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
                    {"word": "there", "start": 0.5, "end": 1.0, "speaker": "SPEAKER_00"},
                ],
            }
        ]
    }


class FakePostTranslationBackend:
    name = "fake"
    model_name = "fake-model"

    def translate_texts(
        self,
        request: TranslationRequest,
        *,
        device: str,
        batch_size: int,
        max_new_tokens: int,
    ) -> TranslationResult:
        return TranslationResult(
            texts=[f"EN: {text}" for text in request.texts],
            backend=self.name,
            model=self.model_name,
            warnings=[],
        )


class FailingPostTranslationBackend:
    name = "failing"
    model_name = "failing-model"

    def translate_texts(
        self,
        request: TranslationRequest,
        *,
        device: str,
        batch_size: int,
        max_new_tokens: int,
    ) -> TranslationResult:
        raise RuntimeError("server unavailable")


class HelperTests(unittest.TestCase):
    def test_build_config_defaults_to_auto(self) -> None:
        cfg = build_config(parse_args([]), interactive=False)
        self.assertEqual(cfg.language, "auto")
        self.assertFalse(cfg.translate_to_english)
        self.assertEqual(cfg.english_output_mode, "auto")
        self.assertFalse(cfg.direct_whisper_translate)
        self.assertTrue(cfg.post_translate_to_english)
        self.assertTrue(cfg.speaker_labels)
        self.assertTrue(cfg.diarize)

    def test_fast_mode_defaults_to_no_speaker_labels(self) -> None:
        cfg = build_config(parse_args(["--mode", "fast"]), interactive=False)
        self.assertFalse(cfg.speaker_labels)
        self.assertFalse(cfg.diarize)
        self.assertEqual(cfg.english_output_mode, "off")

    def test_speaker_label_flags_control_diarization(self) -> None:
        labels_cfg = build_config(parse_args(["--speaker-labels"]), interactive=False)
        no_labels_cfg = build_config(parse_args(["--no-speaker-labels"]), interactive=False)

        self.assertTrue(labels_cfg.speaker_labels)
        self.assertTrue(labels_cfg.diarize)
        self.assertFalse(no_labels_cfg.speaker_labels)
        self.assertFalse(no_labels_cfg.diarize)

    def test_existing_diarize_flags_control_speaker_labels(self) -> None:
        diarize_cfg = build_config(parse_args(["--diarize"]), interactive=False)
        no_diarize_cfg = build_config(parse_args(["--no-diarize"]), interactive=False)

        self.assertTrue(diarize_cfg.speaker_labels)
        self.assertTrue(diarize_cfg.diarize)
        self.assertFalse(no_diarize_cfg.speaker_labels)
        self.assertFalse(no_diarize_cfg.diarize)

    @patch("transcriber.__main__.sys.stdin.isatty", return_value=True)
    @patch("builtins.input", side_effect=["off", "n"])
    def test_interactive_prompt_can_disable_quality_speaker_labels(
        self,
        _input: MagicMock,
        _isatty: MagicMock,
    ) -> None:
        cfg = build_config(parse_args(["--lang", "en", "--mode", "quality"]), interactive=True)

        self.assertFalse(cfg.speaker_labels)
        self.assertFalse(cfg.diarize)

    @patch("transcriber.__main__.sys.stdin.isatty", return_value=True)
    @patch("builtins.input", side_effect=["off", "y"])
    def test_interactive_prompt_can_enable_fast_speaker_labels(
        self,
        _input: MagicMock,
        _isatty: MagicMock,
    ) -> None:
        cfg = build_config(parse_args(["--lang", "en", "--mode", "fast"]), interactive=True)

        self.assertTrue(cfg.speaker_labels)
        self.assertTrue(cfg.diarize)

    @patch("transcriber.__main__.sys.stdin.isatty", return_value=True)
    @patch("builtins.input", return_value="off")
    def test_speaker_label_flags_skip_speaker_prompt(self, prompt: MagicMock, _isatty: MagicMock) -> None:
        cfg = build_config(
            parse_args(["--lang", "en", "--mode", "quality", "--no-speaker-labels"]),
            interactive=True,
        )

        self.assertFalse(cfg.speaker_labels)
        self.assertFalse(cfg.diarize)
        prompt.assert_called_once()

    def test_translate_flag_enables_direct_whisperx_output(self) -> None:
        cfg = build_config(parse_args(["--translate-to-english"]), interactive=False)
        self.assertTrue(cfg.translate_to_english)
        self.assertEqual(cfg.english_output_mode, "direct")
        self.assertTrue(cfg.direct_whisper_translate)
        self.assertFalse(cfg.post_translate_to_english)

    def test_english_output_mode_post_sets_post_translation(self) -> None:
        cfg = build_config(parse_args(["--english-output-mode", "post"]), interactive=False)

        self.assertFalse(cfg.translate_to_english)
        self.assertEqual(cfg.english_output_mode, "post")
        self.assertFalse(cfg.direct_whisper_translate)
        self.assertTrue(cfg.post_translate_to_english)

    def test_post_translate_shortcut_sets_post_translation(self) -> None:
        cfg = build_config(parse_args(["--post-translate-to-english"]), interactive=False)

        self.assertEqual(cfg.english_output_mode, "post")
        self.assertTrue(cfg.post_translate_to_english)

    @patch("transcriber.__main__.sys.stdin.isatty", return_value=True)
    @patch("builtins.input", side_effect=["post", "n"])
    def test_interactive_prompt_can_select_post_english_conversion(
        self,
        _input: MagicMock,
        _isatty: MagicMock,
    ) -> None:
        cfg = build_config(parse_args(["--lang", "es", "--mode", "quality"]), interactive=True)

        self.assertEqual(cfg.english_output_mode, "post")
        self.assertTrue(cfg.post_translate_to_english)

    def test_spanish_language_defaults_to_translation(self) -> None:
        cfg = build_config(parse_args(["--lang", "es"]), interactive=False)
        self.assertFalse(cfg.translate_to_english)
        self.assertEqual(cfg.english_output_mode, "auto")
        self.assertTrue(cfg.post_translate_to_english)

    def test_german_language_defaults_to_auto_post_translation(self) -> None:
        cfg = build_config(parse_args(["--lang", "de"]), interactive=False)

        self.assertEqual(cfg.language, "de")
        self.assertFalse(cfg.translate_to_english)
        self.assertEqual(cfg.english_output_mode, "auto")
        self.assertTrue(cfg.post_translate_to_english)

    def test_german_legacy_language_token_is_supported(self) -> None:
        cfg = build_config(parse_args(["g"]), interactive=False)

        self.assertEqual(cfg.language, "de")
        self.assertFalse(cfg.translate_to_english)
        self.assertEqual(cfg.english_output_mode, "auto")

    @patch("transcriber.__main__.sys.stdin.isatty", return_value=True)
    @patch("builtins.input", side_effect=["g", "auto"])
    def test_interactive_prompt_accepts_german(
        self,
        _input: MagicMock,
        _isatty: MagicMock,
    ) -> None:
        cfg = build_config(parse_args(["--mode", "quality", "--no-speaker-labels"]), interactive=True)

        self.assertEqual(cfg.language, "de")
        self.assertFalse(cfg.translate_to_english)
        self.assertEqual(cfg.english_output_mode, "auto")

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

    @patch("transcriber.__main__.run_whisperx_direct")
    def test_logged_whisperx_run_mirrors_progress_to_console_and_log(self, run_direct: MagicMock) -> None:
        def fake_run(*_args: object, **_kwargs: object) -> str:
            print("[transcriber] fake stdout progress")
            print("[transcriber] fake stderr progress", file=sys.stderr)
            return "en"

        run_direct.side_effect = fake_run
        with TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "run.log"
            stdout = io.StringIO()
            stderr = io.StringIO()

            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                rc, detected_language = run_whisperx_direct_logged(
                    make_cfg(),
                    Path("input.wav"),
                    Path(tmpdir) / "output.srt",
                    None,
                    False,
                    log_path,
                )

            log_text = log_path.read_text(encoding="utf-8")
            self.assertEqual(rc, 0)
            self.assertEqual(detected_language, "en")
            self.assertIn("fake stdout progress", stdout.getvalue())
            self.assertIn("fake stderr progress", stderr.getvalue())
            self.assertIn("fake stdout progress", log_text)
            self.assertIn("fake stderr progress", log_text)

    def test_opus_files_are_watchable(self) -> None:
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "clip.opus"
            source.write_bytes(b"data")
            self.assertTrue(is_watchable_media(source))

    def test_escuela_target_only_watches_videos(self) -> None:
        with TemporaryDirectory() as tmpdir:
            audio = Path(tmpdir) / "clip.mp3"
            video = Path(tmpdir) / "clip.mp4"
            audio.write_bytes(b"audio")
            video.write_bytes(b"video")

            self.assertFalse(is_watchable_media(audio, VIDEO_EXTENSIONS))
            self.assertTrue(is_watchable_media(video, VIDEO_EXTENSIONS))

    def test_default_recordings_watch_also_adds_escuela_target(self) -> None:
        targets = build_watch_targets(make_cfg(), DEFAULT_WATCH_DIR)

        self.assertEqual([target.watch_dir for target in targets], [DEFAULT_WATCH_DIR, DEFAULT_ESCUELA_WATCH_DIR])
        self.assertEqual(targets[0].cfg.language, "auto")
        self.assertTrue(targets[1].cfg.translate_to_english)
        self.assertEqual(targets[1].cfg.english_output_mode, "direct")
        self.assertTrue(targets[1].cfg.direct_whisper_translate)
        self.assertEqual(targets[1].cfg.language, "es")
        self.assertFalse(targets[1].cfg.speaker_labels)
        self.assertFalse(targets[1].cfg.diarize)
        self.assertFalse(targets[1].cfg.write_llm_txt)
        self.assertEqual(targets[1].allowed_extensions, frozenset(VIDEO_EXTENSIONS))
        self.assertEqual(targets[1].move_completed_files_to, DEFAULT_ESCUELA_DEST_DIR)
        self.assertEqual(targets[1].rename_strategy, ESCUELA_RENAME_STRATEGY)

    def test_direct_escuela_watch_uses_special_policy(self) -> None:
        targets = build_watch_targets(make_cfg(), DEFAULT_ESCUELA_WATCH_DIR)

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].watch_dir, DEFAULT_ESCUELA_WATCH_DIR)
        self.assertEqual(targets[0].cfg.language, "es")
        self.assertTrue(targets[0].cfg.translate_to_english)
        self.assertEqual(targets[0].cfg.english_output_mode, "direct")
        self.assertFalse(targets[0].cfg.speaker_labels)
        self.assertFalse(targets[0].cfg.diarize)
        self.assertFalse(targets[0].cfg.write_llm_txt)
        self.assertEqual(targets[0].move_completed_files_to, DEFAULT_ESCUELA_DEST_DIR)
        self.assertEqual(targets[0].rename_strategy, ESCUELA_RENAME_STRATEGY)

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

    def test_output_paths_include_post_translation_files(self) -> None:
        cfg = make_cfg()
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "meeting.mp4"
            source.write_bytes(b"data")
            outputs = output_paths_for_input(source, cfg, create_dirs=False)

            self.assertEqual(outputs.source_srt_path, source.parent / "meeting.source.srt")
            self.assertEqual(outputs.english_srt_path, source.parent / "meeting.en.srt")
            self.assertEqual(outputs.translation_report_path, source.parent / "meeting.translation.json")

    def test_output_paths_include_status_file(self) -> None:
        cfg = make_cfg()
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "meeting.mp4"
            source.write_bytes(b"data")
            outputs = output_paths_for_input(source, cfg, create_dirs=False)

            self.assertEqual(outputs.status_path, source.parent / "meeting.transcription.json")

    def test_needs_transcription_rejects_empty_partial_srt(self) -> None:
        cfg = make_cfg(speaker_labels=False, diarize=False)
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "meeting.wav"
            source.write_bytes(b"audio")
            outputs = output_paths_for_input(source, cfg, create_dirs=False)
            outputs.srt_path.write_text("", encoding="utf-8")

            self.assertTrue(needs_transcription(source, cfg))

    def test_needs_transcription_accepts_matching_success_status(self) -> None:
        cfg = make_cfg(speaker_labels=False, diarize=False)
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "meeting.wav"
            source.write_bytes(b"audio")
            outputs = output_paths_for_input(source, cfg, create_dirs=False)
            outputs.srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
            status_utils.write_status_file(
                outputs.status_path,
                status_utils.build_status_record(
                    status="succeeded",
                    source_path=source,
                    config=cfg,
                    srt_path=outputs.srt_path,
                    started_at="2026-01-01T00:00:00Z",
                    finished_at="2026-01-01T00:00:01Z",
                ),
            )

            self.assertFalse(needs_transcription(source, cfg))
            self.assertTrue(needs_transcription(source, make_cfg(speaker_labels=False, diarize=False, model="medium")))

    @patch("transcriber.__main__.check_transcription_preflight", return_value=PreflightReport())
    @patch("transcriber.__main__.preprocess_audio_for_whisperx")
    @patch("transcriber.__main__.run_whisperx_direct_logged")
    def test_transcribe_file_writes_atomic_outputs_and_status(
        self,
        run_logged: MagicMock,
        preprocess: MagicMock,
        _preflight: MagicMock,
    ) -> None:
        cfg = make_cfg(
            speaker_labels=False,
            diarize=False,
            write_llm_txt=False,
            confidence_cleanup=False,
            compute_type="float32",
        )
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "meeting.wav"
            source.write_bytes(b"audio")
            preprocess.return_value = source

            def fake_run(*args: object, **_kwargs: object) -> tuple[int, str]:
                srt_path = args[2]
                assert isinstance(srt_path, Path)
                srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
                return 0, "en"

            run_logged.side_effect = fake_run
            messages: list[str] = []

            rc = transcribe_file(cfg, source, report=messages.append, stale_lock_seconds=1)
            outputs = output_paths_for_input(source, cfg)
            record = status_utils.read_status_file(outputs.status_path)

            self.assertEqual(rc, 0)
            self.assertTrue(outputs.srt_path.exists())
            self.assertFalse(outputs.lock_path.exists())
            self.assertFalse(list(source.parent.glob(".*.srt.tmp")))
            self.assertIsNotNone(record)
            self.assertEqual((record or {}).get("status"), "succeeded")

    @patch("transcriber.__main__.build_post_translation_backend", return_value=FakePostTranslationBackend())
    @patch("transcriber.__main__.check_transcription_preflight", return_value=PreflightReport())
    @patch("transcriber.__main__.preprocess_audio_for_whisperx")
    @patch("transcriber.__main__.run_whisperx_direct_logged")
    def test_transcribe_file_post_translates_spanish_and_preserves_source_srt(
        self,
        run_logged: MagicMock,
        preprocess: MagicMock,
        _preflight: MagicMock,
        _backend: MagicMock,
    ) -> None:
        cfg = make_cfg(
            language="es",
            english_output_mode="post",
            post_translate_to_english=True,
            speaker_labels=False,
            diarize=False,
            compute_type="float32",
        )
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "meeting.wav"
            source.write_bytes(b"audio")
            preprocess.return_value = source

            def fake_run(*args: object, **_kwargs: object) -> tuple[int, str]:
                srt_path = args[2]
                assert isinstance(srt_path, Path)
                srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nSPEAKER_00: Hola\n", encoding="utf-8")
                return 0, "es"

            run_logged.side_effect = fake_run

            rc = transcribe_file(cfg, source, report=lambda _: None, stale_lock_seconds=1)
            outputs = output_paths_for_input(source, cfg)
            record = status_utils.read_status_file(outputs.status_path) or {}

            self.assertEqual(rc, 0)
            self.assertEqual(
                (source.parent / "meeting.es.srt").read_text(encoding="utf-8").strip().splitlines()[-1],
                "SPEAKER_00: Hola",
            )
            self.assertIn("SPEAKER_00: EN: Hola", outputs.english_srt_path.read_text(encoding="utf-8"))
            self.assertEqual(
                outputs.srt_path.read_text(encoding="utf-8"), outputs.english_srt_path.read_text(encoding="utf-8")
            )
            self.assertIn("SPEAKER_00: EN: Hola", outputs.llm_path.read_text(encoding="utf-8"))
            self.assertEqual(record.get("english_output_mode"), "post")

    @patch("transcriber.__main__.build_post_translation_backend", return_value=FakePostTranslationBackend())
    @patch("transcriber.__main__.check_transcription_preflight", return_value=PreflightReport())
    @patch("transcriber.__main__.preprocess_audio_for_whisperx")
    @patch("transcriber.__main__.run_whisperx_direct_logged")
    def test_transcribe_file_auto_post_translates_german_when_detected(
        self,
        run_logged: MagicMock,
        preprocess: MagicMock,
        _preflight: MagicMock,
        _backend: MagicMock,
    ) -> None:
        cfg = make_cfg(
            language="auto",
            english_output_mode="auto",
            post_translate_to_english=True,
            speaker_labels=False,
            diarize=False,
            compute_type="float32",
        )
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "meeting.wav"
            source.write_bytes(b"audio")
            preprocess.return_value = source

            def fake_run(*args: object, **_kwargs: object) -> tuple[int, str]:
                srt_path = args[2]
                assert isinstance(srt_path, Path)
                srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHallo\n", encoding="utf-8")
                return 0, "de"

            run_logged.side_effect = fake_run

            rc = transcribe_file(cfg, source, report=lambda _: None, stale_lock_seconds=1)
            outputs = output_paths_for_input(source, cfg)

            self.assertEqual(rc, 0)
            self.assertTrue((source.parent / "meeting.de.srt").exists())
            self.assertIn("EN: Hallo", outputs.english_srt_path.read_text(encoding="utf-8"))

    @patch("transcriber.__main__.build_post_translation_backend", return_value=FakePostTranslationBackend())
    @patch("transcriber.__main__.check_transcription_preflight", return_value=PreflightReport())
    @patch("transcriber.__main__.preprocess_audio_for_whisperx")
    @patch("transcriber.__main__.run_whisperx_direct_logged")
    def test_transcribe_file_honors_no_save_source_srt(
        self,
        run_logged: MagicMock,
        preprocess: MagicMock,
        _preflight: MagicMock,
        _backend: MagicMock,
    ) -> None:
        cfg = make_cfg(
            language="es",
            english_output_mode="post",
            post_translate_to_english=True,
            save_source_srt=False,
            speaker_labels=False,
            diarize=False,
            compute_type="float32",
        )
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "meeting.wav"
            source.write_bytes(b"audio")
            preprocess.return_value = source

            def fake_run(*args: object, **_kwargs: object) -> tuple[int, str]:
                srt_path = args[2]
                assert isinstance(srt_path, Path)
                srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHola\n", encoding="utf-8")
                return 0, "es"

            run_logged.side_effect = fake_run

            rc = transcribe_file(cfg, source, report=lambda _: None, stale_lock_seconds=1)
            outputs = output_paths_for_input(source, cfg)

            self.assertEqual(rc, 0)
            self.assertFalse((source.parent / "meeting.es.srt").exists())
            self.assertTrue(outputs.english_srt_path.exists())

    @patch("transcriber.__main__.build_post_translation_backend")
    @patch("transcriber.__main__.check_transcription_preflight", return_value=PreflightReport())
    @patch("transcriber.__main__.preprocess_audio_for_whisperx")
    @patch("transcriber.__main__.run_whisperx_direct_logged")
    def test_transcribe_file_post_mode_skips_english_source(
        self,
        run_logged: MagicMock,
        preprocess: MagicMock,
        _preflight: MagicMock,
        backend: MagicMock,
    ) -> None:
        cfg = make_cfg(
            language="en",
            english_output_mode="post",
            post_translate_to_english=True,
            speaker_labels=False,
            diarize=False,
            write_llm_txt=False,
            compute_type="float32",
        )
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "meeting.wav"
            source.write_bytes(b"audio")
            preprocess.return_value = source

            def fake_run(*args: object, **_kwargs: object) -> tuple[int, str]:
                srt_path = args[2]
                assert isinstance(srt_path, Path)
                srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
                return 0, "en"

            run_logged.side_effect = fake_run

            rc = transcribe_file(cfg, source, report=lambda _: None, stale_lock_seconds=1)
            outputs = output_paths_for_input(source, cfg)

            self.assertEqual(rc, 0)
            self.assertEqual(outputs.srt_path.read_text(encoding="utf-8").strip().splitlines()[-1], "Hello")
            self.assertFalse(outputs.english_srt_path.exists())
            backend.assert_not_called()

    @patch("transcriber.__main__.build_post_translation_backend", return_value=FailingPostTranslationBackend())
    @patch("transcriber.__main__.check_transcription_preflight", return_value=PreflightReport())
    @patch("transcriber.__main__.preprocess_audio_for_whisperx")
    @patch("transcriber.__main__.run_whisperx_direct_logged")
    def test_transcribe_file_explicit_post_fails_when_translation_fails(
        self,
        run_logged: MagicMock,
        preprocess: MagicMock,
        _preflight: MagicMock,
        _backend: MagicMock,
    ) -> None:
        cfg = make_cfg(
            language="es",
            english_output_mode="post",
            post_translate_to_english=True,
            speaker_labels=False,
            diarize=False,
            write_llm_txt=False,
            compute_type="float32",
        )
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "meeting.wav"
            source.write_bytes(b"audio")
            preprocess.return_value = source

            def fake_run(*args: object, **_kwargs: object) -> tuple[int, str]:
                srt_path = args[2]
                assert isinstance(srt_path, Path)
                srt_path.write_text("1\n00:00:00,000 --> 00:00:01,000\nHola\n", encoding="utf-8")
                return 0, "es"

            run_logged.side_effect = fake_run

            rc = transcribe_file(cfg, source, report=lambda _: None, stale_lock_seconds=1)
            outputs = output_paths_for_input(source, cfg)
            record = status_utils.read_status_file(outputs.status_path) or {}

            self.assertEqual(rc, 1)
            self.assertEqual(record.get("status"), "failed")
            self.assertIn("Explicit post-translation failed", str(record.get("error")))
            self.assertTrue((source.parent / "meeting.es.srt").exists())
            self.assertFalse(outputs.srt_path.exists())
            self.assertFalse(outputs.english_srt_path.exists())

    @patch("transcriber.__main__.resolve_input_path")
    @patch("transcriber.__main__.transcribe_file")
    def test_main_passes_stale_lock_seconds_to_transcribe(
        self,
        transcribe: MagicMock,
        resolve_input: MagicMock,
    ) -> None:
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "meeting.wav"
            source.write_bytes(b"audio")
            resolve_input.return_value = source
            transcribe.return_value = 0

            rc = main(["--input", str(source), "--no-speaker-labels", "--stale-lock-seconds", "3"])

            self.assertEqual(rc, 0)
            self.assertEqual(transcribe.call_args.kwargs["stale_lock_seconds"], 3)

    def test_summary_reports_speaker_label_choice(self) -> None:
        cfg = make_cfg(speaker_labels=False, diarize=False)
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "meeting.mp4"
            outputs = output_paths_for_input(source, cfg, create_dirs=False)

            messages: list[str] = []
            print_summary(cfg, source, outputs, report=messages.append)

        self.assertIn("Speakers:  off", messages)
        self.assertIn("Diarize:   off", messages)

    def test_move_completed_watch_outputs_moves_video_and_srt(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            dest_dir = root / "dest"
            source_dir.mkdir()

            video = source_dir / "episode.mp4"
            video.write_bytes(b"video")
            srt = source_dir / "episode.srt"
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
            outputs = output_paths_for_input(video, make_cfg(), create_dirs=False)

            messages: list[str] = []
            moved = move_completed_watch_outputs(video, outputs, dest_dir, messages.append)

            self.assertTrue(moved)
            self.assertFalse(video.exists())
            self.assertFalse(srt.exists())
            self.assertTrue((dest_dir / "episode.mp4").exists())
            self.assertTrue((dest_dir / "episode.srt").exists())
            self.assertTrue(any("Moved" in message for message in messages))

    def test_move_completed_watch_outputs_missing_srt_leaves_source_in_place(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            dest_dir = root / "dest"
            source_dir.mkdir()

            video = source_dir / "episode.mp4"
            video.write_bytes(b"video")
            outputs = output_paths_for_input(video, make_cfg(), create_dirs=False)

            messages: list[str] = []
            moved = move_completed_watch_outputs(video, outputs, dest_dir, messages.append)

            self.assertFalse(moved)
            self.assertTrue(video.exists())
            self.assertFalse((dest_dir / "episode.mp4").exists())
            self.assertTrue(any("missing" in message for message in messages))

    def test_move_completed_watch_outputs_rolls_back_video_when_srt_move_fails(self) -> None:
        original_move = shutil.move

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            dest_dir = root / "dest"
            source_dir.mkdir()

            video = source_dir / "episode.mp4"
            video.write_bytes(b"video")
            srt = source_dir / "episode.srt"
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
            outputs = output_paths_for_input(video, make_cfg(), create_dirs=False)

            def fail_srt_move(src: str, dst: str) -> str:
                if Path(src) == srt:
                    raise OSError("srt locked")
                return str(original_move(src, dst))

            messages: list[str] = []
            with patch("transcriber.__main__.shutil.move", side_effect=fail_srt_move):
                moved = move_completed_watch_outputs(video, outputs, dest_dir, messages.append)

            self.assertFalse(moved)
            self.assertTrue(video.exists())
            self.assertTrue(srt.exists())
            self.assertFalse((dest_dir / "episode.mp4").exists())
            self.assertTrue(any("Rolled" in message for message in messages))

    @patch("transcriber.__main__.translate_spanish_texts")
    def test_move_completed_escuela_outputs_renames_and_updates_counter(self, translate_texts: MagicMock) -> None:
        translate_texts.return_value = ["My First Job"]

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            dest_dir = root / "dest"
            source_dir.mkdir()
            dest_dir.mkdir()

            (dest_dir / ESCUELA_EPISODE_COUNTER_FILE_NAME).write_text(
                f"{DEFAULT_ESCUELA_LAST_EPISODE}\n", encoding="utf-8"
            )

            video = source_dir / "mi_primer_trabajo.mp4"
            video.write_bytes(b"video")
            srt = source_dir / "mi_primer_trabajo.srt"
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
            outputs = output_paths_for_input(video, make_cfg(), create_dirs=False)

            messages: list[str] = []
            moved = move_completed_watch_outputs(
                video,
                outputs,
                dest_dir,
                messages.append,
                cfg=make_cfg(
                    language="es",
                    translate_to_english=True,
                    speaker_labels=False,
                    diarize=False,
                    write_llm_txt=False,
                ),
                rename_strategy=ESCUELA_RENAME_STRATEGY,
            )

            self.assertTrue(moved)
            expected_base = "Escuela de Nada - s01e730 - My First Job"
            self.assertTrue((dest_dir / f"{expected_base}.mp4").exists())
            self.assertTrue((dest_dir / f"{expected_base}.srt").exists())
            self.assertEqual(
                (dest_dir / ESCUELA_EPISODE_COUNTER_FILE_NAME).read_text(encoding="utf-8").strip(),
                "730",
            )
            self.assertFalse(video.exists())
            self.assertFalse(srt.exists())
            translate_texts.assert_called_once()

    @patch("transcriber.__main__.translate_spanish_texts")
    def test_move_completed_escuela_outputs_skips_taken_episode_numbers(self, translate_texts: MagicMock) -> None:
        translate_texts.return_value = ["Next Thing"]

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            dest_dir = root / "dest"
            source_dir.mkdir()
            dest_dir.mkdir()

            (dest_dir / ESCUELA_EPISODE_COUNTER_FILE_NAME).write_text(
                f"{DEFAULT_ESCUELA_LAST_EPISODE}\n", encoding="utf-8"
            )
            (dest_dir / "Escuela de Nada - s01e730 - Existing.mp4").write_bytes(b"video")
            (dest_dir / "Escuela de Nada - s01e730 - Existing.srt").write_text("existing", encoding="utf-8")

            video = source_dir / "algo_nuevo.mp4"
            video.write_bytes(b"video")
            srt = source_dir / "algo_nuevo.srt"
            srt.write_text("1\n00:00:00,000 --> 00:00:01,000\nHello\n", encoding="utf-8")
            outputs = output_paths_for_input(video, make_cfg(), create_dirs=False)

            moved = move_completed_watch_outputs(
                video,
                outputs,
                dest_dir,
                lambda _: None,
                cfg=make_cfg(
                    language="es",
                    translate_to_english=True,
                    speaker_labels=False,
                    diarize=False,
                    write_llm_txt=False,
                ),
                rename_strategy=ESCUELA_RENAME_STRATEGY,
            )

            self.assertTrue(moved)
            expected_base = "Escuela de Nada - s01e731 - Next Thing"
            self.assertTrue((dest_dir / f"{expected_base}.mp4").exists())
            self.assertTrue((dest_dir / f"{expected_base}.srt").exists())

    def test_glossary_parsing(self) -> None:
        glossary = parse_glossary_entries(["OpenAI => OpenAI", "esfuerzo|effort", "termino"])

        self.assertEqual(glossary["OpenAI"], "OpenAI")
        self.assertEqual(glossary["esfuerzo"], "effort")
        self.assertEqual(glossary["termino"], "termino")

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

    @patch("transcriber.__main__.load_spanish_to_english_translator")
    def test_translation_uses_beam_search(self, load_translator: MagicMock) -> None:
        class FakeTensor:
            def to(self, _device: str) -> FakeTensor:
                return self

        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": FakeTensor(),
            "attention_mask": FakeTensor(),
        }
        tokenizer.batch_decode.return_value = ["Hello there"]

        model = MagicMock()
        model.generate.return_value = FakeTensor()
        load_translator.return_value = (tokenizer, model)

        result = translate_spanish_texts(["Hola"], device="cpu", batch_size=1)

        self.assertEqual(result, ["Hello there"])
        model.generate.assert_called_once()
        kwargs = model.generate.call_args.kwargs
        self.assertEqual(kwargs["num_beams"], 4)
        self.assertEqual(kwargs["length_penalty"], 1.0)
        self.assertEqual(kwargs["no_repeat_ngram_size"], 3)
        self.assertTrue(kwargs["early_stopping"])

    @patch("transcriber.preflight.module_available", return_value=True)
    def test_preflight_fails_when_explicit_post_translation_has_no_url(self, _module_available: MagicMock) -> None:
        cfg = make_cfg(
            language="de",
            english_output_mode="post",
            post_translate_to_english=True,
            translation_backend="server",
            translation_server_url=None,
            diarize=False,
            device="cpu",
        )

        report = check_transcription_preflight(cfg=cfg, hf_token_present=False, require_ffmpeg=False)

        self.assertTrue(any("OpenAI-compatible" in error for error in report.errors))
        self.assertFalse(any("Spanish auto-translation" in warning for warning in report.warnings))

    @patch("transcriber.preflight.module_available", return_value=True)
    def test_preflight_warns_when_auto_post_translation_has_no_url(self, _module_available: MagicMock) -> None:
        cfg = make_cfg(
            language="auto",
            english_output_mode="auto",
            post_translate_to_english=True,
            translation_backend="server",
            translation_server_url=None,
            diarize=False,
            device="cpu",
        )

        report = check_transcription_preflight(cfg=cfg, hf_token_present=False, require_ffmpeg=False)

        self.assertTrue(any("OpenAI-compatible" in warning for warning in report.warnings))
        self.assertFalse(report.errors)

    @patch("transcriber.preflight.module_available", return_value=True)
    def test_preflight_does_not_warn_about_server_for_direct_translation(self, _module_available: MagicMock) -> None:
        cfg = make_cfg(
            english_output_mode="direct",
            direct_whisper_translate=True,
            post_translate_to_english=False,
            translation_server_url=None,
            diarize=False,
            device="cpu",
        )

        report = check_transcription_preflight(cfg=cfg, hf_token_present=False, require_ffmpeg=False)

        self.assertFalse(any("OpenAI-compatible" in warning for warning in report.warnings))

    def test_confidence_cleanup_marks_low_confidence(self) -> None:
        cfg = make_cfg()
        result: dict[str, Any] = {
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

    def test_uncertain_markup_renders_for_srt_and_llm(self) -> None:
        marker = "__UNCERTAIN_65__hola__UNCERTAIN_END__"
        self.assertEqual(render_uncertain_markup(marker, "srt"), "hola")
        self.assertEqual(render_uncertain_markup(marker, "llm"), "[hola] [65% confidence]")

    def test_speaker_smoothing_merges_short_blips(self) -> None:
        tokens = [
            TimedToken(text="Hello", start_ms=0, end_ms=300, speaker="SPEAKER_00"),
            TimedToken(text="yes", start_ms=300, end_ms=500, speaker="SPEAKER_01", confidence=0.2),
            TimedToken(text="there", start_ms=500, end_ms=1100, speaker="SPEAKER_00"),
        ]

        smoothed = smooth_timed_tokens(tokens)

        self.assertEqual([token.speaker for token in smoothed], ["SPEAKER_00", "SPEAKER_00", "SPEAKER_00"])

    def test_srt_cues_omit_speaker_prefixes_when_labels_are_disabled(self) -> None:
        cues = build_srt_cues_from_result(speaker_labeled_result(), make_cfg(speaker_labels=False, diarize=False))

        self.assertEqual([cue.text for cue in cues], ["Hello there"])

    def test_srt_cues_preserve_speaker_prefixes_when_labels_are_enabled(self) -> None:
        cues = build_srt_cues_from_result(speaker_labeled_result(), make_cfg(speaker_labels=True, diarize=True))

        self.assertEqual([cue.text for cue in cues], ["SPEAKER_00: Hello there"])

    def test_segment_fallback_cues_omit_speaker_prefixes_when_labels_are_disabled(self) -> None:
        result: dict[str, Any] = {
            "segments": [
                {
                    "text": "Fallback text",
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "SPEAKER_00",
                }
            ]
        }

        cues = build_segment_fallback_cues(result, speaker_labels=False)

        self.assertEqual([cue.text for cue in cues], ["Fallback text"])

    def test_merge_transcripts_recursively_skips_output_and_non_transcripts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "nested"
            nested.mkdir()
            (root / "one.txt").write_text("First transcript", encoding="utf-8")
            (nested / "two.txt").write_text("Second transcript", encoding="utf-8")
            (root / "HF_TOKEN.txt").write_text("secret", encoding="utf-8")
            (root / "hf_token.txt").write_text("lower secret", encoding="utf-8")
            (root / "notes.md").write_text("ignore me", encoding="utf-8")
            cache_dir = root / ".uv-venv"
            cache_dir.mkdir()
            (cache_dir / "cached.txt").write_text("cache text", encoding="utf-8")
            logs_dir = root / "logs"
            logs_dir.mkdir()
            (logs_dir / "run.txt").write_text("log text", encoding="utf-8")

            output = root / "merged_transcript.txt"
            count, resolved_output = merge_transcript_files(root, output)

            self.assertEqual(count, 2)
            self.assertEqual(resolved_output, output.resolve())

            merged = output.read_text(encoding="utf-8")
            self.assertIn("===== 1. nested/two.txt =====", merged)
            self.assertIn("===== 2. one.txt =====", merged)
            self.assertIn("First transcript", merged)
            self.assertIn("Second transcript", merged)
            self.assertNotIn("secret", merged)
            self.assertNotIn("lower secret", merged)
            self.assertNotIn("cache text", merged)
            self.assertNotIn("log text", merged)

            collected = collect_transcript_files(root, output)
            self.assertEqual([path.relative_to(root).as_posix() for path in collected], ["nested/two.txt", "one.txt"])


if __name__ == "__main__":
    unittest.main()
