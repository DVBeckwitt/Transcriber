from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import transcriber.__main__ as transcriber_main
from tests.support import make_cfg
from transcriber.__main__ import (
    flush_gpu_memory,
    transcribe_file,
)


class ModelLifecycleTests(unittest.TestCase):
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

    @patch("transcriber.__main__.flush_gpu_memory")
    def test_cold_cleanup_clears_warm_model_caches_and_flushes_gpu(
        self, flush_gpu: MagicMock
    ) -> None:
        transcriber_main._WARM_ASR_MODEL_CACHE[("asr",)] = object()
        transcriber_main._WARM_ALIGN_MODEL_CACHE[("align",)] = object()
        transcriber_main._WARM_DIARIZATION_MODEL_CACHE[("diarize",)] = object()

        transcriber_main.cleanup_after_transcription_run(
            make_cfg(device="cuda", warm_vram=False)
        )

        self.assertEqual(transcriber_main._WARM_ASR_MODEL_CACHE, {})
        self.assertEqual(transcriber_main._WARM_ALIGN_MODEL_CACHE, {})
        self.assertEqual(transcriber_main._WARM_DIARIZATION_MODEL_CACHE, {})
        flush_gpu.assert_called_once_with("cuda")

    @patch("transcriber.__main__.flush_gpu_memory")
    def test_warm_cleanup_preserves_model_caches_and_skips_gpu_flush(
        self, flush_gpu: MagicMock
    ) -> None:
        asr_model = object()
        align_model = object()
        diarize_model = object()
        transcriber_main._WARM_ASR_MODEL_CACHE[("asr",)] = asr_model
        transcriber_main._WARM_ALIGN_MODEL_CACHE[("align",)] = align_model
        transcriber_main._WARM_DIARIZATION_MODEL_CACHE[("diarize",)] = diarize_model

        try:
            transcriber_main.cleanup_after_transcription_run(
                make_cfg(device="cuda", warm_vram=True)
            )

            self.assertIs(transcriber_main._WARM_ASR_MODEL_CACHE[("asr",)], asr_model)
            self.assertIs(
                transcriber_main._WARM_ALIGN_MODEL_CACHE[("align",)], align_model
            )
            self.assertIs(
                transcriber_main._WARM_DIARIZATION_MODEL_CACHE[("diarize",)],
                diarize_model,
            )
            flush_gpu.assert_not_called()
        finally:
            transcriber_main.clear_warm_model_caches()

    def test_warm_vram_reuses_asr_and_align_models(self) -> None:
        model = MagicMock()
        model.transcribe.return_value = {
            "language": "en",
            "segments": [
                {
                    "words": [
                        {"word": "Hello", "start": 0.0, "end": 0.5},
                        {"word": "there.", "start": 0.5, "end": 1.0},
                    ]
                }
            ],
        }
        fake_whisperx = SimpleNamespace(
            __name__="whisperx",
            load_model=MagicMock(return_value=model),
            load_audio=MagicMock(return_value=object()),
            load_align_model=MagicMock(return_value=(object(), {"language": "en"})),
            align=MagicMock(
                side_effect=lambda segments, *args, **kwargs: {
                    "language": "en",
                    "segments": segments,
                }
            ),
        )

        with (
            TemporaryDirectory() as tmpdir,
            patch.dict(sys.modules, {"whisperx": fake_whisperx}),
            patch(
                "transcriber.__main__.ensure_ffmpeg_available_for_child_processes",
                return_value="ffmpeg",
            ),
        ):
            cfg = make_cfg(language="en", diarize=False, warm_vram=True)
            try:
                for idx in range(2):
                    transcriber_main.run_whisperx_direct(
                        cfg,
                        Path("input.wav"),
                        Path(tmpdir) / f"output-{idx}.srt",
                        hf_token=None,
                        diarize=False,
                    )
            finally:
                transcriber_main.clear_warm_model_caches()

        self.assertEqual(fake_whisperx.load_model.call_count, 1)
        self.assertEqual(fake_whisperx.load_align_model.call_count, 1)
        self.assertEqual(model.transcribe.call_count, 2)

    @patch("transcriber.__main__.ensure_ffmpeg_available_for_child_processes")
    def test_run_whisperx_direct_prepares_ffmpeg_before_model_load(
        self, ensure_ffmpeg: MagicMock
    ) -> None:
        events: list[str] = []
        model = MagicMock()
        model.transcribe.return_value = {
            "language": "en",
            "segments": [{"words": [{"word": "Hello", "start": 0.0, "end": 0.5}]}],
        }

        def prepare_ffmpeg() -> str:
            events.append("ensure_ffmpeg")
            return "ffmpeg"

        def load_model(*args: object, **kwargs: object) -> MagicMock:
            events.append("load_model")
            return model

        def load_audio(*args: object, **kwargs: object) -> object:
            events.append("load_audio")
            return object()

        ensure_ffmpeg.side_effect = prepare_ffmpeg
        fake_whisperx = SimpleNamespace(
            __name__="whisperx",
            load_model=MagicMock(side_effect=load_model),
            load_audio=MagicMock(side_effect=load_audio),
            load_align_model=MagicMock(return_value=(object(), {"language": "en"})),
            align=MagicMock(
                side_effect=lambda segments, *args, **kwargs: {
                    "language": "en",
                    "segments": segments,
                }
            ),
        )

        with (
            TemporaryDirectory() as tmpdir,
            patch.dict(sys.modules, {"whisperx": fake_whisperx}),
        ):
            transcriber_main.run_whisperx_direct(
                make_cfg(language="en", diarize=False),
                Path("input.wav"),
                Path(tmpdir) / "output.srt",
                hf_token=None,
                diarize=False,
            )

        self.assertLess(events.index("ensure_ffmpeg"), events.index("load_model"))
        self.assertLess(events.index("ensure_ffmpeg"), events.index("load_audio"))

    @patch("transcriber.__main__.ensure_ffmpeg_available_for_child_processes")
    def test_run_whisperx_direct_fails_before_model_load_when_ffmpeg_missing(
        self, ensure_ffmpeg: MagicMock
    ) -> None:
        ensure_ffmpeg.side_effect = RuntimeError(
            "ffmpeg executable not found. Set TRANSCRIBE_FFMPEG."
        )
        fake_whisperx = SimpleNamespace(
            __name__="whisperx",
            load_model=MagicMock(),
            load_audio=MagicMock(),
            load_align_model=MagicMock(),
            align=MagicMock(),
        )

        with (
            TemporaryDirectory() as tmpdir,
            patch.dict(sys.modules, {"whisperx": fake_whisperx}),
        ):
            with self.assertRaisesRegex(RuntimeError, "TRANSCRIBE_FFMPEG"):
                transcriber_main.run_whisperx_direct(
                    make_cfg(language="en", diarize=False),
                    Path("input.wav"),
                    Path(tmpdir) / "output.srt",
                    hf_token=None,
                    diarize=False,
                )

        fake_whisperx.load_model.assert_not_called()
        fake_whisperx.load_audio.assert_not_called()

    def test_warm_vram_keeps_direct_run_inside_cache_lock(self) -> None:
        class RecordingLock:
            def __init__(self) -> None:
                self.depth = 0
                self.enter_count = 0

            def __enter__(self) -> RecordingLock:
                self.depth += 1
                self.enter_count += 1
                return self

            def __exit__(
                self, exc_type: object, exc: object, traceback: object
            ) -> None:
                self.depth -= 1

        lock = RecordingLock()
        model = MagicMock()

        def transcribe(*args: object, **kwargs: object) -> dict[str, object]:
            self.assertGreater(lock.depth, 0)
            return {
                "language": "en",
                "segments": [{"words": [{"word": "Hello", "start": 0.0, "end": 0.5}]}],
            }

        model.transcribe.side_effect = transcribe
        fake_whisperx = SimpleNamespace(
            __name__="whisperx",
            load_model=MagicMock(return_value=model),
            load_audio=MagicMock(return_value=object()),
            load_align_model=MagicMock(return_value=(object(), {"language": "en"})),
            align=MagicMock(
                side_effect=lambda segments, *args, **kwargs: {
                    "language": "en",
                    "segments": segments,
                }
            ),
        )

        with (
            TemporaryDirectory() as tmpdir,
            patch.dict(sys.modules, {"whisperx": fake_whisperx}),
            patch("transcriber.__main__._WARM_MODEL_CACHE_LOCK", lock),
            patch(
                "transcriber.__main__.ensure_ffmpeg_available_for_child_processes",
                return_value="ffmpeg",
            ),
        ):
            try:
                transcriber_main.run_whisperx_direct(
                    make_cfg(language="en", diarize=False, warm_vram=True),
                    Path("input.wav"),
                    Path(tmpdir) / "output.srt",
                    hf_token=None,
                    diarize=False,
                )
            finally:
                transcriber_main.clear_warm_model_caches()

        self.assertGreaterEqual(lock.enter_count, 3)

    def test_cold_vram_loads_asr_and_align_models_per_run(self) -> None:
        fake_whisperx = SimpleNamespace(
            __name__="whisperx",
            load_model=MagicMock(),
            load_audio=MagicMock(return_value=object()),
            load_align_model=MagicMock(return_value=(object(), {"language": "en"})),
            align=MagicMock(
                side_effect=lambda segments, *args, **kwargs: {
                    "language": "en",
                    "segments": segments,
                }
            ),
        )
        fake_whisperx.load_model.side_effect = [
            MagicMock(
                transcribe=MagicMock(
                    return_value={
                        "language": "en",
                        "segments": [
                            {"words": [{"word": "Hello", "start": 0.0, "end": 0.5}]}
                        ],
                    }
                )
            ),
            MagicMock(
                transcribe=MagicMock(
                    return_value={
                        "language": "en",
                        "segments": [
                            {"words": [{"word": "Again", "start": 0.0, "end": 0.5}]}
                        ],
                    }
                )
            ),
        ]

        with (
            TemporaryDirectory() as tmpdir,
            patch.dict(sys.modules, {"whisperx": fake_whisperx}),
            patch(
                "transcriber.__main__.ensure_ffmpeg_available_for_child_processes",
                return_value="ffmpeg",
            ),
        ):
            cfg = make_cfg(language="en", diarize=False, warm_vram=False)
            for idx in range(2):
                transcriber_main.run_whisperx_direct(
                    cfg,
                    Path("input.wav"),
                    Path(tmpdir) / f"output-{idx}.srt",
                    hf_token=None,
                    diarize=False,
                )

        self.assertEqual(fake_whisperx.load_model.call_count, 2)
        self.assertEqual(fake_whisperx.load_align_model.call_count, 2)

    def test_warm_vram_reuses_diarization_pipeline_without_raw_token_cache_key(
        self,
    ) -> None:
        model = MagicMock()
        model.transcribe.return_value = {
            "language": "en",
            "segments": [{"words": [{"word": "Hello", "start": 0.0, "end": 0.5}]}],
        }
        diarize_model = MagicMock(return_value="diarize-segments")
        raw_token = "fake"
        fake_whisperx = SimpleNamespace(
            __name__="whisperx",
            load_model=MagicMock(return_value=model),
            load_audio=MagicMock(return_value=object()),
            load_align_model=MagicMock(return_value=(object(), {"language": "en"})),
            align=MagicMock(
                side_effect=lambda segments, *args, **kwargs: {
                    "language": "en",
                    "segments": segments,
                }
            ),
            DiarizationPipeline=MagicMock(return_value=diarize_model),
            assign_word_speakers=MagicMock(
                side_effect=lambda diarize_segments, result: result
            ),
        )
        fake_torch = SimpleNamespace(load=MagicMock())

        with (
            TemporaryDirectory() as tmpdir,
            patch.dict(sys.modules, {"whisperx": fake_whisperx, "torch": fake_torch}),
            patch(
                "transcriber.__main__.ensure_ffmpeg_available_for_child_processes",
                return_value="ffmpeg",
            ),
        ):
            cfg = make_cfg(language="en", diarize=True, warm_vram=True)
            try:
                for idx in range(2):
                    transcriber_main.run_whisperx_direct(
                        cfg,
                        Path("input.wav"),
                        Path(tmpdir) / f"diarized-{idx}.srt",
                        hf_token=raw_token,
                        diarize=True,
                    )
            finally:
                cache_keys = list(transcriber_main._WARM_DIARIZATION_MODEL_CACHE)
                transcriber_main.clear_warm_model_caches()

        self.assertEqual(fake_whisperx.DiarizationPipeline.call_count, 1)
        self.assertEqual(diarize_model.call_count, 2)
        self.assertNotIn(raw_token, repr(cache_keys))

    def test_cold_vram_loads_diarization_pipeline_per_run(self) -> None:
        model = MagicMock()
        model.transcribe.return_value = {
            "language": "en",
            "segments": [{"words": [{"word": "Hello", "start": 0.0, "end": 0.5}]}],
        }
        fake_whisperx = SimpleNamespace(
            __name__="whisperx",
            load_model=MagicMock(return_value=model),
            load_audio=MagicMock(return_value=object()),
            load_align_model=MagicMock(return_value=(object(), {"language": "en"})),
            align=MagicMock(
                side_effect=lambda segments, *args, **kwargs: {
                    "language": "en",
                    "segments": segments,
                }
            ),
            DiarizationPipeline=MagicMock(
                side_effect=[
                    MagicMock(return_value="one"),
                    MagicMock(return_value="two"),
                ]
            ),
            assign_word_speakers=MagicMock(
                side_effect=lambda diarize_segments, result: result
            ),
        )
        fake_torch = SimpleNamespace(load=MagicMock())

        with (
            TemporaryDirectory() as tmpdir,
            patch.dict(sys.modules, {"whisperx": fake_whisperx, "torch": fake_torch}),
            patch(
                "transcriber.__main__.ensure_ffmpeg_available_for_child_processes",
                return_value="ffmpeg",
            ),
        ):
            cfg = make_cfg(language="en", diarize=True, warm_vram=False)
            for idx in range(2):
                transcriber_main.run_whisperx_direct(
                    cfg,
                    Path("input.wav"),
                    Path(tmpdir) / f"diarized-{idx}.srt",
                    hf_token="fake",
                    diarize=True,
                )

        self.assertEqual(fake_whisperx.DiarizationPipeline.call_count, 2)

    @patch("transcriber.__main__.load_hf_token", return_value="hf_token")
    @patch(
        "transcriber.__main__.preprocess_audio_for_whisperx",
        side_effect=lambda path, temp_dir, report=print: path,
    )
    def test_known_diarization_failure_does_not_rerun_transcription(
        self,
        preprocess_audio: MagicMock,
        load_token: MagicMock,
    ) -> None:
        model = MagicMock()
        model.transcribe.return_value = {
            "language": "en",
            "segments": [{"words": [{"word": "Hello", "start": 0.0, "end": 0.5}]}],
        }
        fake_whisperx = SimpleNamespace(
            __name__="whisperx",
            load_model=MagicMock(return_value=model),
            load_audio=MagicMock(return_value=object()),
            load_align_model=MagicMock(return_value=(object(), {"language": "en"})),
            align=MagicMock(
                side_effect=lambda segments, *args, **kwargs: {
                    "language": "en",
                    "segments": segments,
                }
            ),
            DiarizationPipeline=MagicMock(
                side_effect=RuntimeError(
                    "Could not download 'pyannote/speaker-diarization-3.1' pipeline."
                )
            ),
        )
        fake_torch = SimpleNamespace(load=MagicMock())

        with (
            TemporaryDirectory() as tmpdir,
            patch.dict(sys.modules, {"whisperx": fake_whisperx, "torch": fake_torch}),
            patch(
                "transcriber.__main__.ensure_ffmpeg_available_for_child_processes",
                return_value="ffmpeg",
            ),
        ):
            source = Path(tmpdir) / "meeting.wav"
            source.write_bytes(b"audio")
            reports: list[str] = []

            rc = transcribe_file(
                make_cfg(language="en", diarize=True), source, report=reports.append
            )

            srt_path = source.with_suffix(".srt")
            self.assertEqual(rc, 0)
            self.assertTrue(srt_path.exists())
            self.assertIn("Hello", srt_path.read_text(encoding="utf-8"))
            self.assertEqual(model.transcribe.call_count, 1)
            self.assertTrue(
                any("completed without speaker diarization" in line for line in reports)
            )
