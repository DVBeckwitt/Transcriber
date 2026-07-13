from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import transcriber.__main__ as transcriber_main
from tests.support import make_cfg
from transcriber.__main__ import (
    RunConfig,
    TimedToken,
    apply_confidence_cleanup,
    build_llm_file,
    build_srt_cues_from_result,
    render_low_confidence_markup,
    smooth_timed_tokens,
    transcribe_file,
    write_direct_srt_from_result,
)


class SubtitleTests(unittest.TestCase):
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
                        {
                            "word": "Hello",
                            "start": 0.0,
                            "end": 0.5,
                            "speaker": "SPEAKER_00",
                        },
                        {
                            "word": "there.",
                            "start": 0.5,
                            "end": 1.0,
                            "speaker": "SPEAKER_00",
                        },
                        {
                            "word": "Come",
                            "start": 1.0,
                            "end": 1.5,
                            "speaker": "SPEAKER_01",
                        },
                        {
                            "word": "in.",
                            "start": 1.5,
                            "end": 2.0,
                            "speaker": "SPEAKER_01",
                        },
                    ]
                }
            ]
        }

        cues = build_srt_cues_from_result(result, make_cfg(diarize_smoothing=False))

        self.assertEqual(
            [cue.text for cue in cues],
            ["SPEAKER_00: Hello there.", "SPEAKER_01: Come in."],
        )

    def test_timed_srt_hides_speaker_labels_without_losing_speaker_splits(self) -> None:
        result = {
            "segments": [
                {
                    "words": [
                        {
                            "word": "Hello",
                            "start": 0.0,
                            "end": 0.5,
                            "speaker": "SPEAKER_00",
                        },
                        {
                            "word": "there.",
                            "start": 0.5,
                            "end": 1.0,
                            "speaker": "SPEAKER_00",
                        },
                        {
                            "word": "Come",
                            "start": 1.0,
                            "end": 1.5,
                            "speaker": "SPEAKER_01",
                        },
                        {
                            "word": "in.",
                            "start": 1.5,
                            "end": 2.0,
                            "speaker": "SPEAKER_01",
                        },
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

        cues = build_srt_cues_from_result(
            result, make_cfg(include_speaker_labels=False)
        )

        self.assertEqual([cue.text for cue in cues], ["Hello there."])

    def test_llm_file_uses_no_label_srt_body(self) -> None:
        result = {
            "segments": [
                {
                    "words": [
                        {
                            "word": "Hello",
                            "start": 0.0,
                            "end": 0.5,
                            "speaker": "SPEAKER_00",
                        },
                        {
                            "word": "there.",
                            "start": 0.5,
                            "end": 1.0,
                            "speaker": "SPEAKER_00",
                        },
                    ]
                }
            ]
        }
        with TemporaryDirectory() as tmpdir:
            srt_path = Path(tmpdir) / "movie.srt"
            llm_path = Path(tmpdir) / "movie_llm.txt"

            write_direct_srt_from_result(
                result, srt_path, make_cfg(include_speaker_labels=False)
            )
            build_llm_file(srt_path, llm_path)

            transcript_body = llm_path.read_text(encoding="utf-8").split(
                "TRANSCRIPT:\n", 1
            )[1]
            self.assertEqual(transcript_body, "Hello there.")
            self.assertNotIn("SPEAKER_", transcript_body)

    def test_finalize_transcript_outputs_updates_srt_and_llm_together(self) -> None:
        with TemporaryDirectory() as tmpdir:
            srt_path = Path(tmpdir) / "movie.srt"
            llm_path = Path(tmpdir) / "movie_llm.txt"
            srt_path.write_text(
                (
                    "1\n"
                    "00:00:00,000 --> 00:00:01,000\n"
                    "Hello.\n\n"
                    "2\n"
                    "00:00:01,000 --> 00:00:02,000\n"
                    "__LOWCONF_65__unclear__LOWCONF_END__\n"
                ),
                encoding="utf-8",
            )

            transcriber_main.finalize_transcript_outputs(srt_path, llm_path)

            self.assertNotIn("__LOWCONF", srt_path.read_text(encoding="utf-8"))
            transcript_body = llm_path.read_text(encoding="utf-8").split(
                "TRANSCRIPT:\n", 1
            )[1]
            self.assertEqual(transcript_body, "Hello.\n—")

    @patch(
        "transcriber.__main__.preprocess_audio_for_whisperx",
        side_effect=lambda path, temp_dir, report=print: path,
    )
    @patch("transcriber.__main__.run_whisperx_direct_logged")
    def test_transcribe_file_finalizes_outputs_and_rejects_missing_srt(
        self,
        run_logged: MagicMock,
        preprocess_audio: MagicMock,
    ) -> None:
        def run_and_write_srt(
            cfg: RunConfig,
            input_path: Path,
            srt_path: Path,
            hf_token: str | None,
            diarize: bool,
            log_path: Path,
            append: bool = False,
        ) -> tuple[int, str | None]:
            srt_path.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nHello.\n", encoding="utf-8"
            )
            return 0, "en"

        run_logged.side_effect = run_and_write_srt

        with (
            TemporaryDirectory() as tmpdir,
            patch("transcriber.__main__.finalize_transcript_outputs") as finalize,
        ):
            source = Path(tmpdir) / "meeting.wav"
            source.write_bytes(b"audio")

            rc = transcribe_file(
                make_cfg(language="en", diarize=False),
                source,
                report=lambda _message: None,
            )

            run_logged.side_effect = None
            run_logged.return_value = (0, "en")
            missing = Path(tmpdir) / "missing.wav"
            missing.write_bytes(b"audio")
            missing_rc = transcribe_file(
                make_cfg(language="en", diarize=False),
                missing,
                report=lambda _message: None,
            )

        self.assertEqual(rc, 0)
        self.assertEqual(missing_rc, 1)
        finalize.assert_called_once()

    def test_write_direct_srt_allows_empty_transcript_when_no_cues(self) -> None:
        with TemporaryDirectory() as tmpdir:
            srt_path = Path(tmpdir) / "silent.srt"

            write_direct_srt_from_result({"segments": []}, srt_path, make_cfg())

            self.assertEqual(srt_path.read_text(encoding="utf-8"), "")

    def test_speaker_smoothing_merges_short_blips(self) -> None:
        tokens = [
            TimedToken(text="Hello", start_ms=0, end_ms=300, speaker="SPEAKER_00"),
            TimedToken(
                text="yes",
                start_ms=300,
                end_ms=500,
                speaker="SPEAKER_01",
                confidence=0.2,
            ),
            TimedToken(text="there", start_ms=500, end_ms=1100, speaker="SPEAKER_00"),
        ]

        smoothed = smooth_timed_tokens(tokens)

        self.assertEqual(
            [token.speaker for token in smoothed],
            ["SPEAKER_00", "SPEAKER_00", "SPEAKER_00"],
        )
