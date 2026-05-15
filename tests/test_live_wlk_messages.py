from __future__ import annotations

import unittest
from unittest.mock import patch

from transcriber.live_wlk import (
    WhisperLiveKitProtocolError,
    _decode_json_message,
    build_wlk_command,
    caption_state_from_full_update,
    is_ready_to_stop,
    resolve_wlk_executable,
    start_wlk_server,
    validate_config_message,
)


class WhisperLiveKitMessageTests(unittest.TestCase):
    def test_parse_config_message_with_pcm_audio_worklet_enabled(self) -> None:
        validate_config_message({"type": "config", "useAudioWorklet": True, "mode": "full"})

    def test_parse_config_message_rejects_encoded_audio_mode(self) -> None:
        with self.assertRaisesRegex(WhisperLiveKitProtocolError, "raw PCM"):
            validate_config_message({"type": "config", "useAudioWorklet": False, "mode": "full"})

    def test_full_update_prefers_translation_and_ignores_silence_lines(self) -> None:
        state = caption_state_from_full_update(
            {
                "status": "active_transcription",
                "lines": [
                    {"speaker": 1, "text": "hola", "translation": "hello"},
                    {"speaker": -2, "text": None},
                    {"speaker": 1, "text": "mundo"},
                ],
                "buffer_translation": "how are",
                "buffer_transcription": "como estan",
                "remaining_time_transcription": 1.25,
            }
        )

        self.assertEqual(state.committed_lines, ("hello", "mundo"))
        self.assertEqual(state.partial_line, "how are")
        self.assertEqual(state.lag_seconds, 1.25)

    def test_full_update_falls_back_to_buffer_transcription(self) -> None:
        state = caption_state_from_full_update(
            {
                "status": "active_transcription",
                "lines": [],
                "buffer_translation": "",
                "buffer_transcription": "partial source",
            }
        )

        self.assertEqual(state.partial_line, "partial source")

    def test_partial_replaces_previous_update_instead_of_appending(self) -> None:
        first = caption_state_from_full_update({"lines": [], "buffer_translation": "hello wor"})
        second = caption_state_from_full_update({"lines": [], "buffer_translation": "hello world"})

        self.assertEqual(first.partial_line, "hello wor")
        self.assertEqual(second.partial_line, "hello world")

    def test_ready_to_stop_is_recognized(self) -> None:
        self.assertTrue(is_ready_to_stop({"type": "ready_to_stop"}))
        self.assertFalse(is_ready_to_stop({"status": "active_transcription"}))

    def test_json_message_decode_rejects_non_object_payload(self) -> None:
        with self.assertRaisesRegex(WhisperLiveKitProtocolError, "JSON object"):
            _decode_json_message("[]")

    def test_resolve_wlk_executable_prefers_wlk(self) -> None:
        with patch("transcriber.live_wlk.shutil.which", side_effect=["C:/bin/wlk.exe", "C:/bin/alt.exe"]):
            self.assertEqual(resolve_wlk_executable(), "C:/bin/wlk.exe")

    def test_start_wlk_server_terminates_process_when_health_check_fails(self) -> None:
        class FakeProcess:
            terminated = False

            def poll(self) -> None:
                return None

            def terminate(self) -> None:
                self.terminated = True

            def wait(self, timeout: float) -> None:
                return None

        process = FakeProcess()

        with (
            patch("transcriber.live_wlk.resolve_wlk_executable", return_value="wlk"),
            patch("transcriber.live_wlk.subprocess.Popen", return_value=process),
            patch("transcriber.live_wlk.wait_for_wlk_health", side_effect=RuntimeError("not ready")),
        ):
            with self.assertRaisesRegex(RuntimeError, "not ready"):
                start_wlk_server(host="127.0.0.1", port=8123, model="small", language="es", asr_prompt=None)

        self.assertTrue(process.terminated)

    def test_build_wlk_command_uses_pcm_translation_without_diarization(self) -> None:
        command = build_wlk_command(
            "wlk",
            host="127.0.0.1",
            port=8123,
            model="small",
            language="es",
            asr_prompt="OpenAI Codex",
        )

        self.assertEqual(command[:2], ["wlk", "--host"])
        self.assertIn("--pcm-input", command)
        self.assertIn("--direct-english-translation", command)
        self.assertIn("--backend-policy", command)
        self.assertIn("simulstreaming", command)
        self.assertIn("--init-prompt", command)
        self.assertNotIn("--diarization", command)


if __name__ == "__main__":
    unittest.main()
