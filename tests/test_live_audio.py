from __future__ import annotations

import math
import struct
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from transcriber.live_audio import (
    LiveAudioDiagnostics,
    LoopbackDevice,
    _audio_diagnostics,
    convert_to_pcm16_mono_16k,
    list_loopback_devices,
    select_loopback_device,
    write_loopback_test_wav,
)


class LiveAudioConversionTests(unittest.TestCase):
    def stereo_48k_sine(self, *, seconds: float, amplitude: int = 12000) -> bytes:
        frames = int(48_000 * seconds)
        samples: list[int] = []
        for frame in range(frames):
            value = int(amplitude * math.sin(2.0 * math.pi * 440.0 * frame / 48_000))
            samples.extend([value, -value])
        return struct.pack(f"<{len(samples)}h", *samples)

    def decode_int16(self, data: bytes) -> tuple[int, ...]:
        return struct.unpack(f"<{len(data) // 2}h", data)

    def test_stereo_48k_half_second_converts_to_mono_16k_s16le(self) -> None:
        raw = self.stereo_48k_sine(seconds=0.5)

        pcm = convert_to_pcm16_mono_16k(raw, input_sample_rate=48_000, input_channels=2)

        self.assertEqual(len(pcm), 16_000)
        self.assertEqual(len(self.decode_int16(pcm)), 8_000)

    def test_conversion_downmixes_stereo_without_changing_duration(self) -> None:
        frames = 48_000 // 2
        raw = struct.pack(f"<{frames * 2}h", *([12_000, 6_000] * frames))

        pcm = convert_to_pcm16_mono_16k(raw, input_sample_rate=48_000, input_channels=2)
        samples = self.decode_int16(pcm)

        self.assertEqual(len(samples), 8_000)
        self.assertTrue(all(sample == 9_000 for sample in samples[:100]))

    def test_conversion_uses_stronger_channel_when_stereo_phase_cancels(self) -> None:
        frames = 48_000 // 2
        raw = struct.pack(f"<{frames * 2}h", *([12_000, -12_000] * frames))

        pcm = convert_to_pcm16_mono_16k(raw, input_sample_rate=48_000, input_channels=2)
        samples = self.decode_int16(pcm)

        self.assertEqual(len(samples), 8_000)
        self.assertTrue(all(sample == 12_000 for sample in samples[:100]))

    def test_conversion_clips_to_int16_range(self) -> None:
        frames = 48_000 // 2
        raw = struct.pack(f"<{frames}h", *([32_767] * frames))

        pcm = convert_to_pcm16_mono_16k(raw, input_sample_rate=48_000, input_channels=1)
        samples = self.decode_int16(pcm)

        self.assertTrue(all(-32_768 <= sample <= 32_767 for sample in samples))
        self.assertEqual(max(samples), 32_767)

    def test_audio_diagnostics_reports_rms_and_peak(self) -> None:
        raw = struct.pack("<4h", -10_000, 0, 10_000, 20_000)

        diagnostics = _audio_diagnostics(raw, input_sample_rate=48_000, input_channels=1, output_bytes=8)

        self.assertEqual(
            diagnostics,
            LiveAudioDiagnostics(
                input_sample_rate=48_000,
                input_channels=1,
                output_bytes=8,
                rms_level=12247.45,
                peak_level=20000,
            ),
        )


class LiveAudioLoopbackTests(unittest.TestCase):
    def fake_pyaudio_module(self) -> types.SimpleNamespace:
        class FakePyAudio:
            def __enter__(self) -> FakePyAudio:
                return self

            def __exit__(self, *_exc: object) -> None:
                return None

            def get_loopback_device_info_generator(self) -> list[dict[str, object]]:
                return [
                    {
                        "index": 7,
                        "name": "Speakers (loopback)",
                        "defaultSampleRate": 48_000.0,
                        "maxInputChannels": 2,
                    }
                ]

            def get_default_wasapi_loopback(self) -> dict[str, object]:
                return {
                    "index": 9,
                    "name": "Default Speakers (loopback)",
                    "defaultSampleRate": 48_000.0,
                    "maxInputChannels": 2,
                }

            def get_wasapi_loopback_analogue_by_index(self, index: int) -> dict[str, object]:
                return {
                    "index": index,
                    "name": f"Loopback {index}",
                    "defaultSampleRate": 44_100.0,
                    "maxInputChannels": 1,
                }

        return types.SimpleNamespace(PyAudio=FakePyAudio)

    def test_list_loopback_devices_requires_windows(self) -> None:
        with patch("transcriber.live_audio.platform.system", return_value="Linux"):
            with self.assertRaisesRegex(RuntimeError, "Windows"):
                list_loopback_devices()

    def test_list_loopback_devices_imports_pyaudiowpatch_lazily(self) -> None:
        fake_module = self.fake_pyaudio_module()
        with (
            patch("transcriber.live_audio.platform.system", return_value="Windows"),
            patch.dict(sys.modules, {"pyaudiowpatch": fake_module}),
        ):
            devices = list_loopback_devices()

        self.assertEqual(devices, [LoopbackDevice(index=7, name="Speakers (loopback)", sample_rate=48_000, channels=2)])

    def test_select_loopback_device_uses_default_or_explicit_index(self) -> None:
        fake_module = self.fake_pyaudio_module()
        with (
            patch("transcriber.live_audio.platform.system", return_value="Windows"),
            patch.dict(sys.modules, {"pyaudiowpatch": fake_module}),
        ):
            default = select_loopback_device()
            explicit = select_loopback_device(12)

        self.assertEqual(default.index, 9)
        self.assertEqual(explicit.index, 12)
        self.assertEqual(explicit.sample_rate, 44_100)

    def test_loopback_test_writes_pcm_wav(self) -> None:
        chunk = b"\x01\x00" * 16_000
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "loopback.wav"
            with patch("transcriber.live_audio.iter_loopback_pcm_chunks", return_value=iter([chunk])):
                write_loopback_test_wav(output, seconds=1.0, device_index=None, chunk_ms=500)

            data = output.read_bytes()

        self.assertTrue(data.startswith(b"RIFF"))
        self.assertIn(b"WAVE", data[:16])
        self.assertGreater(len(data), len(chunk))


if __name__ == "__main__":
    unittest.main()
