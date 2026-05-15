from __future__ import annotations

import contextlib
import importlib
import platform
import struct
import wave
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LIVE_SAMPLE_RATE = 16_000
PCM_SAMPLE_WIDTH_BYTES = 2


@dataclass(frozen=True)
class LoopbackDevice:
    index: int
    name: str
    sample_rate: int
    channels: int


def _ensure_windows() -> None:
    if platform.system() != "Windows":
        raise RuntimeError("Live system-audio capture requires Windows WASAPI loopback.")


def _load_pyaudiowpatch() -> Any:
    try:
        return importlib.import_module("pyaudiowpatch")
    except ImportError as exc:
        raise RuntimeError('Install live extras with: pip install -e ".[live]"') from exc


def _device_from_info(info: Mapping[str, Any]) -> LoopbackDevice:
    channels = int(info.get("maxInputChannels") or info.get("maxOutputChannels") or 1)
    return LoopbackDevice(
        index=int(info.get("index", -1)),
        name=str(info.get("name") or "WASAPI loopback device"),
        sample_rate=int(float(info.get("defaultSampleRate") or LIVE_SAMPLE_RATE)),
        channels=max(1, channels),
    )


def _clip_int16(value: float) -> int:
    return max(-32_768, min(32_767, int(round(value))))


def convert_to_pcm16_mono_16k(
    raw: bytes,
    *,
    input_sample_rate: int,
    input_channels: int,
) -> bytes:
    if input_sample_rate <= 0:
        raise ValueError("input_sample_rate must be greater than 0")
    if input_channels <= 0:
        raise ValueError("input_channels must be greater than 0")

    frame_width = input_channels * PCM_SAMPLE_WIDTH_BYTES
    if len(raw) % frame_width != 0:
        raise ValueError("raw PCM length is not aligned to the input channel count")
    if not raw:
        return b""

    values = struct.unpack(f"<{len(raw) // PCM_SAMPLE_WIDTH_BYTES}h", raw)
    input_frames = len(values) // input_channels
    mono = [
        sum(values[offset : offset + input_channels]) / input_channels
        for offset in range(0, len(values), input_channels)
    ]

    if input_sample_rate == LIVE_SAMPLE_RATE:
        return struct.pack(f"<{len(mono)}h", *(_clip_int16(sample) for sample in mono))

    output_frames = int(round(input_frames * LIVE_SAMPLE_RATE / input_sample_rate))
    if output_frames <= 0:
        return b""

    ratio = input_sample_rate / LIVE_SAMPLE_RATE
    output: list[int] = []
    for output_index in range(output_frames):
        position = output_index * ratio
        left_index = min(int(position), input_frames - 1)
        right_index = min(left_index + 1, input_frames - 1)
        fraction = position - left_index
        sample = mono[left_index] * (1.0 - fraction) + mono[right_index] * fraction
        output.append(_clip_int16(sample))

    return struct.pack(f"<{len(output)}h", *output)


def list_loopback_devices() -> list[LoopbackDevice]:
    _ensure_windows()
    pyaudio = _load_pyaudiowpatch()
    with pyaudio.PyAudio() as audio:
        return [_device_from_info(info) for info in audio.get_loopback_device_info_generator()]


def _select_loopback_info(audio: Any, device_index: int | None) -> Mapping[str, Any]:
    if device_index is not None:
        if hasattr(audio, "get_wasapi_loopback_analogue_by_index"):
            return audio.get_wasapi_loopback_analogue_by_index(device_index)
        for info in audio.get_loopback_device_info_generator():
            if int(info.get("index", -1)) == device_index:
                return info
        raise RuntimeError(f"Could not find WASAPI loopback device index {device_index}.")

    if hasattr(audio, "get_default_wasapi_loopback"):
        return audio.get_default_wasapi_loopback()

    devices = list(audio.get_loopback_device_info_generator())
    if not devices:
        raise RuntimeError("Could not find a WASAPI loopback device.")
    return devices[0]


def select_loopback_device(device_index: int | None = None) -> LoopbackDevice:
    _ensure_windows()
    pyaudio = _load_pyaudiowpatch()
    with pyaudio.PyAudio() as audio:
        return _device_from_info(_select_loopback_info(audio, device_index))


def iter_loopback_pcm_chunks(device_index: int | None = None, chunk_ms: int = 500) -> Iterator[bytes]:
    _ensure_windows()
    pyaudio = _load_pyaudiowpatch()
    with pyaudio.PyAudio() as audio:
        info = _select_loopback_info(audio, device_index)
        device = _device_from_info(info)
        frames_per_buffer = max(1, int(device.sample_rate * max(20, chunk_ms) / 1000))
        stream = audio.open(
            format=getattr(pyaudio, "paInt16", 8),
            channels=device.channels,
            rate=device.sample_rate,
            input=True,
            input_device_index=device.index,
            frames_per_buffer=frames_per_buffer,
        )
        try:
            while True:
                try:
                    raw = stream.read(frames_per_buffer, exception_on_overflow=False)
                except TypeError:
                    raw = stream.read(frames_per_buffer)
                yield convert_to_pcm16_mono_16k(
                    raw,
                    input_sample_rate=device.sample_rate,
                    input_channels=device.channels,
                )
        finally:
            with contextlib.suppress(Exception):
                stream.stop_stream()
            with contextlib.suppress(Exception):
                stream.close()


def write_loopback_test_wav(
    output_path: Path,
    *,
    seconds: float,
    device_index: int | None,
    chunk_ms: int,
) -> None:
    target_bytes = max(0, int(seconds * LIVE_SAMPLE_RATE) * PCM_SAMPLE_WIDTH_BYTES)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(PCM_SAMPLE_WIDTH_BYTES)
        wav.setframerate(LIVE_SAMPLE_RATE)
        remaining = target_bytes
        for chunk in iter_loopback_pcm_chunks(device_index=device_index, chunk_ms=chunk_ms):
            if remaining <= 0:
                break
            data = chunk[:remaining]
            wav.writeframes(data)
            remaining -= len(data)
