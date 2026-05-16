from __future__ import annotations

import asyncio
import contextlib
import json
import queue
import shutil
import subprocess
import sys
import sysconfig
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class WhisperLiveKitProtocolError(RuntimeError):
    pass


class LiveTranslationMode(str, Enum):
    DIRECT = "direct"
    CASCADE = "cascade"


@dataclass(frozen=True)
class CaptionPair:
    source_text: str
    translated_text: str


@dataclass(frozen=True)
class CaptionState:
    committed_lines: tuple[str, ...]
    partial_line: str
    lag_seconds: float | None = None
    committed_pairs: tuple[CaptionPair, ...] = ()


def resolve_wlk_executable() -> str:
    script_paths: list[str | None] = []
    for raw_path in (sysconfig.get_path("scripts"), str(Path(sys.executable).parent), None):
        if raw_path not in script_paths:
            script_paths.append(raw_path)

    for script_path in script_paths:
        for executable in ("wlk", "whisperlivekit-server"):
            resolved = shutil.which(executable, path=script_path)
            if resolved:
                return resolved
    raise RuntimeError(
        'Could not find WhisperLiveKit. Install live extras with: uv sync --extra live (or pip install -e ".[live]").'
    )


def build_wlk_command(
    executable: str,
    *,
    host: str,
    port: int,
    model: str,
    language: str,
    translation_mode: LiveTranslationMode,
    asr_prompt: str | None = None,
    static_prompt: str | None = None,
    backend: str = "auto",
    backend_policy: str = "localagreement",
    frame_threshold: int = 25,
    beams: int = 1,
    decoder: str = "auto",
    audio_min_len: float = 0.0,
    audio_max_len: float = 30.0,
    nllb_backend: str = "transformers",
    nllb_size: str = "600M",
) -> list[str]:
    command = [
        executable,
        "--host",
        host,
        "--port",
        str(port),
        "--model",
        model,
        "--language",
        language,
        "--backend-policy",
        backend_policy,
        "--pcm-input",
    ]
    if backend != "auto":
        command.extend(["--backend", backend])
    if translation_mode == LiveTranslationMode.DIRECT:
        command.append("--direct-english-translation")
    else:
        command.extend(["--target-language", "en"])
    if frame_threshold != 25:
        command.extend(["--frame-threshold", str(frame_threshold)])
    if beams != 1:
        command.extend(["--beams", str(beams)])
    if decoder != "auto":
        command.extend(["--decoder", decoder])
    if audio_min_len != 0.0:
        command.extend(["--audio-min-len", str(audio_min_len)])
    if audio_max_len != 30.0:
        command.extend(["--audio-max-len", str(audio_max_len)])
    if nllb_backend != "transformers":
        command.extend(["--nllb-backend", nllb_backend])
    if nllb_size != "600M":
        command.extend(["--nllb-size", nllb_size])
    if asr_prompt:
        command.extend(["--init-prompt", asr_prompt])
    if static_prompt:
        command.extend(["--static-init-prompt", static_prompt])
    return command


def build_asr_url(*, host: str, port: int, language: str) -> str:
    return f"ws://{host}:{port}/asr?language={language}&mode=full"


def wait_for_wlk_health(*, host: str, port: int, process: subprocess.Popen[Any], timeout_seconds: float = 60.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"http://{host}:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("WhisperLiveKit exited before it became ready.")
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("ready") is True or payload.get("status") == "ok":
                return
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(0.25)
    raise RuntimeError(f"WhisperLiveKit did not become ready at {url}.")


def start_wlk_server(
    *,
    host: str,
    port: int,
    model: str,
    language: str,
    translation_mode: LiveTranslationMode,
    asr_prompt: str | None,
    static_prompt: str | None = None,
    backend: str = "auto",
    backend_policy: str = "localagreement",
    frame_threshold: int = 25,
    beams: int = 1,
    decoder: str = "auto",
    audio_min_len: float = 0.0,
    audio_max_len: float = 30.0,
    nllb_backend: str = "transformers",
    nllb_size: str = "600M",
) -> subprocess.Popen[Any]:
    executable = resolve_wlk_executable()
    command = build_wlk_command(
        executable,
        host=host,
        port=port,
        model=model,
        language=language,
        translation_mode=translation_mode,
        asr_prompt=asr_prompt,
        static_prompt=static_prompt,
        backend=backend,
        backend_policy=backend_policy,
        frame_threshold=frame_threshold,
        beams=beams,
        decoder=decoder,
        audio_min_len=audio_min_len,
        audio_max_len=audio_max_len,
        nllb_backend=nllb_backend,
        nllb_size=nllb_size,
    )
    process = subprocess.Popen(command)
    try:
        wait_for_wlk_health(host=host, port=port, process=process)
    except Exception:
        stop_wlk_server(process)
        raise
    return process


def stop_wlk_server(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def validate_config_message(message: Mapping[str, Any]) -> None:
    if message.get("type") != "config":
        raise WhisperLiveKitProtocolError("Expected WhisperLiveKit config message.")
    if message.get("useAudioWorklet") is not True:
        raise WhisperLiveKitProtocolError("WhisperLiveKit must be started with --pcm-input for raw PCM streaming.")


def is_ready_to_stop(message: Mapping[str, Any]) -> bool:
    return message.get("type") == "ready_to_stop"


def caption_state_from_full_update(
    message: Mapping[str, Any],
    *,
    translation_mode: LiveTranslationMode,
) -> CaptionState:
    committed_pairs: list[CaptionPair] = []
    for raw_line in message.get("lines") or []:
        if not isinstance(raw_line, Mapping):
            continue
        if raw_line.get("speaker") == -2:
            continue
        if translation_mode == LiveTranslationMode.DIRECT:
            source_text = ""
            translated_text = str(raw_line.get("translation") or raw_line.get("text") or "").strip()
        else:
            source_text = str(raw_line.get("text") or "").strip()
            translated_text = str(raw_line.get("translation") or "").strip()
        if translated_text or source_text:
            committed_pairs.append(CaptionPair(source_text=source_text, translated_text=translated_text))

    partial = str(message.get("buffer_translation") or message.get("buffer_transcription") or "").strip()
    lag = message.get("remaining_time_transcription")
    lag_seconds = float(lag) if isinstance(lag, int | float) else None
    return CaptionState(
        committed_lines=tuple(pair.translated_text or pair.source_text for pair in committed_pairs),
        partial_line=partial,
        lag_seconds=lag_seconds,
        committed_pairs=tuple(committed_pairs),
    )


def _decode_json_message(raw_message: str | bytes) -> Mapping[str, Any]:
    text = raw_message.decode("utf-8") if isinstance(raw_message, bytes) else raw_message
    message = json.loads(text)
    if not isinstance(message, Mapping):
        raise WhisperLiveKitProtocolError("Expected WhisperLiveKit JSON object message.")
    return message


def _get_audio_chunk(audio_queue: queue.Queue[bytes], timeout: float) -> bytes | None:
    try:
        return audio_queue.get(timeout=timeout)
    except queue.Empty:
        return None


async def stream_pcm_queue(
    *,
    host: str,
    port: int,
    language: str,
    translation_mode: LiveTranslationMode,
    audio_queue: queue.Queue[bytes],
    stop_event: Any,
    on_state: Callable[[CaptionState], None],
) -> None:
    from websockets.asyncio.client import connect

    ready_to_stop = asyncio.Event()
    url = build_asr_url(host=host, port=port, language=language)
    async with connect(url) as websocket:
        raw_config = await websocket.recv()
        validate_config_message(_decode_json_message(raw_config))

        async def receive_updates() -> None:
            while True:
                raw_message = await websocket.recv()
                message = _decode_json_message(raw_message)
                if is_ready_to_stop(message):
                    ready_to_stop.set()
                    return
                if message.get("type") == "config":
                    validate_config_message(message)
                    continue
                if "lines" in message:
                    on_state(caption_state_from_full_update(message, translation_mode=translation_mode))

        receiver = asyncio.create_task(receive_updates())
        try:
            while not stop_event.is_set() and not receiver.done():
                chunk = await asyncio.to_thread(_get_audio_chunk, audio_queue, 0.1)
                if chunk:
                    await websocket.send(chunk)
            if receiver.done():
                await receiver
                return
            await websocket.send(b"")
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(ready_to_stop.wait(), timeout=10.0)
            if receiver.done():
                await receiver
        finally:
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver
