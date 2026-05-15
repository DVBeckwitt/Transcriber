from __future__ import annotations

import argparse
import asyncio
import contextlib
import queue
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from transcriber.live_wlk import CaptionPair, CaptionState, LiveTranslationMode


@dataclass(frozen=True)
class LiveConfig:
    language: str
    model: str
    engine: str
    source: str
    device_index: int | None
    port: int
    chunk_ms: int
    preset: str
    translation_mode: LiveTranslationMode
    show_window: bool
    save_transcript_path: str | None
    save_bilingual_transcript_path: str | None
    speaker_labels: bool
    diarize: bool
    translate_to_english: bool
    asr_prompt: str | None


def build_live_asr_prompt(args: argparse.Namespace) -> str | None:
    from transcriber.__main__ import (
        build_asr_prompt,
        load_glossary_file,
        looks_like_glossary_file,
        parse_glossary_entries,
    )

    glossary_entries: list[str] = []
    glossary_files: list[Path] = []
    if args.glossary_file:
        glossary_files.append(Path(args.glossary_file).expanduser())
    for entry in args.glossary or []:
        if looks_like_glossary_file(entry):
            glossary_files.append(Path(entry).expanduser())
        else:
            glossary_entries.append(entry)

    file_entries: list[str] = []
    for glossary_file in glossary_files:
        file_entries.extend(load_glossary_file(glossary_file))
    glossary = parse_glossary_entries([*file_entries, *glossary_entries])
    return build_asr_prompt(
        glossary=glossary,
        prompt_text=args.asr_prompt,
        prompt_file=args.asr_prompt_file,
    )


def _effective_live_translation_mode(args: argparse.Namespace) -> LiveTranslationMode:
    if args.live_translation_mode:
        return LiveTranslationMode(args.live_translation_mode)
    if args.live_preset == "quality":
        return LiveTranslationMode.CASCADE
    return LiveTranslationMode.DIRECT


def _effective_live_chunk_ms(args: argparse.Namespace) -> int:
    if args.live_chunk_ms is not None:
        return max(20, int(args.live_chunk_ms))
    if args.live_preset == "latency":
        return 250
    return 500


def build_live_config(args: argparse.Namespace) -> LiveConfig:
    translation_mode = _effective_live_translation_mode(args)
    if translation_mode == LiveTranslationMode.DIRECT and args.live_save_bilingual_transcript:
        raise RuntimeError("--live-save-bilingual-transcript requires --live-translation-mode cascade.")

    return LiveConfig(
        language=args.lang or "es",
        model=args.model or "small",
        engine=args.live_engine,
        source=args.live_source,
        device_index=args.live_device_index,
        port=max(1, int(args.live_port)),
        chunk_ms=_effective_live_chunk_ms(args),
        preset=args.live_preset,
        translation_mode=translation_mode,
        show_window=not bool(args.live_no_window),
        save_transcript_path=args.live_save_transcript,
        save_bilingual_transcript_path=args.live_save_bilingual_transcript,
        speaker_labels=False,
        diarize=False,
        translate_to_english=True,
        asr_prompt=build_live_asr_prompt(args),
    )


def _put_latest_audio(audio_queue: queue.Queue[bytes], chunk: bytes) -> None:
    try:
        audio_queue.put(chunk, timeout=0.2)
    except queue.Full:
        with contextlib.suppress(queue.Empty):
            audio_queue.get_nowait()
        with contextlib.suppress(queue.Full):
            audio_queue.put_nowait(chunk)


def _capture_loop(
    *,
    config: LiveConfig,
    audio_queue: queue.Queue[bytes],
    stop_event: threading.Event,
    error_queue: queue.Queue[BaseException],
) -> None:
    from transcriber.live_audio import iter_loopback_pcm_chunks

    try:
        for chunk in iter_loopback_pcm_chunks(device_index=config.device_index, chunk_ms=config.chunk_ms):
            if stop_event.is_set():
                break
            _put_latest_audio(audio_queue, chunk)
    except BaseException as exc:
        error_queue.put(exc)
        stop_event.set()


def _write_committed_transcript(path: Path, state: CaptionState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(state.committed_lines)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _write_bilingual_transcript(path: Path, state: CaptionState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = [
        f"{index}.\nES: {pair.source_text}\nEN: {pair.translated_text}"
        for index, pair in enumerate(state.committed_pairs, start=1)
    ]
    text = "\n\n".join(blocks)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def _put_latest_state(state_queue: queue.Queue[CaptionState], state: CaptionState) -> None:
    try:
        state_queue.put_nowait(state)
    except queue.Full:
        with contextlib.suppress(queue.Empty):
            state_queue.get_nowait()
        with contextlib.suppress(queue.Full):
            state_queue.put_nowait(state)


def _state_handler(
    *,
    state_queue: queue.Queue[CaptionState] | None,
    save_transcript_path: str | None,
    save_bilingual_transcript_path: str | None,
) -> Callable[[CaptionState], None]:
    transcript_path = Path(save_transcript_path).expanduser() if save_transcript_path else None
    bilingual_transcript_path = (
        Path(save_bilingual_transcript_path).expanduser() if save_bilingual_transcript_path else None
    )
    last_written_lines: tuple[str, ...] | None = None
    last_written_pairs: tuple[CaptionPair, ...] | None = None

    def handle_state(state: CaptionState) -> None:
        nonlocal last_written_lines, last_written_pairs
        if state_queue is not None:
            _put_latest_state(state_queue, state)
        if transcript_path is not None and state.committed_lines != last_written_lines:
            _write_committed_transcript(transcript_path, state)
            last_written_lines = state.committed_lines
        if bilingual_transcript_path is not None and state.committed_pairs != last_written_pairs:
            _write_bilingual_transcript(bilingual_transcript_path, state)
            last_written_pairs = state.committed_pairs

    return handle_state


def _stream_loop(
    *,
    config: LiveConfig,
    audio_queue: queue.Queue[bytes],
    state_queue: queue.Queue[CaptionState] | None,
    stop_event: threading.Event,
    error_queue: queue.Queue[BaseException],
) -> None:
    from transcriber.live_wlk import stream_pcm_queue

    try:
        asyncio.run(
            stream_pcm_queue(
                host="127.0.0.1",
                port=config.port,
                language=config.language,
                translation_mode=config.translation_mode,
                audio_queue=audio_queue,
                stop_event=stop_event,
                on_state=_state_handler(
                    state_queue=state_queue,
                    save_transcript_path=config.save_transcript_path,
                    save_bilingual_transcript_path=config.save_bilingual_transcript_path,
                ),
            )
        )
    except BaseException as exc:
        error_queue.put(exc)
        stop_event.set()


def run_live_session(config: LiveConfig) -> int:
    from transcriber.live_wlk import start_wlk_server, stop_wlk_server

    stop_event = threading.Event()
    audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=8)
    state_queue: queue.Queue[CaptionState] | None = queue.Queue(maxsize=16) if config.show_window else None
    error_queue: queue.Queue[BaseException] = queue.Queue()
    server = None
    capture_thread: threading.Thread | None = None
    stream_thread: threading.Thread | None = None

    try:
        server = start_wlk_server(
            host="127.0.0.1",
            port=config.port,
            model=config.model,
            language=config.language,
            translation_mode=config.translation_mode,
            asr_prompt=config.asr_prompt,
        )
        capture_thread = threading.Thread(
            target=_capture_loop,
            kwargs={
                "config": config,
                "audio_queue": audio_queue,
                "stop_event": stop_event,
                "error_queue": error_queue,
            },
            daemon=True,
        )
        stream_thread = threading.Thread(
            target=_stream_loop,
            kwargs={
                "config": config,
                "audio_queue": audio_queue,
                "state_queue": state_queue,
                "stop_event": stop_event,
                "error_queue": error_queue,
            },
            daemon=True,
        )
        capture_thread.start()
        stream_thread.start()

        if config.show_window:
            from transcriber.live_window import CaptionWindow

            if state_queue is not None:
                CaptionWindow(state_queue=state_queue, stop_event=stop_event).run()
        else:
            while stream_thread.is_alive() and not stop_event.is_set():
                time.sleep(0.2)

        if not error_queue.empty():
            raise error_queue.get()
        return 0
    except KeyboardInterrupt:
        print("\nLive mode stopped.\n")
        return 0
    except Exception as exc:
        print(f"\nLive mode failed: {exc}\n")
        return 1
    finally:
        stop_event.set()
        if capture_thread is not None:
            capture_thread.join(timeout=2.0)
        if stream_thread is not None:
            stream_thread.join(timeout=12.0)
        stop_wlk_server(server)


def run_live_mode(args: argparse.Namespace) -> int:
    if sys.version_info < (3, 11):
        print("\nLive mode requires Python 3.11 or newer.\n")
        return 1

    try:
        if args.live_list_devices:
            from transcriber.live_audio import list_loopback_devices

            for device in list_loopback_devices():
                print(f"{device.index}: {device.name} ({device.channels} ch, {device.sample_rate} Hz)")
            return 0

        if args.live_loopback_test:
            from transcriber.live_audio import write_loopback_test_wav

            output_path = Path(args.output).expanduser()
            write_loopback_test_wav(
                output_path,
                seconds=max(0.0, float(args.seconds)),
                device_index=args.live_device_index,
                chunk_ms=_effective_live_chunk_ms(args),
            )
            print(f'\nWrote loopback test WAV: "{output_path}"\n')
            return 0

        return run_live_session(build_live_config(args))
    except RuntimeError as exc:
        print(f"\n{exc}\n")
        return 1
