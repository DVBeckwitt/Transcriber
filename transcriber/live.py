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
from typing import Any

from transcriber.io import atomic_write_text
from transcriber.live_wlk import CaptionPair, CaptionState, LiveTranslationMode

DEFAULT_SPANISH_TO_ENGLISH_STATIC_PROMPT = "\n".join(
    (
        "This is casual Spanish conversation.",
        "Translate into natural English.",
        "Spanish words casarse, casarnos, casarme, casar, boda, playa, and matrimonio refer to marriage or weddings when context fits.",
        "Do not translate casarse/casarnos as hunting.",
        "Preserve pronouns carefully.",
    )
)


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
    static_prompt: str | None
    backend: str
    backend_policy: str
    frame_threshold: int
    beams: int
    decoder: str
    audio_min_len: float
    audio_max_len: float
    nllb_backend: str
    nllb_size: str
    audio_diagnostics: bool


@dataclass
class LiveAudioQueueStats:
    dropped_chunks: int = 0


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
    return LiveTranslationMode.CASCADE if args.live_preset == "quality" else LiveTranslationMode.DIRECT


def _effective_live_chunk_ms(args: argparse.Namespace) -> int:
    if args.live_chunk_ms is not None:
        return max(20, int(args.live_chunk_ms))
    return 250 if args.live_preset == "latency" else 500


def _effective_live_model(args: argparse.Namespace) -> str:
    if args.model:
        return str(args.model)
    return "medium" if args.live_preset == "quality" else "small"


def _quality_default(args: argparse.Namespace, attribute: str, quality_value: Any, fallback_value: Any) -> Any:
    value = getattr(args, attribute)
    if value is not None:
        return value
    return quality_value if args.live_preset == "quality" else fallback_value


def _effective_live_static_prompt(args: argparse.Namespace, language: str) -> str | None:
    if args.live_static_prompt:
        return str(args.live_static_prompt)
    if args.live_preset == "quality" and language == "es":
        return DEFAULT_SPANISH_TO_ENGLISH_STATIC_PROMPT
    return None


def _effective_live_audio_lengths(args: argparse.Namespace) -> tuple[float, float]:
    audio_min_len = max(0.0, float(_quality_default(args, "live_audio_min_len", 0.0, 0.0)))
    audio_max_len = max(0.0, float(_quality_default(args, "live_audio_max_len", 30.0, 30.0)))
    if audio_max_len <= 0.0:
        raise RuntimeError("--live-audio-max-len must be greater than 0.")
    if audio_min_len > audio_max_len:
        raise RuntimeError("--live-audio-min-len cannot exceed --live-audio-max-len.")
    return audio_min_len, audio_max_len


def _live_audio_queue_maxsize(config: LiveConfig) -> int:
    return 0 if config.preset == "quality" else 8


def build_live_config(args: argparse.Namespace) -> LiveConfig:
    translation_mode = _effective_live_translation_mode(args)
    if translation_mode == LiveTranslationMode.DIRECT and args.live_save_bilingual_transcript:
        raise RuntimeError("--live-save-bilingual-transcript requires --live-translation-mode cascade.")
    language = args.lang or "es"
    audio_min_len, audio_max_len = _effective_live_audio_lengths(args)

    return LiveConfig(
        language=language,
        model=_effective_live_model(args),
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
        static_prompt=_effective_live_static_prompt(args, language),
        backend=str(_quality_default(args, "live_backend", "faster-whisper", "auto")),
        backend_policy=str(_quality_default(args, "live_backend_policy", "localagreement", "localagreement")),
        frame_threshold=max(1, int(_quality_default(args, "live_frame_threshold", 35, 25))),
        beams=max(1, int(_quality_default(args, "live_beams", 3, 1))),
        decoder=str(_quality_default(args, "live_decoder", "beam", "auto")),
        audio_min_len=audio_min_len,
        audio_max_len=audio_max_len,
        nllb_backend=str(_quality_default(args, "live_nllb_backend", "ctranslate2", "transformers")),
        nllb_size=str(_quality_default(args, "live_nllb_size", "600M", "600M")),
        audio_diagnostics=bool(args.live_audio_diagnostics),
    )


def _put_live_audio(
    audio_queue: queue.Queue[bytes],
    chunk: bytes,
    *,
    drop_oldest: bool,
    stats: LiveAudioQueueStats,
    stop_event: threading.Event,
) -> None:
    try:
        audio_queue.put(chunk, timeout=0.2)
    except queue.Full:
        if drop_oldest:
            with contextlib.suppress(queue.Empty):
                audio_queue.get_nowait()
                stats.dropped_chunks += 1
            with contextlib.suppress(queue.Full):
                audio_queue.put_nowait(chunk)
            return
        while not stop_event.is_set():
            try:
                audio_queue.put(chunk, timeout=0.2)
                return
            except queue.Full:
                continue


def _print_live_audio_diagnostics(
    *,
    config: LiveConfig,
    audio_queue: queue.Queue[bytes],
    stats: LiveAudioQueueStats,
    metrics: object,
) -> None:
    input_sample_rate = getattr(metrics, "input_sample_rate", "?")
    input_channels = getattr(metrics, "input_channels", "?")
    output_bytes = getattr(metrics, "output_bytes", "?")
    rms_level = getattr(metrics, "rms_level", "?")
    peak_level = getattr(metrics, "peak_level", "?")
    queue_depth = audio_queue.qsize()
    queue_delay_ms = queue_depth * config.chunk_ms
    print(
        "[live-audio] "
        f"input={input_sample_rate}Hz/{input_channels}ch "
        f"output_bytes={output_bytes} "
        f"rms={rms_level} peak={peak_level} "
        f"queue={queue_depth}/{audio_queue.maxsize} "
        f"dropped={stats.dropped_chunks} "
        f"queue_delay_ms={queue_delay_ms}"
    )


def _capture_loop(
    *,
    config: LiveConfig,
    audio_queue: queue.Queue[bytes],
    stop_event: threading.Event,
    error_queue: queue.Queue[BaseException],
) -> None:
    from transcriber.live_audio import LiveAudioDiagnostics, iter_loopback_pcm_chunks

    stats = LiveAudioQueueStats()
    on_metrics: Callable[[LiveAudioDiagnostics], None] | None = None
    if config.audio_diagnostics:

        def on_metrics(metrics: LiveAudioDiagnostics) -> None:
            _print_live_audio_diagnostics(
                config=config,
                audio_queue=audio_queue,
                stats=stats,
                metrics=metrics,
            )

    try:
        for chunk in iter_loopback_pcm_chunks(
            device_index=config.device_index,
            chunk_ms=config.chunk_ms,
            on_diagnostics=on_metrics,
        ):
            if stop_event.is_set():
                break
            _put_live_audio(
                audio_queue,
                chunk,
                drop_oldest=config.preset != "quality",
                stats=stats,
                stop_event=stop_event,
            )
    except BaseException as exc:
        error_queue.put(exc)
        stop_event.set()


def _write_committed_transcript(path: Path, state: CaptionState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(state.committed_lines)
    if text:
        text += "\n"
    atomic_write_text(path, text)


def _write_bilingual_transcript(path: Path, state: CaptionState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = [
        f"{index}.\nES: {pair.source_text}\nEN: {pair.translated_text}"
        for index, pair in enumerate(state.committed_pairs, start=1)
    ]
    text = "\n\n".join(blocks)
    if text:
        text += "\n"
    atomic_write_text(path, text)


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
    audio_diagnostics: bool = False,
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
        if audio_diagnostics and state.lag_seconds is not None:
            print(f"[live-audio] wlk_lag_seconds={state.lag_seconds:.2f}")
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
                    audio_diagnostics=config.audio_diagnostics,
                ),
            )
        )
    except BaseException as exc:
        error_queue.put(exc)
        stop_event.set()


def run_live_session(config: LiveConfig) -> int:
    from transcriber.live_wlk import start_wlk_server, stop_wlk_server

    stop_event = threading.Event()
    audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=_live_audio_queue_maxsize(config))
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
            static_prompt=config.static_prompt,
            backend=config.backend,
            backend_policy=config.backend_policy,
            frame_threshold=config.frame_threshold,
            beams=config.beams,
            decoder=config.decoder,
            audio_min_len=config.audio_min_len,
            audio_max_len=config.audio_max_len,
            nllb_backend=config.nllb_backend,
            nllb_size=config.nllb_size,
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
            from transcriber.live_audio import LiveAudioDiagnostics, write_loopback_test_wav

            output_path = Path(args.output).expanduser()
            on_metrics: Callable[[LiveAudioDiagnostics], None] | None = None
            if args.live_audio_diagnostics:

                def on_metrics(metrics: LiveAudioDiagnostics) -> None:
                    print(
                        "[live-audio] "
                        f"input={metrics.input_sample_rate}Hz/{metrics.input_channels}ch "
                        f"output_bytes={metrics.output_bytes} "
                        f"rms={metrics.rms_level} peak={metrics.peak_level}"
                    )

            write_loopback_test_wav(
                output_path,
                seconds=max(0.0, float(args.seconds)),
                device_index=args.live_device_index,
                chunk_ms=_effective_live_chunk_ms(args),
                on_diagnostics=on_metrics,
            )
            print(f'\nWrote loopback test WAV: "{output_path}"\n')
            return 0

        return run_live_session(build_live_config(args))
    except RuntimeError as exc:
        print(f"\n{exc}\n")
        return 1
