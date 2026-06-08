from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass
from typing import Any

from transcriber.errors import PreflightError
from transcriber.translation import (
    DEFAULT_TRANSLATION_MODEL,
    DEFAULT_TRANSLATION_SERVER_HOST,
    DEFAULT_TRANSLATION_SERVER_PORT,
    local_vllm_command,
    openai_server_ready,
)


@dataclass(frozen=True)
class PreflightReport:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise PreflightError("\n".join(self.errors))

    def format(self) -> list[str]:
        lines: list[str] = []
        if self.errors:
            lines.append("Preflight failed:")
            lines.extend(f"  - {message}" for message in self.errors)
        if self.warnings:
            lines.append("Preflight warnings:")
            lines.extend(f"  - {message}" for message in self.warnings)
        return lines


def module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def executable_available(executable_name: str) -> bool:
    return shutil.which(executable_name) is not None


def cuda_available() -> bool:
    if not module_available("torch"):
        return False
    try:
        torch = importlib.import_module("torch")
    except Exception:
        return False
    return bool(getattr(torch.cuda, "is_available", lambda: False)())


def check_transcription_preflight(*, cfg: Any, hf_token_present: bool, require_ffmpeg: bool = True) -> PreflightReport:
    errors: list[str] = []
    warnings: list[str] = []

    if not module_available("whisperx"):
        errors.append("Missing Python package: whisperx. Install it before running file transcription.")
    if require_ffmpeg and not executable_available("ffmpeg"):
        errors.append("ffmpeg is not available on PATH. Install ffmpeg before file transcription.")

    device = str(getattr(cfg, "device", ""))
    if device.startswith("cuda") and not cuda_available():
        errors.append(
            "--device cuda was selected, but CUDA is not available to torch. Use --device cpu or install a CUDA-enabled torch stack."
        )

    if bool(getattr(cfg, "diarize", False)) and not hf_token_present:
        errors.append("Speaker diarization requires HF_TOKEN or hf_token.txt/HF_TOKEN.txt.")

    needs_post_translation = bool(getattr(cfg, "post_translate_to_english", False)) and not bool(
        getattr(cfg, "direct_whisper_translate", False)
    )
    if needs_post_translation and str(getattr(cfg, "translation_backend", "")) == "server":
        if not str(getattr(cfg, "translation_server_url", "") or "").strip():
            default_server_url = f"http://{DEFAULT_TRANSLATION_SERVER_HOST}:{DEFAULT_TRANSLATION_SERVER_PORT}/v1"
            server_ready = openai_server_ready(default_server_url)
            can_start_server = server_ready or bool(
                local_vllm_command(
                    DEFAULT_TRANSLATION_MODEL,
                    DEFAULT_TRANSLATION_SERVER_PORT,
                )
            )
            if can_start_server:
                warnings.append(
                    "No post-translation server URL was supplied; transcriber will reuse or auto-start a local server."
                )
            else:
                message = (
                    "Post-translation auto-start needs the vllm command. Install vLLM in Windows or WSL, "
                    "put vllm on PATH, or set --translation-server-url for an existing local server."
                )
                if str(getattr(cfg, "english_output_mode", "")) == "post":
                    errors.append(message)
                else:
                    warnings.append(message)

    return PreflightReport(errors=tuple(errors), warnings=tuple(warnings))


def validate_run_config(cfg: Any | None = None, **values: float | None) -> PreflightReport:
    errors: list[str] = []

    if cfg is not None:
        if str(getattr(cfg, "language", "")) not in {"auto", "en", "es", "de"}:
            errors.append("language must be one of: auto, en, es, de.")
        if str(getattr(cfg, "mode", "")) not in {"quality", "fast"}:
            errors.append("mode must be one of: quality, fast.")
        if str(getattr(cfg, "english_output_mode", "")) not in {"off", "direct", "post", "auto"}:
            errors.append("english_output_mode must be one of: off, direct, post, auto.")
        if str(getattr(cfg, "translation_backend", "")) not in {"server"}:
            errors.append("translation_backend must be one of: server.")
        for field in (
            "batch_size",
            "beam_size",
            "translation_batch_size",
            "translation_max_new_tokens",
        ):
            try:
                if int(getattr(cfg, field)) <= 0:
                    errors.append(f"{field} must be greater than 0.")
            except Exception:
                errors.append(f"{field} must be an integer.")
        for field in ("min_speaker_turn_ms", "min_speaker_turn_tokens"):
            try:
                if int(getattr(cfg, field)) < 0:
                    errors.append(f"{field} must be 0 or greater.")
            except Exception:
                errors.append(f"{field} must be an integer.")
        for temp in getattr(cfg, "temperature_schedule", ()) or ():
            try:
                value = float(temp)
            except Exception:
                errors.append("temperature_schedule must contain numeric values.")
                continue
            if not 0.0 <= value <= 1.0:
                errors.append("temperature_schedule values must be between 0 and 1.")
        for field in ("temperature", "low_confidence_word_prob", "high_no_speech_prob"):
            try:
                value = float(getattr(cfg, field))
            except Exception:
                errors.append(f"{field} must be numeric.")
                continue
            if not 0.0 <= value <= 1.0:
                errors.append(f"{field} must be between 0 and 1.")
        if float(getattr(cfg, "patience", 0.0)) <= 0.0:
            errors.append("patience must be greater than 0.")
        if str(getattr(cfg, "compute_type", "")) not in {"float16", "float32", "int8"}:
            errors.append("compute_type must be one of: float16, float32, int8.")

    poll_interval = values.get("poll_interval")
    settle_seconds = values.get("settle_seconds")
    stale_lock_seconds = values.get("stale_lock_seconds")
    if poll_interval is not None and poll_interval <= 0:
        errors.append("--poll-interval must be greater than 0.")
    if settle_seconds is not None and settle_seconds < 0:
        errors.append("--settle-seconds must be 0 or greater.")
    if stale_lock_seconds is not None and stale_lock_seconds <= 0:
        errors.append("--stale-lock-seconds must be greater than 0.")

    report = PreflightReport(errors=tuple(errors))
    if errors:
        raise ValueError("\n".join(errors))
    return report
