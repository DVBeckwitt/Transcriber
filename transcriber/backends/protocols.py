from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TranscriptionBackend(Protocol):
    def transcribe(self, input_path: Path) -> dict[str, Any]:
        """Return a WhisperX-compatible transcription result."""


@runtime_checkable
class AlignmentBackend(Protocol):
    def align(self, result: dict[str, Any], audio: Any, *, language: str) -> dict[str, Any]:
        """Return a WhisperX-compatible aligned transcription result."""


@runtime_checkable
class DiarizationBackend(Protocol):
    def diarize(self, result: dict[str, Any], audio: Any) -> dict[str, Any]:
        """Return a WhisperX-compatible diarized transcription result."""


@runtime_checkable
class TranslationBackend(Protocol):
    def translate(self, texts: list[str], *, device: str) -> list[str]:
        """Translate texts while preserving input order."""


@runtime_checkable
class LiveCaptionBackend(Protocol):
    async def run(self) -> int:
        """Run a live caption session."""
