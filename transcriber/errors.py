from __future__ import annotations


class TranscriberError(Exception):
    """Base class for user-facing transcriber failures."""

    exit_code = 1


class DependencyError(TranscriberError):
    """A required third-party executable or package is missing."""


class PreflightError(TranscriberError):
    """The requested run cannot start safely."""


class TranscriptionError(TranscriberError):
    """WhisperX transcription failed."""


class AlignmentError(TranscriberError):
    """Word alignment failed."""


class DiarizationError(TranscriberError):
    """Speaker diarization failed."""


class TranslationError(TranscriberError):
    """Translation failed."""


class OutputWriteError(TranscriberError):
    """An output artifact could not be written or validated."""


class WatcherMoveError(TranscriberError):
    """Watch mode could not move completed artifacts."""
