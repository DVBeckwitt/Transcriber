"""Small filesystem and timestamp helpers shared across runtime layers."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

LOG_TAIL_READ_CHARS = 256 * 1024


def project_dir() -> Path:
    """Return the package project directory used for logs and local configuration."""

    return Path(__file__).resolve().parents[1]


def utc_now_iso() -> str:
    """Return the current UTC time in second-precision ISO 8601 form."""

    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def read_text_tail(path: Path, max_chars: int = LOG_TAIL_READ_CHARS) -> str:
    """Read at most the final ``max_chars`` characters from a UTF-8 text file."""

    if max_chars <= 0:
        return ""

    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_chars * 4))
            data = handle.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="ignore")[-max_chars:]
