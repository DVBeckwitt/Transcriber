from __future__ import annotations

import contextlib
import os
import tempfile
import time
from pathlib import Path

from transcriber.errors import OutputWriteError


def temporary_sibling_path(path: Path, suffix: str = ".tmp") -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}{suffix}")


def atomic_replace_path(source_path: Path, destination_path: Path) -> None:
    try:
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source_path, destination_path)
    except OSError as exc:
        raise OutputWriteError(f'Could not atomically replace "{destination_path}": {exc}') from exc


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd: int | None = None
    tmp_path: Path | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
        tmp_path = Path(tmp_name)
        with os.fdopen(fd, "w", encoding=encoding, newline="\n") as fh:
            fd = None
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except OSError as exc:
        raise OutputWriteError(f'Could not atomically write "{path}": {exc}') from exc
    finally:
        if fd is not None:
            os.close(fd)
        if tmp_path is not None and tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()
