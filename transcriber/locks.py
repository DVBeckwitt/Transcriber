from __future__ import annotations

import contextlib
import os
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

Reporter = Callable[[str], None]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_lock_payload(input_path: Path) -> str:
    return f"source_path={input_path}\ncreated_at={utc_now_iso()}\npid={os.getpid()}\nhostname={socket.gethostname()}\n"


def is_stale_lock(lock_path: Path, stale_lock_seconds: float) -> bool:
    try:
        age_seconds = time.time() - lock_path.stat().st_mtime
    except OSError:
        return False
    return age_seconds >= stale_lock_seconds


def try_remove_lock(lock_path: Path) -> bool:
    with contextlib.suppress(OSError):
        lock_path.unlink()
        return True
    return False


def touch_lock(lock_path: Path) -> None:
    with contextlib.suppress(OSError):
        os.utime(lock_path, None)


def acquire_lock(input_path: Path, lock_path: Path, stale_lock_seconds: float, report: Reporter) -> bool:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_lock_payload(input_path)
    for _ in range(2):
        try:
            with lock_path.open("x", encoding="utf-8") as fh:
                fh.write(payload)
            return True
        except FileExistsError:
            if is_stale_lock(lock_path, stale_lock_seconds):
                report(f'Clearing stale lock "{lock_path.name}".')
                if try_remove_lock(lock_path):
                    continue
            try:
                lock_text = lock_path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                lock_text = ""
            lock_info: dict[str, str] = {}
            for line in lock_text.splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                lock_info[key.strip()] = value.strip()
            owner = lock_info.get("hostname") or "unknown host"
            pid = lock_info.get("pid")
            created_at = lock_info.get("created_at") or "unknown time"
            report(
                f'Skipping "{input_path.name}" because it is already locked '
                f"(host={owner}, pid={pid}, created_at={created_at})."
            )
            return False
    return False


def release_lock(lock_path: Path, report: Reporter | None = None) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        if report is not None:
            report(f'Could not remove lock "{lock_path}": {exc}')


@contextmanager
def heartbeat_thread(lock_path: Path, stale_lock_seconds: float) -> Iterator[None]:
    interval = min(60.0, max(1.0, stale_lock_seconds / 3.0)) if stale_lock_seconds > 0 else 1.0
    stop_event = threading.Event()

    def beat() -> None:
        while not stop_event.wait(interval):
            touch_lock(lock_path)

    thread = threading.Thread(target=beat, name=f"transcriber-lock-{lock_path.name}", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1.0)
