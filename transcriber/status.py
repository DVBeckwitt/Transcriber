from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from transcriber.io import atomic_write_text

STATUS_VERSION = 1


def status_path_for_srt(srt_path: Path) -> Path:
    return srt_path.with_suffix(".transcription.json")


def source_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve(strict=False)),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def config_payload(config: object) -> dict[str, Any]:
    payload: dict[str, Any]
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        payload = dict(dataclasses.asdict(config))
    elif isinstance(config, Mapping):
        payload = dict(config)
    else:
        payload = {
            name: getattr(config, name)
            for name in dir(config)
            if not name.startswith("_") and not callable(getattr(config, name))
        }
    payload.pop("dry_run", None)
    return payload


def config_digest(config: object) -> str:
    raw = json.dumps(config_payload(config), sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_status_record(
    *,
    status: str,
    source_path: Path,
    config: object,
    srt_path: Path,
    started_at: str,
    finished_at: str | None = None,
    error: str | None = None,
    fallback_no_diarize: bool = False,
    detected_language: str | None = None,
) -> dict[str, Any]:
    return {
        "version": STATUS_VERSION,
        "status": status,
        "source": source_signature(source_path),
        "config_hash": config_digest(config),
        "srt_path": str(srt_path.resolve(strict=False)),
        "started_at": started_at,
        "finished_at": finished_at,
        "error": error,
        "fallback_no_diarize": fallback_no_diarize,
        "detected_language": detected_language,
        "english_output_mode": getattr(config, "english_output_mode", None),
    }


def write_status_file(path: Path, record: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(dict(record), indent=2, sort_keys=True, default=str) + "\n")


def read_status_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def status_matches_source_and_config(record: Mapping[str, Any], source_path: Path, config: object) -> bool:
    if record.get("version") != STATUS_VERSION:
        return False
    if record.get("status") != "succeeded":
        return False
    try:
        current_source = source_signature(source_path)
    except OSError:
        return False
    recorded_source = record.get("source")
    if not isinstance(recorded_source, Mapping):
        return False
    return (
        recorded_source.get("path") == current_source["path"]
        and recorded_source.get("size") == current_source["size"]
        and recorded_source.get("mtime_ns") == current_source["mtime_ns"]
        and record.get("config_hash") == config_digest(config)
    )
