from __future__ import annotations

import contextlib
import ipaddress
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from transcriber.io import atomic_write_text

SUPPORTED_POST_TRANSLATION_LANGS = {"es", "de"}
DEFAULT_TRANSLATION_MODEL = "utter-project/EuroLLM-1.7B-Instruct"
GLOSSARY_PLACEHOLDER_TEMPLATE = "ZXGLOSSARY{index:03d}ZX"
DEFAULT_TRANSLATION_SERVER_HOST = "127.0.0.1"
DEFAULT_TRANSLATION_SERVER_PORT = 8000
DEFAULT_TRANSLATION_SERVER_START_TIMEOUT_SECONDS = 30 * 60.0


@dataclass(frozen=True)
class TranslationRequest:
    source_lang: str
    target_lang: str
    texts: Sequence[str]
    glossary: dict[str, str]
    preserve_markers: tuple[str, ...]


@dataclass(frozen=True)
class TranslationResult:
    texts: list[str]
    backend: str
    model: str
    warnings: list[str]


@dataclass(frozen=True)
class SubtitleCue:
    index: int
    start: str
    end: str
    text: str


@dataclass(frozen=True)
class _ParsedTranslationBatch:
    texts: list[str]
    warnings: list[str]


class _TranslationResponseFormatError(RuntimeError):
    pass


class TranslationBackend(Protocol):
    name: str
    model_name: str

    def translate_texts(
        self,
        request: TranslationRequest,
        *,
        device: str,
        batch_size: int,
        max_new_tokens: int,
    ) -> TranslationResult: ...


@dataclass
class ManagedTranslationServer:
    server_url: str
    process: Any | None = None
    started: bool = False
    _log_handle: Any | None = None

    @classmethod
    def start(
        cls,
        *,
        server_url: str | None,
        model_name: str,
        log_path: Path | None,
        executable_resolver: Callable[[], str | None] | None = None,
        port_picker: Callable[[], int] = lambda: pick_translation_server_port(),
        readiness_check: Callable[[str], bool] = lambda url: openai_server_ready(url),
        command_resolver: Callable[[str, int], Sequence[str] | None] = lambda model_name, port: local_vllm_command(
            model_name,
            port,
        ),
        popen: Callable[..., Any] = subprocess.Popen,
        sleep: Callable[[float], None] = time.sleep,
        timeout_seconds: float = DEFAULT_TRANSLATION_SERVER_START_TIMEOUT_SECONDS,
    ) -> ManagedTranslationServer:
        normalized_url = (server_url or "").strip().rstrip("/")
        if normalized_url:
            _validate_local_server_url(normalized_url)
            return cls(server_url=normalized_url)

        default_url = _translation_server_url(DEFAULT_TRANSLATION_SERVER_PORT)
        if readiness_check(default_url):
            return cls(server_url=default_url)

        port = int(port_picker())
        auto_url = _translation_server_url(port)
        if executable_resolver is not None:
            executable = executable_resolver()
            command = _native_vllm_command(executable, model_name, port) if executable else None
        else:
            resolved_command = command_resolver(model_name, port)
            command = list(resolved_command) if resolved_command else None
        if not command:
            raise RuntimeError(
                "Could not auto-start post-translation because the vllm command was not found. "
                "Install vLLM in Windows or WSL, put vllm on PATH, or pass --translation-server-url for an "
                "existing local server."
            )
        log_handle: Any | None = None
        stdout: Any = subprocess.DEVNULL
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_handle = log_path.open("ab")
            stdout = log_handle

        try:
            process = popen(command, stdout=stdout, stderr=subprocess.STDOUT)
        except Exception:
            if log_handle is not None:
                log_handle.close()
            raise

        server = cls(server_url=auto_url, process=process, started=True, _log_handle=log_handle)
        try:
            server._wait_until_ready(
                readiness_check=readiness_check,
                sleep=sleep,
                timeout_seconds=timeout_seconds,
                log_path=log_path,
            )
        except Exception:
            server.close()
            raise
        return server

    def _wait_until_ready(
        self,
        *,
        readiness_check: Callable[[str], bool],
        sleep: Callable[[float], None],
        timeout_seconds: float,
        log_path: Path | None,
    ) -> None:
        deadline = time.monotonic() + max(0.1, timeout_seconds)
        while True:
            if self.process is not None and self.process.poll() is not None:
                message = f"Post-translation server exited before it became ready (code {self.process.returncode})."
                if log_path is not None:
                    message += f' See "{log_path}".'
                raise RuntimeError(message)
            if readiness_check(self.server_url):
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                message = "Timed out waiting for the post-translation server to become ready."
                if log_path is not None:
                    message += f' See "{log_path}".'
                raise RuntimeError(message)
            sleep(min(2.0, max(0.1, remaining)))

    def close(self) -> None:
        try:
            if self.started and self.process is not None and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=15.0)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=15.0)
        finally:
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None

    def __enter__(self) -> ManagedTranslationServer:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def local_vllm_executable() -> str | None:
    script_dir = Path(sys.executable).parent
    candidate_names = ("vllm.exe", "vllm") if os.name == "nt" else ("vllm",)
    for name in candidate_names:
        candidate = script_dir / name
        if candidate.exists():
            return str(candidate)
    return shutil.which("vllm")


def local_vllm_command(model_name: str, port: int) -> list[str] | None:
    executable = local_vllm_executable()
    if executable:
        return _native_vllm_command(executable, model_name, port)
    if os.name == "nt" and wsl_vllm_available():
        return wsl_vllm_command(model_name, port)
    return None


def wsl_vllm_available(
    *,
    wsl_executable: str | None = None,
    runner: Callable[..., Any] = subprocess.run,
    timeout_seconds: float = 5.0,
) -> bool:
    wsl = wsl_executable or shutil.which("wsl.exe") or shutil.which("wsl")
    if not wsl:
        return False
    try:
        result = runner(
            [wsl, "-e", "sh", "-lc", "command -v vllm >/dev/null 2>&1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_seconds,
            check=False,
        )
    except Exception:
        return False
    return int(getattr(result, "returncode", 1)) == 0


def wsl_vllm_command(model_name: str, port: int, *, wsl_executable: str | None = None) -> list[str]:
    wsl = wsl_executable or shutil.which("wsl.exe") or shutil.which("wsl") or "wsl"
    shell_command = "exec " + " ".join(shlex.quote(part) for part in _native_vllm_command("vllm", model_name, port))
    return [wsl, "-e", "sh", "-lc", shell_command]


def _native_vllm_command(executable: str, model_name: str, port: int) -> list[str]:
    return [
        executable,
        "serve",
        model_name,
        "--host",
        DEFAULT_TRANSLATION_SERVER_HOST,
        "--port",
        str(int(port)),
    ]


def pick_translation_server_port(preferred_port: int = DEFAULT_TRANSLATION_SERVER_PORT) -> int:
    if _loopback_port_available(preferred_port):
        return preferred_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((DEFAULT_TRANSLATION_SERVER_HOST, 0))
        return int(sock.getsockname()[1])


def _urlopen_no_proxy(request: urllib.request.Request, *, timeout: float) -> Any:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=timeout)


def openai_server_ready(
    server_url: str,
    timeout_seconds: float = 1.0,
    opener: Callable[..., Any] = _urlopen_no_proxy,
) -> bool:
    try:
        _validate_local_server_url(server_url)
        _http_json_request(f"{server_url.rstrip('/')}/models", opener=opener, timeout=timeout_seconds)
    except Exception:
        return False
    return True


def _translation_server_url(port: int) -> str:
    return f"http://{DEFAULT_TRANSLATION_SERVER_HOST}:{int(port)}/v1"


def _loopback_port_available(port: int) -> bool:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((DEFAULT_TRANSLATION_SERVER_HOST, int(port)))
        except OSError:
            return False
    return True


def normalize_subtitle_whitespace(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    text = re.sub(r'([("])\s+', r"\1", text)
    text = re.sub(r'\s+([)")])', r"\1", text)
    return text


def split_speaker_prefix(text: str) -> tuple[str, str]:
    match = re.match(r"^([A-Z][A-Z0-9_ ]{1,31}:\s+)(.+)$", text.strip())
    if not match:
        return "", normalize_subtitle_whitespace(text)
    return match.group(1), normalize_subtitle_whitespace(match.group(2))


def apply_glossary_placeholders(text: str, glossary: dict[str, str]) -> tuple[str, dict[str, str]]:
    if not glossary or not text:
        return text, {}

    placeholder_map: dict[str, str] = {}
    updated = text
    placeholder_index = 0
    for source, target in sorted(glossary.items(), key=lambda item: len(item[0]), reverse=True):
        if not source:
            continue
        escaped = re.escape(source)
        pattern = rf"\b{escaped}\b" if re.fullmatch(r"[A-Za-z0-9_]+", source) else escaped
        if not re.search(pattern, updated):
            continue
        placeholder = GLOSSARY_PLACEHOLDER_TEMPLATE.format(index=placeholder_index)
        placeholder_index += 1
        updated = re.sub(pattern, placeholder, updated)
        placeholder_map[placeholder] = target
    return updated, placeholder_map


def replace_glossary_placeholders(text: str, placeholder_map: dict[str, str]) -> str:
    updated = text
    for placeholder, target in placeholder_map.items():
        updated = updated.replace(placeholder, target)
    return updated


def parse_srt_cues(srt_text: str) -> list[SubtitleCue]:
    cues: list[SubtitleCue] = []
    for block in re.split(r"\n\s*\n", srt_text.strip()):
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        line_index = 1 if lines[0].strip().isdigit() else 0
        if line_index >= len(lines) or "-->" not in lines[line_index]:
            continue
        start_raw, end_raw = [part.strip() for part in lines[line_index].split("-->", 1)]
        text = " ".join(line.strip() for line in lines[line_index + 1 :] if line.strip())
        if not text:
            continue
        cues.append(SubtitleCue(index=len(cues) + 1, start=start_raw, end=end_raw, text=text))
    return cues


def render_srt_cues(cues: Sequence[SubtitleCue]) -> str:
    rendered: list[str] = []
    for idx, cue in enumerate(cues, start=1):
        rendered.append(str(idx))
        rendered.append(f"{cue.start} --> {cue.end}")
        rendered.append(cue.text)
        rendered.append("")
    return "\n".join(rendered).rstrip() + "\n"


def write_translation_report(
    report_path: Path,
    *,
    source_lang: str,
    target_lang: str,
    result: TranslationResult,
    cue_count: int,
    fallback_count: int,
    english_output_mode: str,
) -> None:
    payload = {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "backend": result.backend,
        "model": result.model,
        "cue_count": cue_count,
        "warnings": result.warnings,
        "fallback_count": fallback_count,
        "english_output_mode": english_output_mode,
    }
    atomic_write_text(report_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def translate_srt_to_english(
    *,
    source_srt_path: Path,
    english_srt_path: Path,
    source_lang: str,
    backend: TranslationBackend,
    glossary: dict[str, str],
    device: str,
    batch_size: int,
    max_new_tokens: int,
    report_path: Path | None = None,
    english_output_mode: str = "post",
) -> TranslationResult:
    cues = parse_srt_cues(source_srt_path.read_text(encoding="utf-8", errors="ignore"))
    if not cues:
        raise RuntimeError("Source SRT has no translatable cues.")

    source_texts: list[str] = []
    cue_meta: list[tuple[SubtitleCue, str, dict[str, str]]] = []
    for cue in cues:
        prefix, text = split_speaker_prefix(cue.text)
        protected_text, placeholders = apply_glossary_placeholders(text, glossary)
        source_texts.append(protected_text)
        cue_meta.append((cue, prefix, placeholders))

    result = backend.translate_texts(
        TranslationRequest(
            source_lang=source_lang,
            target_lang="en",
            texts=source_texts,
            glossary=glossary,
            preserve_markers=tuple(sorted(glossary.values())),
        ),
        device=device,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )
    if len(result.texts) != len(cues):
        raise RuntimeError("Translation produced an unexpected cue count.")

    translated_cues: list[SubtitleCue] = []
    restored_texts: list[str] = []
    for translated_text, (cue, prefix, placeholders) in zip(result.texts, cue_meta, strict=True):
        restored = normalize_subtitle_whitespace(replace_glossary_placeholders(translated_text, placeholders))
        if cue.text.strip() and not restored:
            raise RuntimeError(f"Translation produced empty output for cue {cue.index}.")
        final_text = f"{prefix}{restored}".strip() if prefix else restored
        translated_cues.append(SubtitleCue(index=cue.index, start=cue.start, end=cue.end, text=final_text))
        restored_texts.append(final_text)

    atomic_write_text(english_srt_path, render_srt_cues(translated_cues))
    final_result = TranslationResult(
        texts=restored_texts,
        backend=result.backend,
        model=result.model,
        warnings=list(result.warnings),
    )
    if report_path is not None:
        write_translation_report(
            report_path,
            source_lang=source_lang,
            target_lang="en",
            result=final_result,
            cue_count=len(cues),
            fallback_count=0,
            english_output_mode=english_output_mode,
        )
    return final_result


class ServerTranslationBackend:
    name = "server"

    def __init__(
        self,
        *,
        server_url: str,
        model_name: str = DEFAULT_TRANSLATION_MODEL,
        client: Any | None = None,
    ) -> None:
        if not server_url:
            raise ValueError("translation server URL is required for the server backend.")
        _validate_local_server_url(server_url)
        self.server_url = server_url.rstrip("/")
        self.model_name = model_name
        self._client = client

    def translate_texts(
        self,
        request: TranslationRequest,
        *,
        device: str,
        batch_size: int,
        max_new_tokens: int,
    ) -> TranslationResult:
        if not request.texts:
            return TranslationResult(texts=[], backend=self.name, model=self.model_name, warnings=[])

        translated: list[str] = []
        warnings: list[str] = []
        for batch in _chunked_texts(request.texts, batch_size):
            try:
                batch_result = self._translate_batch(
                    source_lang=request.source_lang,
                    target_lang=request.target_lang,
                    texts=batch,
                    max_new_tokens=max_new_tokens,
                )
            except _TranslationResponseFormatError as exc:
                if len(batch) <= 1:
                    raise
                batch_result = self._retry_batch_as_single_items(
                    source_lang=request.source_lang,
                    target_lang=request.target_lang,
                    texts=batch,
                    max_new_tokens=max_new_tokens,
                    original_error=exc,
                )
            translated.extend(batch_result.texts)
            warnings.extend(batch_result.warnings)
        return TranslationResult(texts=translated, backend=self.name, model=self.model_name, warnings=warnings)

    def _translate_batch(
        self,
        *,
        source_lang: str,
        target_lang: str,
        texts: Sequence[str],
        max_new_tokens: int,
    ) -> _ParsedTranslationBatch:
        items = [{"index": idx, "text": text} for idx, text in enumerate(texts)]
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Translate subtitle text to English. Return exactly one JSON object with an items array. "
                        "Preserve item indexes, speaker labels, placeholders, names, numbers, and tags."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "source_lang": source_lang,
                            "target_lang": target_lang,
                            "items": items,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0,
            "top_p": 1,
            "max_tokens": max_new_tokens,
        }
        data = _http_json_request(
            f"{self.server_url}/chat/completions",
            payload=payload,
            opener=self._client if self._client is not None else _urlopen_no_proxy,
            timeout=120.0,
        )
        content = _chat_completion_content(data)
        return _parse_indexed_translation_content(content, expected_count=len(texts))

    def _retry_batch_as_single_items(
        self,
        *,
        source_lang: str,
        target_lang: str,
        texts: Sequence[str],
        max_new_tokens: int,
        original_error: _TranslationResponseFormatError,
    ) -> _ParsedTranslationBatch:
        translated: list[str] = []
        warnings = [f"Retried translation batch as single-item requests after response format error: {original_error}"]
        for text in texts:
            try:
                result = self._translate_batch(
                    source_lang=source_lang,
                    target_lang=target_lang,
                    texts=[text],
                    max_new_tokens=max_new_tokens,
                )
            except _TranslationResponseFormatError as exc:
                raise original_error from exc
            translated.extend(result.texts)
            warnings.extend(result.warnings)
        return _ParsedTranslationBatch(texts=translated, warnings=warnings)


def _chunked_texts(texts: Sequence[str], batch_size: int) -> list[Sequence[str]]:
    size = max(1, int(batch_size))
    return [texts[index : index + size] for index in range(0, len(texts), size)]


def _validate_local_server_url(server_url: str) -> None:
    parsed = urlparse(server_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("translation server URL must be an http(s) localhost or loopback URL.")
    if parsed.username or parsed.password:
        raise ValueError("translation server URL must not include credentials.")

    hostname = parsed.hostname.strip("[]").lower()
    if hostname == "localhost":
        return
    try:
        if ipaddress.ip_address(hostname).is_loopback:
            return
    except ValueError:
        pass
    raise ValueError("translation server URL must point to localhost or a loopback address.")


def _http_json_request(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    opener: Callable[..., Any],
    timeout: float,
) -> Any:
    data = None
    method = "GET"
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        method = "POST"
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with opener(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Translation server request failed with HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Translation server request failed: {exc.reason}") from exc
    except OSError as exc:
        raise RuntimeError(f"Translation server request failed: {exc}") from exc

    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("Translation server response was not valid JSON.") from exc


def _chat_completion_content(data: Any) -> str:
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Translation server response did not include chat message content.") from exc
    if not isinstance(content, str):
        raise RuntimeError("Translation server response content was not text.")
    return content


def _parse_indexed_translation_content(content: str, *, expected_count: int) -> _ParsedTranslationBatch:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise _TranslationResponseFormatError("Translation server response was not valid JSON.") from exc

    raw_items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(raw_items, list):
        raise _TranslationResponseFormatError("Translation server JSON did not contain an items array.")
    if len(raw_items) != expected_count:
        raise _TranslationResponseFormatError("Translation server returned an unexpected item count.")

    translated: dict[int, str] = {}
    positional_texts: list[str] = []
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            raise _TranslationResponseFormatError("Translation server item was not an object.")
        index = raw_item.get("index")
        text = raw_item.get("text")
        if not isinstance(text, str):
            raise _TranslationResponseFormatError("Translation server item must include text.")
        positional_texts.append(text)
        if type(index) is int and index not in translated:
            translated[index] = text

    expected_indexes = set(range(expected_count))
    if set(translated) == expected_indexes:
        return _ParsedTranslationBatch(texts=[translated[index] for index in range(expected_count)], warnings=[])
    return _ParsedTranslationBatch(
        texts=positional_texts,
        warnings=["Recovered translation batch by item order after invalid item indexes."],
    )
