from __future__ import annotations

import json as json_module
import shlex
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from urllib.error import URLError

from transcriber.translation import (
    DEFAULT_TRANSLATION_MODEL,
    SUPPORTED_POST_TRANSLATION_LANGS,
    ManagedTranslationServer,
    ServerTranslationBackend,
    TranslationRequest,
    TranslationResult,
    apply_glossary_placeholders,
    openai_server_ready,
    replace_glossary_placeholders,
    translate_srt_to_english,
    wsl_vllm_available,
    wsl_vllm_command,
)


class FakeTranslationBackend:
    name = "fake"
    model_name = "fake-model"

    def __init__(self, outputs: list[str] | None = None, warnings: list[str] | None = None) -> None:
        self.outputs = outputs
        self.warnings = warnings or []
        self.requests: list[TranslationRequest] = []

    def translate_texts(
        self,
        request: TranslationRequest,
        *,
        device: str,
        batch_size: int,
        max_new_tokens: int,
    ) -> TranslationResult:
        self.requests.append(request)
        texts = self.outputs if self.outputs is not None else [f"EN: {text}" for text in request.texts]
        return TranslationResult(
            texts=list(texts),
            backend=self.name,
            model=self.model_name,
            warnings=list(self.warnings),
        )


class FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.body = json_module.dumps(payload).encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class FakeUrlOpen:
    def __init__(self, fail_with: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_with = fail_with

    def __call__(self, request: Any, *, timeout: float) -> FakeResponse:
        if self.fail_with is not None:
            raise self.fail_with
        body = request.data.decode("utf-8") if request.data else ""
        payload = json_module.loads(body) if body else {}
        self.calls.append(
            {
                "url": request.full_url,
                "method": request.get_method(),
                "headers": dict(request.header_items()),
                "json": payload,
                "timeout": timeout,
            }
        )
        if request.get_method() == "GET":
            return FakeResponse({"data": []})
        translation_request = json_module.loads(payload["messages"][1]["content"])
        translations = {"Hola": "Hello", "Gracias": "Thanks"}
        items = [
            {"index": item["index"], "text": translations.get(item["text"], f"EN: {item['text']}")}
            for item in translation_request["items"]
        ]
        return FakeResponse({"choices": [{"message": {"content": json_module.dumps({"items": items})}}]})


class FakeServerProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode or 0


class TranslationTests(unittest.TestCase):
    def test_translation_module_does_not_import_heavy_model_packages(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import transcriber.translation; print('torch' in sys.modules, 'vllm' in sys.modules)",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "False False")

    def test_supported_post_translation_languages_include_spanish_and_german(self) -> None:
        self.assertEqual(SUPPORTED_POST_TRANSLATION_LANGS, {"es", "de"})

    def test_glossary_placeholders_are_exact_and_restored(self) -> None:
        protected, placeholders = apply_glossary_placeholders(
            "Hola OpenAI y Falconer",
            {"OpenAI": "OpenAI", "Falcon": "Falcon"},
        )

        self.assertIn("ZXGLOSSARY000ZX", protected)
        self.assertIn("Falconer", protected)
        self.assertEqual(replace_glossary_placeholders(protected, placeholders), "Hola OpenAI y Falconer")

    def test_translate_srt_to_english_preserves_structure_and_writes_report(self) -> None:
        with TemporaryDirectory() as tmpdir:
            source_srt = Path(tmpdir) / "meeting.es.srt"
            english_srt = Path(tmpdir) / "meeting.en.srt"
            report_path = Path(tmpdir) / "meeting.translation.json"
            source_srt.write_text(
                "\n".join(
                    [
                        "1",
                        "00:00:00,000 --> 00:00:01,000",
                        "SPEAKER_00: Hola OpenAI",
                        "",
                        "2",
                        "00:00:01,000 --> 00:00:02,000",
                        "Gracias",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            result = translate_srt_to_english(
                source_srt_path=source_srt,
                english_srt_path=english_srt,
                source_lang="es",
                backend=FakeTranslationBackend(warnings=["low confidence"]),
                glossary={"OpenAI": "OpenAI"},
                device="cpu",
                batch_size=2,
                max_new_tokens=256,
                report_path=report_path,
                english_output_mode="post",
            )

            english_text = english_srt.read_text(encoding="utf-8")
            report = json_module.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(result.backend, "fake")
            self.assertIn("00:00:00,000 --> 00:00:01,000", english_text)
            self.assertIn("00:00:01,000 --> 00:00:02,000", english_text)
            self.assertIn("SPEAKER_00: EN: Hola OpenAI", english_text)
            self.assertEqual(report["source_lang"], "es")
            self.assertEqual(report["target_lang"], "en")
            self.assertEqual(report["cue_count"], 2)
            self.assertEqual(report["warnings"], ["low confidence"])
            self.assertEqual(report["english_output_mode"], "post")
            self.assertNotIn("Hola", json_module.dumps(report))

    def test_translate_srt_to_english_rejects_changed_cue_count(self) -> None:
        with TemporaryDirectory() as tmpdir:
            source_srt = Path(tmpdir) / "meeting.de.srt"
            english_srt = Path(tmpdir) / "meeting.en.srt"
            source_srt.write_text(
                "1\n00:00:00,000 --> 00:00:01,000\nHallo\n\n2\n00:00:01,000 --> 00:00:02,000\nDanke\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "cue count"):
                translate_srt_to_english(
                    source_srt_path=source_srt,
                    english_srt_path=english_srt,
                    source_lang="de",
                    backend=FakeTranslationBackend(outputs=["Hello"]),
                    glossary={},
                    device="cpu",
                    batch_size=2,
                    max_new_tokens=256,
                )

    def test_openai_server_ready_uses_stdlib_opener(self) -> None:
        opener = FakeUrlOpen()

        self.assertTrue(openai_server_ready("http://localhost:8000/v1", opener=opener))
        self.assertEqual(opener.calls[0]["url"], "http://localhost:8000/v1/models")
        self.assertEqual(opener.calls[0]["method"], "GET")

    def test_openai_server_ready_returns_false_on_url_error(self) -> None:
        opener = FakeUrlOpen(fail_with=URLError("connection refused"))

        self.assertFalse(openai_server_ready("http://localhost:8000/v1", opener=opener))

    def test_server_backend_posts_openai_compatible_chat_request(self) -> None:
        client = FakeUrlOpen()
        backend = ServerTranslationBackend(
            server_url="http://localhost:8000/v1",
            model_name="Unbabel/Tower-Plus-72B",
            client=client,
        )

        result = backend.translate_texts(
            TranslationRequest(
                source_lang="es",
                target_lang="en",
                texts=["Hola", "Gracias"],
                glossary={},
                preserve_markers=(),
            ),
            device="cpu",
            batch_size=2,
            max_new_tokens=128,
        )

        self.assertEqual(result.texts, ["Hello", "Thanks"])
        self.assertEqual(client.calls[0]["url"], "http://localhost:8000/v1/chat/completions")
        self.assertEqual(client.calls[0]["method"], "POST")
        self.assertEqual(client.calls[0]["json"]["model"], "Unbabel/Tower-Plus-72B")
        self.assertEqual(client.calls[0]["json"]["temperature"], 0)
        self.assertEqual(client.calls[0]["json"]["max_tokens"], 128)
        content_type = {key.lower(): value for key, value in client.calls[0]["headers"].items()}["content-type"]
        self.assertEqual(content_type, "application/json")

    def test_server_backend_batches_chat_requests(self) -> None:
        client = FakeUrlOpen()
        backend = ServerTranslationBackend(
            server_url="http://127.0.0.1:8000/v1",
            model_name="Unbabel/Tower-Plus-72B",
            client=client,
        )

        result = backend.translate_texts(
            TranslationRequest(
                source_lang="de",
                target_lang="en",
                texts=["eins", "zwei", "drei", "vier", "funf"],
                glossary={},
                preserve_markers=(),
            ),
            device="cpu",
            batch_size=2,
            max_new_tokens=128,
        )

        batch_sizes = [len(json_module.loads(call["json"]["messages"][1]["content"])["items"]) for call in client.calls]
        self.assertEqual(batch_sizes, [2, 2, 1])
        self.assertEqual(result.texts, ["EN: eins", "EN: zwei", "EN: drei", "EN: vier", "EN: funf"])

    def test_server_backend_raises_clear_error_on_network_failure(self) -> None:
        backend = ServerTranslationBackend(
            server_url="http://localhost:8000/v1",
            client=FakeUrlOpen(fail_with=URLError("connection refused")),
        )

        with self.assertRaisesRegex(RuntimeError, "Translation server request failed.*connection refused"):
            backend.translate_texts(
                TranslationRequest(
                    source_lang="de",
                    target_lang="en",
                    texts=["Hallo"],
                    glossary={},
                    preserve_markers=(),
                ),
                device="cpu",
                batch_size=1,
                max_new_tokens=128,
            )

    def test_server_backend_rejects_non_loopback_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "localhost|loopback"):
            ServerTranslationBackend(server_url="https://example.com/v1")

    def test_server_backend_rejects_url_credentials(self) -> None:
        with self.assertRaisesRegex(ValueError, "credentials"):
            ServerTranslationBackend(server_url="http://user:pass@localhost:8000/v1")

    def test_managed_translation_server_starts_vllm_and_stops_owned_process(self) -> None:
        process = FakeServerProcess()
        popen_calls: list[list[str]] = []

        def fake_popen(command: list[str], **_kwargs: Any) -> FakeServerProcess:
            popen_calls.append(command)
            return process

        def fake_readiness_check(url: str) -> bool:
            return url == "http://127.0.0.1:8123/v1" and bool(popen_calls)

        with ManagedTranslationServer.start(
            server_url=None,
            model_name=DEFAULT_TRANSLATION_MODEL,
            log_path=None,
            executable_resolver=lambda: "vllm",
            port_picker=lambda: 8123,
            readiness_check=fake_readiness_check,
            popen=fake_popen,
            sleep=lambda _seconds: None,
            timeout_seconds=1.0,
        ) as server:
            self.assertEqual(server.server_url, "http://127.0.0.1:8123/v1")
            self.assertEqual(
                popen_calls[0],
                ["vllm", "serve", DEFAULT_TRANSLATION_MODEL, "--host", "127.0.0.1", "--port", "8123"],
            )
            self.assertTrue(server.started)

        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)

    def test_managed_translation_server_can_start_wsl_vllm_command(self) -> None:
        process = FakeServerProcess()
        popen_calls: list[list[str]] = []

        def fake_popen(command: list[str], **_kwargs: Any) -> FakeServerProcess:
            popen_calls.append(command)
            return process

        with ManagedTranslationServer.start(
            server_url=None,
            model_name=DEFAULT_TRANSLATION_MODEL,
            log_path=None,
            port_picker=lambda: 8124,
            readiness_check=lambda url: url == "http://127.0.0.1:8124/v1" and bool(popen_calls),
            command_resolver=lambda model_name, port: wsl_vllm_command(
                model_name,
                port,
                wsl_executable="wsl.exe",
            ),
            popen=fake_popen,
            sleep=lambda _seconds: None,
            timeout_seconds=1.0,
        ) as server:
            self.assertEqual(server.server_url, "http://127.0.0.1:8124/v1")
            self.assertEqual(popen_calls[0][:4], ["wsl.exe", "-e", "sh", "-lc"])
            self.assertIn("vllm serve", popen_calls[0][4])
            self.assertIn("--port 8124", popen_calls[0][4])

        self.assertTrue(process.terminated)

    def test_wsl_vllm_available_checks_default_distro_for_vllm(self) -> None:
        calls: list[list[str]] = []

        def fake_runner(command: list[str], **_kwargs: Any) -> Any:
            calls.append(command)
            return SimpleNamespace(returncode=0)

        self.assertTrue(wsl_vllm_available(wsl_executable="wsl.exe", runner=fake_runner))
        self.assertEqual(calls[0], ["wsl.exe", "-e", "sh", "-lc", "command -v vllm >/dev/null 2>&1"])

    def test_wsl_vllm_command_quotes_model_name_for_shell(self) -> None:
        model_name = "model; touch /tmp/pwn && echo 'x'"

        command = wsl_vllm_command(model_name, 8124, wsl_executable="wsl.exe")

        self.assertEqual(
            shlex.split(command[4]),
            ["exec", "vllm", "serve", model_name, "--host", "127.0.0.1", "--port", "8124"],
        )

    def test_managed_translation_server_reuses_supplied_url_without_starting_process(self) -> None:
        with ManagedTranslationServer.start(
            server_url="http://localhost:8000/v1",
            model_name=DEFAULT_TRANSLATION_MODEL,
            log_path=None,
            popen=lambda _command, **_kwargs: self.fail("server should not start"),
        ) as server:
            self.assertEqual(server.server_url, "http://localhost:8000/v1")
            self.assertFalse(server.started)


if __name__ == "__main__":
    unittest.main()
