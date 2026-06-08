from __future__ import annotations

import json as json_module
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from transcriber.translation import (
    SUPPORTED_POST_TRANSLATION_LANGS,
    ServerTranslationBackend,
    TranslationRequest,
    TranslationResult,
    apply_glossary_placeholders,
    replace_glossary_placeholders,
    translate_srt_to_english,
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
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeHttpClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, *, json: dict[str, Any], timeout: float) -> FakeResponse:
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        request = json_module.loads(json["messages"][1]["content"])
        translations = {"Hola": "Hello", "Gracias": "Thanks"}
        items = [
            {"index": item["index"], "text": translations.get(item["text"], f"EN: {item['text']}")}
            for item in request["items"]
        ]
        return FakeResponse({"choices": [{"message": {"content": json_module.dumps({"items": items})}}]})


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

    def test_server_backend_posts_openai_compatible_chat_request(self) -> None:
        client = FakeHttpClient()
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
        self.assertEqual(client.calls[0]["json"]["model"], "Unbabel/Tower-Plus-72B")
        self.assertEqual(client.calls[0]["json"]["temperature"], 0)
        self.assertEqual(client.calls[0]["json"]["max_tokens"], 128)

    def test_server_backend_batches_chat_requests(self) -> None:
        client = FakeHttpClient()
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

    def test_server_backend_rejects_non_loopback_url(self) -> None:
        with self.assertRaisesRegex(ValueError, "localhost|loopback"):
            ServerTranslationBackend(server_url="https://example.com/v1")

    def test_server_backend_rejects_url_credentials(self) -> None:
        with self.assertRaisesRegex(ValueError, "credentials"):
            ServerTranslationBackend(server_url="http://user:pass@localhost:8000/v1")


if __name__ == "__main__":
    unittest.main()
