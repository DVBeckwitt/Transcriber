from __future__ import annotations

import os
import tempfile
import time
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

try:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Using `httpx` with `starlette.testclient` is deprecated.*",
        )
        from fastapi.testclient import TestClient
except ImportError as exc:  # pragma: no cover - exercised only without optional deps
    raise unittest.SkipTest("FastAPI server extras are not installed") from exc

import transcriber.server as server_module
from transcriber.__main__ import RunConfig, output_paths_for_input
from transcriber.server import ServerConfig, config_from_env_and_args, create_app

TOKEN = "test-token"


def auth_headers(token: str = TOKEN) -> dict[str, str]:
    return {"X-Transcribe-Proxy-Token": token}


def write_srt(cfg: RunConfig, source_path: Path, text: str = "Hello.") -> None:
    output_paths_for_input(source_path, cfg, create_dirs=True).srt_path.write_text(
        f"1\n00:00:00,000 --> 00:00:01,000\n{text}\n", encoding="utf-8"
    )


class ServerTests(unittest.TestCase):
    def make_client(
        self,
        tmpdir: str,
        *,
        max_upload_bytes: int = 1024,
        runner=None,
        device: str = "cpu",
        compute_type: str = "float32",
        max_pending_jobs: int | None = None,
        job_ttl_seconds: int = 24 * 60 * 60,
    ) -> TestClient:
        config = ServerConfig(
            proxy_token=TOKEN,
            work_dir=Path(tmpdir),
            max_upload_bytes=max_upload_bytes,
            max_pending_jobs=max_pending_jobs,
            job_ttl_seconds=job_ttl_seconds,
            device=device,
            compute_type=compute_type,
        )
        return TestClient(create_app(config, transcribe_runner=runner))

    def assert_error(self, response, status_code: int, code: str) -> None:
        self.assertEqual(response.status_code, status_code)
        payload = response.json()
        self.assertEqual(set(payload), {"error"})
        self.assertEqual(set(payload["error"]), {"code", "message"})
        self.assertEqual(payload["error"]["code"], code)
        self.assertIsInstance(payload["error"]["message"], str)
        self.assertTrue(payload["error"]["message"])

    def wait_for_terminal_status(
        self, client: TestClient, job_id: str
    ) -> dict[str, object]:
        for _ in range(100):
            response = client.get(
                f"/api/transcriptions/{job_id}", headers=auth_headers()
            )
            self.assertEqual(response.status_code, 200)
            payload = response.json()
            if payload["status"] in {"succeeded", "failed"}:
                return payload
            time.sleep(0.01)
        self.fail(f"job {job_id} did not finish")

    def test_health_requires_auth_and_sets_security_headers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = self.make_client(tmpdir)
            responses = [
                client.get("/api/transcriptions/health"),
                client.get("/api/transcriptions/health", headers=auth_headers("wrong")),
                client.get("/api/transcriptions/health", headers=auth_headers()),
            ]
            for response in responses[:2]:
                self.assert_error(response, 401, "UNAUTHORIZED")
            self.assertEqual(responses[2].json()["status"], "ok")
            for response in responses:
                self.assertEqual(response.headers["cache-control"], "no-store")
                self.assertEqual(response.headers["x-content-type-options"], "nosniff")
                self.assertEqual(response.headers["referrer-policy"], "no-referrer")

    def test_rejects_invalid_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = self.make_client(tmpdir)

            response = client.post(
                "/api/transcriptions",
                data={"language": "fr"},
                files={"file": ("clip.mp3", b"audio", "audio/mpeg")},
                headers=auth_headers(),
            )

            self.assert_error(response, 422, "INVALID_LANGUAGE")

    def test_rejects_unsupported_extension_and_oversize_upload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = self.make_client(tmpdir)

            bad_extension = client.post(
                "/api/transcriptions",
                data={"language": "auto"},
                files={"file": ("notes.txt", b"abc", "text/plain")},
                headers=auth_headers(),
            )
            self.assert_error(bad_extension, 415, "UNSUPPORTED_MEDIA_TYPE")

        with tempfile.TemporaryDirectory() as tmpdir:
            client = self.make_client(tmpdir, max_upload_bytes=3)

            too_large = client.post(
                "/api/transcriptions",
                data={"language": "auto"},
                files={"file": ("clip.mp3", b"abcd", "audio/mpeg")},
                headers=auth_headers(),
            )
            self.assert_error(too_large, 413, "UPLOAD_TOO_LARGE")

        with tempfile.TemporaryDirectory() as tmpdir:
            client = self.make_client(tmpdir, max_pending_jobs=1)
            client.app.state.transcriber_jobs.create("en", ".wav")
            busy = client.post(
                "/api/transcriptions",
                files={"file": ("clip.wav", b"audio", "audio/wav")},
                headers=auth_headers(),
            )
            self.assert_error(busy, 429, "SERVER_BUSY")

    def test_upload_extension_mapping(self) -> None:
        seen_sources: list[Path] = []

        def runner(cfg: RunConfig, source_path: Path, report=print) -> int:
            seen_sources.append(source_path)
            write_srt(cfg, source_path)
            return 0

        cases = [
            ("clip.weba", "audio/webm", ".weba"),
            ("blob", "audio/webm", ".webm"),
            *(("clip.txt", mime, ".webm") for mime in server_module.WEBM_MIME_TYPES),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            client = self.make_client(tmpdir, runner=runner)
            for filename, content_type, expected_suffix in cases:
                with self.subTest(
                    filename=filename,
                    content_type=content_type,
                    expected_suffix=expected_suffix,
                ):
                    response = client.post(
                        "/api/transcriptions",
                        data={"language": "auto"},
                        files={"file": (filename, b"audio", content_type)},
                        headers=auth_headers(),
                    )
                    self.assertEqual(response.status_code, 202)
                    self.wait_for_terminal_status(client, response.json()["jobId"])

        self.assertEqual(
            [source.suffix for source in seen_sources], [case[2] for case in cases]
        )

    def test_maps_server_request_to_quality_no_speaker_labels_config(self) -> None:
        seen: list[RunConfig] = []

        def runner(cfg: RunConfig, source_path: Path, report=print) -> int:
            seen.append(cfg)
            write_srt(cfg, source_path, "Hola.")
            return 0

        with tempfile.TemporaryDirectory() as tmpdir:
            client = self.make_client(
                tmpdir, runner=runner, device="cuda", compute_type="int8"
            )
            response = client.post(
                "/api/transcriptions",
                data={"language": "es"},
                files={"file": ("clip.m4a", b"audio", "audio/mp4")},
                headers=auth_headers(),
            )
            self.assertEqual(response.status_code, 202)

            payload = self.wait_for_terminal_status(client, response.json()["jobId"])

            self.assertEqual(payload["status"], "succeeded")
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0].mode, "quality")
            self.assertEqual(seen[0].language, "es")
            self.assertFalse(seen[0].include_speaker_labels)
            self.assertEqual(seen[0].device, "cuda")
            self.assertEqual(seen[0].compute_type, "int8")
            self.assertEqual(seen[0].low_confidence_word_prob, 0.10)
            self.assertFalse(seen[0].warm_vram)

    def test_server_warm_vram_config_defaults_off_and_can_be_toggled(self) -> None:
        base_env = {"TRANSCRIBE_PROXY_TOKEN": TOKEN}

        self.assertFalse(config_from_env_and_args([], env=base_env).warm_vram)
        self.assertTrue(
            config_from_env_and_args(
                [], env={**base_env, "TRANSCRIBE_WARM_VRAM": "true"}
            ).warm_vram
        )
        self.assertTrue(
            config_from_env_and_args(["--warm-vram"], env=base_env).warm_vram
        )
        self.assertFalse(
            config_from_env_and_args(
                ["--warm-vram", "--no-warm-vram"],
                env={**base_env, "TRANSCRIBE_WARM_VRAM": "true"},
            ).warm_vram
        )
        self.assertEqual(
            server_module.normalize_config(
                ServerConfig(proxy_token=TOKEN, max_workers=3)
            ).max_pending_jobs,
            6,
        )
        with self.assertRaisesRegex(ValueError, "max-workers 1"):
            server_module.normalize_config(
                ServerConfig(proxy_token=TOKEN, max_workers=2, warm_vram=True)
            )

    def test_downloads_srt_and_extracted_plain_text_transcript(self) -> None:
        def runner(cfg: RunConfig, source_path: Path, report=print) -> int:
            outputs = output_paths_for_input(source_path, cfg, create_dirs=True)
            outputs.srt_path.write_text(
                (
                    "1\n"
                    "00:00:00,000 --> 00:00:01,000\n"
                    "Hello there.\n\n"
                    "2\n"
                    "00:00:01,000 --> 00:00:02,000\n"
                    "__LOWCONF_65__Second line.__LOWCONF_END__\n"
                ),
                encoding="utf-8",
            )
            return 0

        with tempfile.TemporaryDirectory() as tmpdir:
            client = self.make_client(tmpdir, runner=runner)
            response = client.post(
                "/api/transcriptions",
                data={"language": "en"},
                files={"file": ("clip.wav", b"audio", "audio/wav")},
                headers=auth_headers(),
            )
            job_id = response.json()["jobId"]
            self.wait_for_terminal_status(client, job_id)

            srt = client.get(
                f"/api/transcriptions/{job_id}/transcript.srt", headers=auth_headers()
            )
            self.assertEqual(srt.status_code, 200)
            self.assertIn("Hello there.", srt.text)
            self.assertIn("00:00:00,000 --> 00:00:01,000", srt.text)

            text = client.get(
                f"/api/transcriptions/{job_id}/transcript.txt", headers=auth_headers()
            )
            self.assertEqual(text.status_code, 200)
            self.assertEqual(text.text, "Hello there.\n—\n")

    def test_failed_job_deletes_uploaded_source_and_uses_consistent_errors(
        self,
    ) -> None:
        uploaded_sources: list[Path] = []

        def runner(cfg: RunConfig, source_path: Path, report=print) -> int:
            uploaded_sources.append(source_path)
            self.assertTrue(source_path.exists())
            report('Log: "C:\\Users\\Example\\secret\\transcript_whisperx.log"')
            report("Worker failure: no subtitle cues.")
            return 1

        with tempfile.TemporaryDirectory() as tmpdir:
            client = self.make_client(tmpdir, runner=runner)
            response = client.post(
                "/api/transcriptions",
                data={"language": "auto"},
                files={"file": ("clip.mp4", b"audio", "video/mp4")},
                headers=auth_headers(),
            )
            job_id = response.json()["jobId"]

            status_payload = self.wait_for_terminal_status(client, job_id)

            self.assertEqual(status_payload["status"], "failed")
            self.assertEqual(status_payload["error"]["code"], "TRANSCRIPTION_FAILED")
            self.assertEqual(
                status_payload["error"]["message"], "Worker failure: no subtitle cues."
            )
            self.assertEqual(
                status_payload["error"]["details"]["logName"], f"{job_id}_whisperx.log"
            )
            self.assertIn(
                "Worker failure: no subtitle cues.",
                status_payload["error"]["details"]["reports"],
            )
            self.assertNotIn(
                "C:\\Users\\Example",
                "\n".join(status_payload["error"]["details"]["reports"]),
            )
            self.assertFalse(uploaded_sources[0].exists())
            self.assert_error(
                client.get(
                    f"/api/transcriptions/{job_id}/transcript.txt",
                    headers=auth_headers(),
                ),
                409,
                "JOB_NOT_READY",
            )
            self.assert_error(
                client.get("/api/transcriptions/missing-job", headers=auth_headers()),
                404,
                "JOB_NOT_FOUND",
            )

    def test_failure_log_reader_uses_bounded_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "worker.log"
            log_path.write_text("old\n" * 1000 + "recent\n", encoding="utf-8")

            with patch(
                "transcriber.server.read_text_tail", return_value="line one\nline two\n"
            ) as read_tail:
                lines = server_module.read_failure_log_lines(log_path)

        self.assertEqual(lines, ["line one", "line two"])
        read_tail.assert_called_once_with(
            log_path, max_chars=server_module.MAX_FAILURE_LOG_CHARS
        )

    def test_native_transcription_crash_is_recorded_as_failed_job(self) -> None:
        class CrashedProcess:
            returncode = 3221225477

            def wait(self) -> int:
                return self.returncode

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            store = server_module.JobStore(work_dir, 24 * 60 * 60)
            record = store.create("en", ".wav")
            record.source_path.write_bytes(b"audio")
            config = ServerConfig(
                proxy_token=TOKEN,
                work_dir=work_dir,
                device="cpu",
                compute_type="float32",
            )

            with patch.object(
                server_module.subprocess, "Popen", return_value=CrashedProcess()
            ):
                server_module.run_transcription_job(
                    record.job_id,
                    store,
                    config,
                    server_module.transcribe_file,
                )

            failed = store.get(record.job_id)
            self.assertIsNotNone(failed)
            self.assertEqual(failed.status, "failed")
            self.assertEqual(failed.error["code"], "TRANSCRIPTION_FAILED")
            self.assertIn("3221225477", failed.error["message"])
            self.assertFalse(record.source_path.exists())

    def test_warm_worker_runs_in_process_and_cold_worker_stays_isolated(self) -> None:
        def runner(cfg: RunConfig, source_path: Path, report=print) -> int:
            write_srt(cfg, source_path)
            return 0

        with tempfile.TemporaryDirectory() as tmpdir:
            store = server_module.JobStore(Path(tmpdir), 3600, 2)
            record = store.create("en", ".wav")
            record.source_path.write_bytes(b"audio")
            config = server_module.normalize_config(
                ServerConfig(proxy_token=TOKEN, work_dir=Path(tmpdir), warm_vram=True)
            )
            with (
                patch.object(
                    server_module, "transcribe_file", side_effect=runner
                ) as in_process,
                patch.object(
                    server_module, "run_transcription_subprocess"
                ) as subprocess_runner,
            ):
                server_module.run_transcription_job(
                    record.job_id, store, config, in_process
                )

            self.assertEqual(store.get(record.job_id).status, "succeeded")
            in_process.assert_called_once()
            subprocess_runner.assert_not_called()

    def test_job_store_bounds_active_work_and_restores_status_after_restart(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            store = server_module.JobStore(work_dir, 3600, 1)
            completed = store.create("en", ".wav")
            completed.srt_path.write_text("subtitle", encoding="utf-8")
            completed.txt_path.write_text("transcript", encoding="utf-8")
            store.mark_succeeded(completed.job_id)
            interrupted = store.create("es", ".m4a")
            interrupted.source_path.write_bytes(b"audio")
            store.mark_running(interrupted.job_id)

            with self.assertRaises(server_module.ApiError) as busy:
                store.create("auto", ".mp3")
            self.assertEqual(busy.exception.status_code, 429)

            restored = server_module.JobStore(work_dir, 3600, 1)
            self.assertEqual(restored.get(completed.job_id).status, "succeeded")
            failed = restored.get(interrupted.job_id)
            self.assertEqual(failed.status, "failed")
            self.assertEqual(failed.error["code"], "WORKER_RESTARTED")
            self.assertFalse(interrupted.source_path.exists())

    def test_cleanup_removes_stale_untracked_job_directories_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)
            stale_job_dir = work_dir / ("a" * 32)
            fresh_job_dir = work_dir / ("b" * 32)
            non_job_dir = work_dir / "manual-notes"
            stale_job_dir.mkdir()
            fresh_job_dir.mkdir()
            non_job_dir.mkdir()
            old = time.time() - 10
            os.utime(stale_job_dir, (old, old))

            client = self.make_client(tmpdir, job_ttl_seconds=1)
            response = client.get("/api/transcriptions/health", headers=auth_headers())

            self.assertEqual(response.status_code, 200)
            self.assertFalse(stale_job_dir.exists())
            self.assertTrue(fresh_job_dir.exists())
            self.assertTrue(non_job_dir.exists())


if __name__ == "__main__":
    unittest.main()
