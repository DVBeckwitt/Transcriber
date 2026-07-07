# Security Notes

This project is a local transcription launcher plus an optional LAN worker API. It is not intended to be exposed directly to the public internet.

## Trust Boundaries

| Boundary | Risk | Control |
| --- | --- | --- |
| Homepage proxy to worker API | Unauthorized uploads or transcript reads | Shared bearer token in `X-Transcribe-Proxy-Token` on every request |
| Multipart uploads | Disk exhaustion, unsupported inputs, malformed forms | Upload size limit, language validation, supported extension allowlist |
| Worker filesystem | Transcript/source media leakage | Per-job directories, source deletion after success/failure, TTL cleanup |
| WhisperX/pyannote models | Third-party checkpoint execution/trust | Diarization token and model terms documented as trusted-model boundary |
| Client-facing errors | Internal path or stack trace disclosure | Consistent JSON errors with generic messages |
| Browser/proxy caching | Sensitive transcript caching | `Cache-Control: no-store`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer` |

## Implemented Controls

- The FastAPI app disables generated docs and OpenAPI routes.
- Every route, including health and transcript downloads, requires `X-Transcribe-Proxy-Token`.
- The server fails closed at startup if no token is configured.
- Token comparison uses constant-time comparison.
- No CORS middleware is added; the homepage proxy should provide same-origin access.
- File uploads are limited by configurable byte count, defaulting to 10 GB.
- Upload filenames are not used for storage paths; the server stores each upload with the UUID job ID as its stem inside the UUID job directory.
- Only extensions already supported by the CLI are accepted.
- Supported languages are limited to `auto`, `en`, and `es`.
- Errors use `{ "error": { "code": "...", "message": "...", "details": { ... } } }`; `details` is optional.
- Client errors do not include raw stack traces, local paths, process IDs, or full WhisperX logs.
- Uploaded source media is deleted in a `finally` block after each job.
- Completed artifacts are retained only until the configured TTL.
- Stale UUID-shaped job directories older than the TTL are cleaned after restarts.
- The Windows launcher checks for a token and required Python modules before starting.

## Secret Handling

Use environment variables for secrets:

```powershell
[Environment]::SetEnvironmentVariable("TRANSCRIBE_PROXY_TOKEN", "<long-random-token>", "User")
```

Avoid passing `--proxy-token` except for short manual tests. Command-line arguments can be visible to local users and process inspection tools.

Do not commit real values in:

- `.env`
- `.env.local`
- `.env.*.local`
- `HF_TOKEN.txt`
- `hf_token.txt`
- certificate/private-key files

The repo includes `.env.example` with placeholders only.

## Network Hardening

Recommended production/LAN posture:

1. Bind the worker to `0.0.0.0:8092` only on a trusted private LAN.
2. Add a Windows Firewall inbound allow rule for TCP 8092 scoped to the homepage PC LAN IP.
3. Do not add a broad allow rule for all private network clients.
4. Keep the homepage proxy admin-authenticated.
5. Rotate `TRANSCRIBE_PROXY_TOKEN` if it is pasted into chat, logs, shell history, process arguments, or committed files.

The worker speaks plain HTTP on the LAN. If the LAN is not trusted, put the worker behind a TLS-terminating proxy or a VPN and keep the firewall scope narrow.

## Residual Risks

- The bearer token is sufficient to upload files and retrieve transcripts. Anyone with the token and network access can use the worker.
- Local administrators can inspect process environments and worker job artifacts.
- Large uploads still consume disk and CPU/GPU resources. Keep `TRANSCRIBE_MAX_WORKERS` low and set `TRANSCRIBE_MAX_UPLOAD_BYTES` to the smallest practical value.
- WhisperX and diarization dependencies load third-party model artifacts. Treat those dependencies and model sources as trusted code.
- Job state is in memory. Restarting the worker drops status knowledge for existing jobs, though old job directories are still cleaned by TTL.

## Security Review Checklist

Before exposing the worker to the homepage:

- [ ] Worker runs from the venv that contains WhisperX and server extras.
- [ ] `TRANSCRIBE_PROXY_TOKEN` is set in the Windows user environment.
- [ ] The token is not passed on the process command line.
- [ ] Worker listens on `0.0.0.0:8092` only when LAN access is needed.
- [ ] Windows Firewall allows TCP 8092 only from the homepage PC LAN IP.
- [ ] Homepage `.env` uses the same token and current worker IP.
- [ ] `GET /api/transcriptions/health` returns 401 without the token.
- [ ] Authenticated health returns 200 from both worker and homepage PCs.
- [ ] `.env`, token files, logs, and job artifacts are not staged in git.
