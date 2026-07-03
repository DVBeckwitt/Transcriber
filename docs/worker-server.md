# Transcriber LAN Worker Runbook

This runbook covers the optional FastAPI worker owned by this repo. The worker accepts authenticated media uploads from the homepage proxy, runs the existing Transcriber quality-mode pipeline, and exposes transcript downloads.

## Architecture

```text
Browser
  -> homepage app / nginx / caddy
  -> TRANSCRIBE_API_UPSTREAM=http://<worker-lan-ip>:8092
  -> transcriber-server
  -> existing transcriber internals / WhisperX
  -> transcript artifacts in TRANSCRIBE_WORK_DIR
```

The worker does not enable CORS. The homepage should proxy same-origin requests and add `X-Transcribe-Proxy-Token` before forwarding to the worker.

## Worker Requirements

- Python 3.10+
- WhisperX installed in the same virtual environment that runs the server
- FastAPI server extras from this package: `pip install -e ".[server]"`
- `ffmpeg` on `PATH` for preprocessing
- Optional CUDA-compatible PyTorch for GPU acceleration
- Hugging Face token if diarization is enabled by the existing quality preset

Check the selected Python environment:

```powershell
python -c "import whisperx, fastapi, uvicorn, multipart; print('ok')"
```

## Install

Use the same virtual environment for WhisperX, PyTorch, and the worker server extras.

If WhisperX is already installed in `%USERPROFILE%\.venv`:

```powershell
cd "C:\Users\Kenpo\Nextcloud\Git Projects\Transcriber"
.\install_transcriber_server.ps1 -VenvPath "$env:USERPROFILE\.venv"
```

If you want a separate worker venv:

```powershell
cd "C:\Users\Kenpo\Nextcloud\Git Projects\Transcriber"
.\install_transcriber_server.ps1
```

Then install WhisperX and the correct PyTorch build into that same venv before running real transcription jobs.

## Token Setup

Generate one shared random token and set the exact same value on the worker and homepage.

Generate a token:

```powershell
$bytes = New-Object byte[] 32
[System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
-join ($bytes | ForEach-Object { $_.ToString("x2") })
```

Set the token on the worker PC:

```powershell
[Environment]::SetEnvironmentVariable("TRANSCRIBE_PROXY_TOKEN", "<generated-token>", "User")
```

Prefer environment variables for the token. The `--proxy-token` CLI option exists for compatibility, but command-line arguments can be visible to local process inspection tools.

## Worker Configuration

Set persistent Windows user environment values:

```powershell
[Environment]::SetEnvironmentVariable("TRANSCRIBE_SERVER_HOST", "0.0.0.0", "User")
[Environment]::SetEnvironmentVariable("TRANSCRIBE_SERVER_PORT", "8092", "User")
[Environment]::SetEnvironmentVariable("TRANSCRIBE_WORK_DIR", "D:\TranscriberJobs", "User")
```

Useful optional values:

| Variable | Default | Purpose |
| --- | --- | --- |
| `TRANSCRIBE_MAX_UPLOAD_BYTES` | `10737418240` | Upload limit in bytes, default 10 GB |
| `TRANSCRIBE_MAX_WORKERS` | `1` | Concurrent background transcription jobs |
| `TRANSCRIBE_JOB_TTL_SECONDS` | `86400` | Completed job retention, default 24 hours |
| `TRANSCRIBE_DEVICE` | `cuda` | WhisperX device, for example `cuda` or `cpu` |
| `TRANSCRIBE_COMPUTE_TYPE` | `float16` | WhisperX compute type: `float16`, `float32`, or `int8` |

Current scripts also accept equivalent CLI args such as `--host`, `--port`, and `--work-dir`.

## Start The Worker

Foreground start:

```powershell
cd "C:\Users\Kenpo\Nextcloud\Git Projects\Transcriber"
.\run_transcriber_server.bat --host 0.0.0.0 --port 8092
```

Local-only start:

```powershell
.\run_transcriber_server.bat
```

The server fails closed if `TRANSCRIBE_PROXY_TOKEN` is missing. The batch launcher also checks for `whisperx`, `fastapi`, `uvicorn`, and `multipart` in the selected Python environment before starting.

## Find The Worker LAN IP

On the worker PC:

```powershell
ipconfig
```

Use the active adapter IPv4 address on the same subnet as the homepage PC. For example, if the homepage PC is `192.168.0.192/24`, the worker should usually be another `192.168.0.x` address.

## Firewall

Run PowerShell as Administrator on the worker PC. Replace the remote address with the homepage PC LAN IP:

```powershell
New-NetFirewallRule -DisplayName "Transcriber Worker 8092 from homepage" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8092 -RemoteAddress <homepage-lan-ip> -Profile Private
```

Verify the rule:

```powershell
Get-NetFirewallRule -DisplayName "Transcriber Worker 8092 from homepage" | Get-NetFirewallPortFilter
```

## Homepage Configuration

In the homepage `.env`:

```dotenv
TRANSCRIBE_PROXY_TOKEN=<same-token-as-worker>
TRANSCRIBE_API_UPSTREAM=http://<worker-lan-ip>:8092
```

Then recreate the homepage containers so nginx regenerates its proxy config:

```powershell
cd C:\docker\homepage
docker-compose up -d --force-recreate homepage caddy
```

## API Contract

Every request must include:

```text
X-Transcribe-Proxy-Token: <same-token>
```

### Health

```http
GET /api/transcriptions/health
```

Success:

```json
{
  "status": "ok",
  "maxUploadBytes": 10737418240,
  "maxWorkers": 1
}
```

### Create Transcription

```http
POST /api/transcriptions
Content-Type: multipart/form-data
```

Fields:

| Field | Required | Value |
| --- | --- | --- |
| `file` | Yes | Supported audio/video media file |
| `language` | No | `auto`, `en`, or `es`; defaults to `auto` |

Accepted response:

```json
{
  "jobId": "0123456789abcdef0123456789abcdef",
  "status": "queued",
  "statusUrl": "/api/transcriptions/0123456789abcdef0123456789abcdef"
}
```

### Job Status

```http
GET /api/transcriptions/{jobId}
```

Statuses:

| Status | Meaning |
| --- | --- |
| `queued` | Upload accepted, waiting for worker slot |
| `running` | Transcription is in progress |
| `succeeded` | Transcript artifacts are ready |
| `failed` | Job failed; response includes a generic error |

### Downloads

```http
GET /api/transcriptions/{jobId}/transcript.txt
GET /api/transcriptions/{jobId}/transcript.srt
```

Downloads are only available for `succeeded` jobs.

## Error Format

All client-facing errors use:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Safe client-facing message."
  }
}
```

The worker intentionally avoids returning stack traces, local file paths, or WhisperX internals.

## Runtime Behavior

- Uploads are copied into per-job directories under `TRANSCRIBE_WORK_DIR`.
- The uploaded source file is deleted after success or failure.
- Transcript artifacts stay in the job directory until TTL cleanup removes old jobs.
- Cleanup runs opportunistically when API requests arrive.
- The server also removes stale UUID-shaped job directories older than the TTL after restarts.
- Jobs invoke the existing CLI-equivalent internals with quality mode, selected language, configurable `device` and `compute_type`, and hidden rendered speaker labels.
- CUDA jobs use the shared CLI cleanup path: GPU cache flushing runs after transcription and Spanish post-translation attempts, including error paths.
- The worker process keeps job state in memory. After a process restart, old status URLs are not restored, but stale directories are still cleaned by TTL.

## Validation Checklist

On the worker:

```powershell
Get-NetTCPConnection -LocalPort 8092
$token = [Environment]::GetEnvironmentVariable("TRANSCRIBE_PROXY_TOKEN", "User")
Invoke-RestMethod -Uri "http://127.0.0.1:8092/api/transcriptions/health" -Headers @{ "X-Transcribe-Proxy-Token" = $token }
```

From the homepage PC:

```powershell
Test-NetConnection -ComputerName <worker-lan-ip> -Port 8092
curl.exe -H "X-Transcribe-Proxy-Token: <same-token>" http://<worker-lan-ip>:8092/api/transcriptions/health
```

From inside the homepage nginx/container environment, verify the upstream and token header are generated correctly.

## Troubleshooting

### Homepage cannot reach worker

Symptoms:

- `curl http://<worker-ip>:8092/...` times out
- Ping has 100% loss
- nginx container times out connecting to worker

Checks:

1. Confirm the worker IP is on the same subnet as the homepage PC.
2. Confirm the worker is listening on `0.0.0.0:8092`, not only `127.0.0.1:8092`.
3. Confirm Windows Firewall allows TCP 8092 from the homepage PC LAN IP.
4. Confirm the homepage `.env` uses the current worker IP.

### Health returns 401

The token value does not match or the homepage proxy is not forwarding `X-Transcribe-Proxy-Token`.

### Jobs fail immediately

Check that the server is running in the same virtual environment that has WhisperX:

```powershell
python -c "import whisperx; print('whisperx ok')"
```

Also confirm Hugging Face token setup if diarization is enabled.

### Upload rejected

Likely causes:

- Unsupported extension
- `language` is not `auto`, `en`, or `es`
- Upload exceeds `TRANSCRIBE_MAX_UPLOAD_BYTES`
- Missing multipart `file` field

### No transcript download

Poll `GET /api/transcriptions/{jobId}` until `status` is `succeeded`. Downloads return `JOB_NOT_READY` while a job is queued, running, or failed.
