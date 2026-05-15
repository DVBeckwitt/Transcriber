# Transcriber

Local WhisperX launcher for fast transcription from audio/video files.
It can transcribe first, or ask WhisperX to translate directly into English output when you want an English SRT from Spanish audio.

It generates:
- `your_file.srt` subtitle transcript
- `your_file_llm.txt` clean text block for LLM workflows
- `logs/your_file_whisperx.log` full run log in the project folder
- `logs/transcriber-watcher.log` watch-mode activity log in the project folder

## What you need before running

- Python 3.10+
- A working WhisperX install in your virtual environment
- `ffmpeg` available on PATH
- Optional but recommended: NVIDIA GPU + CUDA-compatible PyTorch for speed
- Hugging Face token if you want speaker diarization
- Python 3.11+ plus the `live` extra if you want Windows live system-audio captions through WhisperLiveKit

## Install

1. Create/activate a virtual environment.
2. Install WhisperX in that environment.
3. Install this launcher:

```powershell
pip install -e .
```

Live caption mode has heavier optional dependencies and requires Python 3.11+:

```powershell
pip install -e ".[live]"
```

With `uv`, use a Python 3.11+ environment and install the optional live extra:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".uv-venv"
uv sync --extra live
```

## Development

Use `uv` for repeatable local checks:

```powershell
make validate
```

If `make` is not available on your Windows shell, run the underlying commands directly:

```powershell
uv sync
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python -m transcriber --help
uv build
uv run --with pip-audit pip-audit .
uv run pre-commit run --all-files
```

If the project `.venv` is locked by OneDrive or another process on Windows, point uv at a disposable local environment first:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".uv-venv"
uv sync
```

Project workflow and governance:
- `AGENTS.md` maps the repo for agents and maintainers.
- `CONTRIBUTING.md` defines branch, review, validation, and rollback expectations.
- `SECURITY.md` defines vulnerability reporting, secret handling, and dependency audit policy.
- `docs/architecture.md` maps runtime boundaries and change locations.
- `docs/decisions/` records durable decisions.

## Current status

- Agentic legibility: accepted. The repo now has `pyproject.toml` tool config, `uv.lock`, CI, generated-file ignore rules, and `AGENTS.md`.
- Governance: accepted. Contributor workflow, PR checklist, issue templates, CODEOWNERS, security policy, Dependabot, and architecture docs are in place.
- A+ hardening: shipped on 2026-05-11. `Makefile` validation, coverage gate, pre-commit hooks, and Gitleaks secret scanning are in place. This is a tooling/governance feature; public CLI behavior is unchanged.
- Watcher move bug: fixed. Moving completed watcher outputs now checks for the `.srt` before moving media and rolls the media file back if the `.srt` move fails.
- Transcript merge security bug: fixed. `merge_transcripts.py` skips Hugging Face token files case-insensitively and avoids generated/cache directories.
- Speaker label option: ready for release. The interactive CLI prompts for speaker labels after language and quality/fast choices. `--speaker-labels` and `--no-speaker-labels` still control whether SRT output includes `SPEAKER_00:` style labels; `--no-speaker-labels` skips diarization and Hugging Face token loading. Existing `--diarize` and `--no-diarize` flags remain supported aliases.
- Live caption mode: startup smoke passed with the `live` extra installed. The launcher now resolves the active Python environment's WhisperLiveKit executable before PATH, starts WLK with the LocalAgreement backend policy, avoids the observed SimulStreaming `StorageView` encoder crash, and writes committed Spanish/English caption pairs to `logs\live_bilingual_transcript.txt` by default. Full live audio validation with Windows loopback input is still pending.
- Code simplification: accepted. Config preset setup, temporary directory candidate handling, SRT finalization, confidence cleanup, transcript merge collection, and speaker-label prompt/config control flow were simplified without changing public CLI behavior.
- Generated artifacts: cleaned. Bytecode caches, sample media/log output, build output, and local uv environments are not part of the committed source.
- Release posture: local quality gates, coverage, hook checks, dependency audit, and secret scan pass; deployment is a local CLI/source release. No migration or deprecation is required. Rollback is `git revert` of the release commit.

## Hugging Face token (for speaker labels)

Diarization is only needed when SRT speaker labels are enabled. It requires a token and accepted model terms.

1. Copy `HF_TOKEN.example.txt` to `HF_TOKEN.txt`.
2. Replace the placeholder with your own Hugging Face token.
3. Keep `HF_TOKEN.txt` local (it is git-ignored).

Alternative:
- Set `HF_TOKEN` as an environment variable instead of using a file.

Important:
- Accept terms with the same Hugging Face account used by your token:
  - https://hf.co/pyannote/speaker-diarization-3.1
  - https://hf.co/pyannote/segmentation-3.0

Optional:
- Add `transcriber_glossary.txt` in the project folder, or pass `--glossary` / `--glossary-file`, to preserve names and terminology during Spanish-to-English translation.
- Add `--asr-prompt` or `--asr-prompt-file` to bias WhisperX toward names and jargon during transcription.
- Add `--translate-to-english` to use WhisperX translation mode so the `.srt` is written in English directly instead of generating a Spanish SRT first.
- Add `--no-speaker-labels` when you do not want `SPEAKER_00:` style labels in the SRT; this also skips diarization and the Hugging Face token requirement.
- Use `--temperature-schedule`, `--best-of`, `--logprob-threshold`, and related options to control decode fallbacks.

## Run

### Windows batch launcher

```powershell
.\transcribe.bat
```

Notes:
- This is the manual one-off workflow from before.
- If `--input` is not provided, it opens a file picker.
- If file picker is unavailable, it falls back to terminal input.
- Watch mode is optional and only starts when you use `--watch`, `.\watch_recordings.bat`, or `.\start_recordings_watcher.bat`.
- `transcribe.bat` looks for Python at:
  - `%VIRTUAL_ENV%\Scripts\python.exe` (if `VIRTUAL_ENV` is set), else
  - `%USERPROFILE%\.venv\Scripts\python.exe`

Live Spanish-to-English system-audio captions:

```powershell
.\live_translate.bat
```

The live launcher changes to the repository root before starting, uses `VIRTUAL_ENV` when active, then falls back to repo-local `.uv-venv`, repo-local `.venv`, or `%USERPROFILE%\.venv`. It is a thin wrapper around:

```powershell
python -m transcriber --live --lang es --model small --live-source system --no-speaker-labels --live-save-bilingual-transcript logs\live_bilingual_transcript.txt
```

Live mode status: implemented and covered by unit tests for CLI dispatch, PCM conversion, WhisperLiveKit message parsing, bilingual transcript formatting, window text formatting, dependency diagnostics, executable resolution, and WLK startup cleanup. Startup smoke has passed with the `live` extra installed; full live audio validation still requires Windows audio hardware.

### Watch `C:\Users\Kenpo\OneDrive\recordings`

Foreground watcher:

```powershell
.\watch_recordings.bat
```

Hidden background watcher:

```powershell
.\start_recordings_watcher.bat
```

Stop the hidden background watcher:

```powershell
.\stop_recordings_watcher.bat
```

Notes:
- Both launchers watch `%USERPROFILE%\OneDrive\recordings`.
- They also watch `%USERPROFILE%\Videos\escuela` for supported video files.
- Files from `%USERPROFILE%\Videos\escuela` are treated as Spanish, written as translated English `.srt` subtitles only without diarization speaker names, renamed to `Escuela de Nada - s01e<next> - <translated title>`, and then the source video plus `.srt` are moved to `\\BECKWITT-SERVER\Plex\TV\Escuela de Nada`.
- The destination folder keeps `episode_counter.txt` with the last assigned episode number. If the file is missing, the watcher starts from episode `729` and uses the next available episode number.
- They default to auto language detection in `quality` mode.
- Watch activity is appended to `<project>\logs\transcriber-watcher.log`.
- `.\watch_recordings.bat` runs in the foreground, so stop it with `Ctrl+C`.

### CLI/module

```powershell
python -m transcriber --input "C:\path\to\audio.mp3"
```

or:

```powershell
transcriber --input "C:\path\to\audio.mp3"
```

Manual file-picker flow from the CLI:

```powershell
transcriber
```

This prompts for language, quality/fast mode, and whether to add speaker labels before transcription starts.

Watch mode from the CLI:

```powershell
transcriber --watch --watch-dir "C:\Users\Kenpo\OneDrive\recordings" --lang auto --mode quality
```

## Common command examples

Quality mode:

```powershell
transcriber --input "C:\media\meeting.mp4" --mode quality
```

Fast mode:

```powershell
transcriber --input "C:\media\meeting.mp4" --mode fast
```

Spanish source audio with direct English SRT output:

```powershell
transcriber --input "C:\media\call.wav" --lang es --translate-to-english
```

Spanish audio translated to English:

```powershell
transcriber --input "C:\media\call.wav" --lang auto
```

Write subtitles without speaker labels:

```powershell
transcriber --input "C:\media\meeting.mp4" --no-speaker-labels
```

Watch a folder and wait for files to stop changing before transcribing:

```powershell
transcriber --watch --watch-dir "C:\Users\Kenpo\OneDrive\recordings" --settle-seconds 20
```

List Windows WASAPI loopback devices for live captions:

```powershell
python -m transcriber --live-list-devices
```

Record a short loopback test WAV:

```powershell
python -m transcriber --live-loopback-test --seconds 10 --output loopback_test.wav
```

## Runtime behavior

- Output files are written next to the input media file.
- Logs are written under `<project>\logs\`.
- Before transcription, inputs are normalized into a temporary mono 16 kHz WAV with a light speech-band filter.
- If preprocessing fails, the launcher falls back to the original source file.
- WhisperX receives an initial prompt built from `--asr-prompt`, `--asr-prompt-file`, and glossary terms when present.
- Quality mode uses a fallback temperature schedule by default; fast mode uses a single-pass decode.
- Interactive one-off runs prompt for language, quality/fast mode, and speaker labels unless those choices are already provided with CLI arguments.
- Watch mode monitors the top level of the watched folder for supported media files.
- The default recordings watcher also monitors `%USERPROFILE%\Videos\escuela` for supported video files.
- Watch mode waits for a file to stop changing before transcription starts.
- Watch mode skips files that already have an up-to-date `.srt` next to them.
- When `%USERPROFILE%\Videos\escuela` already has both the video and `.srt`, watch mode will still try to move them to `\\BECKWITT-SERVER\Plex\TV\Escuela de Nada`.
- `%USERPROFILE%\Videos\escuela` files are renamed into `Escuela de Nada - s01e<next> - <translated title>` and tracked with `episode_counter.txt` in the destination folder.
- Default mode presets:
  - `quality`: model `large-v3`, speaker labels and diarization on by default
  - `fast`: model `medium`, speaker labels and diarization off by default
- Language is auto-detected unless you force `--lang en` or `--lang es`.
- If you pass `--translate-to-english`, WhisperX writes English subtitle text directly.
- If the detected language is Spanish and `--translate-to-english` is not set, the launcher falls back to the existing post-translation step.
- If speaker labels are disabled, the launcher skips diarization and writes SRT text without `SPEAKER_00:` prefixes.
- When diarization is enabled, short speaker blips are smoothed by default.
- Low-confidence words are italicized in the `.srt` output and shown with confidence percentages in `*_llm.txt`.
- If diarization fails due token/access issues, the launcher can retry without diarization and continue transcription.
- Live mode is separate from file/watch transcription. It captures Windows PC speaker output through WASAPI loopback, converts it to 16 kHz mono signed 16-bit PCM, streams it to a local WhisperLiveKit server at `/asr` with the LocalAgreement backend policy, requests direct English translation for Spanish speech, and displays committed captions plus one replaceable partial line in a small always-on-top Tkinter window.
- Live mode never enables diarization, never renders speaker labels, never loads Hugging Face tokens, and does not write SRT or `*_llm.txt` files.
- `--live-save-transcript` can write the current committed English caption lines to a text file while live mode runs.
- `--live-save-bilingual-transcript` writes committed Spanish source and English translation pairs to a text file. `live_translate.bat` enables this by default at `logs\live_bilingual_transcript.txt`.
- Closing the caption window or pressing `Ctrl+C` stops capture and terminates the local WhisperLiveKit subprocess.

## CLI options

```text
--input, -i        Path to audio/video file
--lang             auto | en | es
--glossary         Glossary entry: 'source=target' or 'source' to preserve (repeatable)
--glossary-file    Glossary text file (one entry per line)
--asr-prompt       Optional text prompt to bias WhisperX toward names and jargon
--asr-prompt-file  Text file with extra ASR prompt lines
--translate-to-english  Use WhisperX translation mode to write English subtitles directly
--temperature      Single decoding temperature
--temperature-schedule  Comma-separated fallback temperatures
--best-of          Sampling candidates when temperature is above 0
--compression-ratio-threshold  Compression-ratio fallback threshold
--logprob-threshold  Avg logprob fallback threshold
--no-speech-threshold  No-speech fallback threshold
--condition-on-previous-text / --no-condition-on-previous-text
--mode             quality | fast
--model            Override model name
--device           Default: cuda
--compute-type     float16 | float32 | int8 (default: float16)
--watch            Continuously watch a folder for new media files
--watch-dir        Folder to watch (default: %USERPROFILE%\OneDrive\recordings)
--poll-interval    Seconds between folder scans in watch mode
--settle-seconds   Delay after file changes before watch mode starts a run
--speaker-labels   Add diarization speaker labels to SRT output
--no-speaker-labels  Do not add speaker labels; skips diarization
--diarize          Backward-compatible alias for --speaker-labels
--no-diarize       Backward-compatible alias for --no-speaker-labels
--no-diarize-smoothing     Disable speaker diarization smoothing
--min-speaker-turn-ms      Minimum speaker turn duration for smoothing
--min-speaker-turn-tokens  Minimum speaker turn size (in tokens) for smoothing
--confidence-cleanup       Enable low-confidence cleanup (default)
--no-confidence-cleanup    Disable low-confidence cleanup
--confidence-cleanup-mode  mark | redact
--low-confidence-logprob   Avg logprob threshold for low confidence
--high-no-speech-prob      No-speech probability threshold
--low-confidence-word-prob Word confidence threshold
--live                     Stream Windows system audio to live English captions
--live-source              system
--live-list-devices        List WASAPI loopback devices and exit
--live-loopback-test       Record a short loopback test WAV and exit
--live-device-index        Explicit WASAPI loopback device index
--live-port                Local WhisperLiveKit server port
--live-chunk-ms            Live audio chunk size in milliseconds
--live-no-window           Run live mode without the caption popup
--live-save-transcript     Write committed live captions to a text file
--live-save-bilingual-transcript  Write committed Spanish/English live caption pairs to a text file
--live-engine              whisperlivekit
--seconds                  Seconds to record for --live-loopback-test
--output                   WAV path for --live-loopback-test
```

## Troubleshooting

- "Could not find Python at ...\\.venv\\Scripts\\python.exe"
  - Activate your environment first, or set `VIRTUAL_ENV` correctly.

- "Missing Hugging Face token"
  - Add `HF_TOKEN.txt` in repo root or set `HF_TOKEN` env var.

- Diarization blocked/unavailable
  - Accept both pyannote model terms (links above).
  - Confirm the token belongs to that same HF account.

- "Live mode requires Python 3.11 or newer"
  - Create or activate a Python 3.11, 3.12, or 3.13 environment and install `pip install -e ".[live]"`.

- Live mode cannot find WhisperLiveKit
  - Run `uv sync --extra live` from the repository root, or install `pip install -e ".[live]"` in the active environment.
  - The launcher checks the active Python environment's Scripts directory before falling back to `PATH` for `wlk` or `whisperlivekit-server`.

- Live loopback device missing
  - Run `python -m transcriber --live-list-devices`.
  - Pick an explicit device with `--live-device-index`.

- No `.srt` created
  - Check `logs\\*_whisperx.log` in the project folder for the underlying error.
