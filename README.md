# Transcriber

Local WhisperX launcher for fast transcription from audio/video files.

It generates:
- `your_file.srt` (subtitle transcript)
- `your_file_llm.txt` (clean text block for LLM workflows)
- `your_file_whisperx.log` (full run log)
- `transcriber-watcher.log` (watch-mode activity log, when using folder watch)

## What you need before running

- Python 3.10+
- A working WhisperX install in your virtual environment
- `ffmpeg` available on PATH
- Optional but recommended: NVIDIA GPU + CUDA-compatible PyTorch for speed
- Hugging Face token if you want speaker diarization

## Install

1. Create/activate a virtual environment.
2. Install WhisperX in that environment (follow WhisperX's current install instructions).
3. Install this launcher:

```powershell
pip install -e .
```

## Hugging Face token (for diarization)

Diarization requires a token and accepted model terms.

1. Copy `HF_TOKEN.example.txt` to `HF_TOKEN.txt`.
2. Replace the placeholder with your own Hugging Face token.
3. Keep `HF_TOKEN.txt` local (it is git-ignored).

Alternative:
- Set `HF_TOKEN` as an environment variable instead of using a file.

Important:
- Accept terms with the same Hugging Face account used by your token:
  - https://hf.co/pyannote/speaker-diarization-3.1
  - https://hf.co/pyannote/segmentation-3.0

## Run

### Windows batch launcher

```powershell
.\transcribe.bat
```

Notes:
- This is still the manual one-off workflow from before.
- If `--input` is not provided, it opens a file picker.
- If file picker is unavailable, it falls back to terminal input.
- Watch mode is optional and only starts when you use `--watch`, `.\watch_recordings.bat`, or `.\start_recordings_watcher.bat`.
- `transcribe.bat` looks for Python at:
  - `%VIRTUAL_ENV%\Scripts\python.exe` (if `VIRTUAL_ENV` is set), else
  - `%USERPROFILE%\.venv\Scripts\python.exe`

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
- They default to English transcription in `quality` mode.
- Watch activity is appended to `%USERPROFILE%\OneDrive\recordings\transcriber-watcher.log`.
- `.\watch_recordings.bat` runs in the foreground, so stop it with `Ctrl+C`.

### CLI/module

```powershell
python -m transcriber --input "C:\path\to\audio.mp3"
```

or (after install via `pip install -e .`):

```powershell
transcriber --input "C:\path\to\audio.mp3"
```

Manual file-picker flow from the CLI:

```powershell
transcriber
```

Watch mode from the CLI:

```powershell
transcriber --watch --watch-dir "C:\Users\Kenpo\OneDrive\recordings" --lang en --mode quality
```

## Common command examples

Quality mode (best default quality):

```powershell
transcriber --input "C:\media\meeting.mp4" --mode quality
```

Fast mode (lower latency, diarization off by default):

```powershell
transcriber --input "C:\media\meeting.mp4" --mode fast
```

Spanish transcript:

```powershell
transcriber --input "C:\media\call.wav" --lang es --task transcribe
```

Spanish audio translated to English:

```powershell
transcriber --input "C:\media\call.wav" --lang es --task translate
```

Force no diarization:

```powershell
transcriber --input "C:\media\meeting.mp4" --no-diarize
```

Watch a folder and wait for files to stop changing before transcribing:

```powershell
transcriber --watch --watch-dir "C:\Users\Kenpo\OneDrive\recordings" --settle-seconds 20
```

## Runtime behavior

- Output files are written next to the input media file.
- Watch mode monitors the top level of the watched folder for supported media files.
- Watch mode waits for a file to stop changing before transcription starts.
- Watch mode skips files that already have an up-to-date `.srt` next to them.
- Default mode presets:
  - `quality`: model `large-v3`, diarization on by default
  - `fast`: model `medium`, diarization off by default
- If diarization fails due token/access issues, the launcher can retry without diarization and continue transcription.

## CLI options

```text
--input, -i        Path to audio/video file
--lang             en | es
--task             transcribe | translate
--mode             quality | fast
--model            Override model name
--device           Default: cuda
--compute-type     float16 | float32 | int8 (default: float16)
--watch            Continuously watch a folder for new media files
--watch-dir        Folder to watch (default: %USERPROFILE%\OneDrive\recordings)
--poll-interval    Seconds between folder scans in watch mode
--settle-seconds   Delay after file changes before watch mode starts a run
--diarize          Force diarization on
--no-diarize       Force diarization off
```

## Troubleshooting

- "Could not find Python at ...\\.venv\\Scripts\\python.exe"
  - Activate your environment first, or set `VIRTUAL_ENV` correctly.

- "Missing Hugging Face token"
  - Add `HF_TOKEN.txt` in repo root or set `HF_TOKEN` env var.

- Diarization blocked/unavailable
  - Accept both pyannote model terms (links above).
  - Confirm the token belongs to that same HF account.

- No `.srt` created
  - Check `*_whisperx.log` next to your media for the underlying error.
