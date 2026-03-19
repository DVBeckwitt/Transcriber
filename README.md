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

## Install

1. Create/activate a virtual environment.
2. Install WhisperX in that environment.
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

Optional:
- Add `transcriber_glossary.txt` in the project folder, or pass `--glossary` / `--glossary-file`, to preserve names and terminology during Spanish-to-English translation.
- Add `--asr-prompt` or `--asr-prompt-file` to bias WhisperX toward names and jargon during transcription.
- Add `--translate-to-english` to use WhisperX translation mode so the `.srt` is written in English directly instead of generating a Spanish SRT first.
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
- Logs are written under `<project>\logs\`.
- Before transcription, inputs are normalized into a temporary mono 16 kHz WAV with a light speech-band filter.
- If preprocessing fails, the launcher falls back to the original source file.
- WhisperX receives an initial prompt built from `--asr-prompt`, `--asr-prompt-file`, and glossary terms when present.
- Quality mode uses a fallback temperature schedule by default; fast mode uses a single-pass decode.
- Watch mode monitors the top level of the watched folder for supported media files.
- Watch mode waits for a file to stop changing before transcription starts.
- Watch mode skips files that already have an up-to-date `.srt` next to them.
- Default mode presets:
  - `quality`: model `large-v3`, diarization on by default
  - `fast`: model `medium`, diarization off by default
- Language is auto-detected unless you force `--lang en` or `--lang es`.
- If you pass `--translate-to-english`, WhisperX writes English subtitle text directly.
- If the detected language is Spanish and `--translate-to-english` is not set, the launcher falls back to the existing post-translation step.
- When diarization is enabled, short speaker blips are smoothed by default.
- Low-confidence words are italicized in the `.srt` output and shown with confidence percentages in `*_llm.txt`.
- If diarization fails due token/access issues, the launcher can retry without diarization and continue transcription.

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
--diarize          Force diarization on
--no-diarize       Force diarization off
--no-diarize-smoothing     Disable speaker diarization smoothing
--min-speaker-turn-ms      Minimum speaker turn duration for smoothing
--min-speaker-turn-tokens  Minimum speaker turn size (in tokens) for smoothing
--confidence-cleanup       Enable low-confidence cleanup (default)
--no-confidence-cleanup    Disable low-confidence cleanup
--confidence-cleanup-mode  mark | redact
--low-confidence-logprob   Avg logprob threshold for low confidence
--high-no-speech-prob      No-speech probability threshold
--low-confidence-word-prob Word confidence threshold
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
  - Check `logs\\*_whisperx.log` in the project folder for the underlying error.
