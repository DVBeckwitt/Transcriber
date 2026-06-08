# Transcriber

Local WhisperX launcher for fast transcription from audio/video files.
It can transcribe first, convert generated Spanish/German files to English through a local OpenAI-compatible translation server, or use WhisperX direct English translation for legacy Spanish workflows.

It generates:
- `your_file.srt` subtitle transcript
- `your_file_llm.txt` clean text block for LLM workflows
- optional `your_file.es.srt` / `your_file.de.srt` source subtitles plus `your_file.en.srt` English subtitles when post-translation is enabled
- optional `your_file.translation.json` audit metadata for post-translation runs
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

Local post-translation uses Python's standard-library HTTP client to talk to an OpenAI-compatible localhost server. No HTTP client extra is required.

With `uv`, use a Python 3.11+ environment and install optional extras as needed:

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".uv-venv"
uv sync --extra live
```

When `--english-output-mode post` or `auto` needs post-translation and no `--translation-server-url` is provided, the launcher tries to start `vllm serve` from the active Python environment or `PATH`, then stops the process when the run finishes. On Windows, if native `vllm` is not available, it falls back to `vllm` in the default WSL2 distribution and still exposes the server through localhost. Install vLLM separately in Windows or WSL if you want this automatic server startup path.

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
- File English conversion: ready for release. `--english-output-mode` is visible in the normal settings flow and supports `off`, `direct`, `post`, and `auto`. Server post-translation is local-only by default, auto-starts a local vLLM server when no URL is supplied, falls back to default-distro WSL2 vLLM on Windows when native vLLM is unavailable, preserves source `.es.srt` / `.de.srt`, and writes English `.en.srt` plus compatibility `.srt` when translation succeeds. Post-translation defaults to one subtitle cue per request and a 1024-token generation cap for reliability. It retries residual source-language or dropped-content cues once as single-cue requests, records the quality gate in `your_file.translation.json`, and does not promote failed English candidates. Explicit `post` fails if the local server cannot start, translation fails, or the quality gate still fails; `auto` warns and keeps source output when post-translation is unavailable or rejected. The false preflight failure for missing `httpx` is fixed because post-translation now uses stdlib HTTP. The default local model is `utter-project/EuroLLM-1.7B-Instruct` for local 12 GB GPU compatibility.
- WSL2 post-translation server fallback: ready for release and smoke-tested on 2026-06-08 with WSL vLLM serving `utter-project/EuroLLM-1.7B-Instruct` through the OpenAI-compatible localhost API. Windows native vLLM remains preferred; default-distro WSL2 vLLM is used only when native vLLM is unavailable. No CLI migration is required, and rollback is a normal git revert.
- Live caption mode: startup smoke passed with the `live` extra installed. The latency launcher remains direct English and writes `logs\live_english_transcript.txt` by default. The quality launcher uses cascade mode, writes English and bilingual logs, applies mode-aware Spanish prompts, and prints audio diagnostics. The previous direct-mode bilingual log bug is fixed by rejecting bilingual transcript output unless cascade mode is selected. Full live audio validation with Windows loopback input is still pending.
- Code simplification: accepted. Config preset setup, temporary directory candidate handling, SRT finalization, confidence cleanup, transcript merge collection, speaker-label prompt/config control flow, and live translation-mode helper/parser cleanup were simplified without changing public CLI behavior.
- Generated artifacts: cleaned. Bytecode caches, sample media/log output, build output, and local uv environments are not part of the committed source.
- Release posture: local quality gates, coverage, hook checks, dependency audit, and secret scan pass; deployment is a local CLI/source release. No migration or deprecation is required because direct WhisperX translation and old diarization aliases remain supported. Rollback is `git revert` of the release commit.

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
- Add `transcriber_glossary.txt` in the project folder, or pass `--glossary` / `--glossary-file`, to preserve names and terminology during post-translation.
- Add `--asr-prompt` or `--asr-prompt-file` to bias WhisperX toward names and jargon during transcription.
- Add `--english-output-mode post` to transcribe the source language first, then convert generated Spanish/German output to English.
- Add `--english-output-mode auto` to post-translate Spanish/German, skip English, and keep unsupported languages in the source language.
- Add `--translate-to-english` or `--english-output-mode direct` to use legacy WhisperX translation mode so the `.srt` is written in English directly.
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
python -m transcriber --live --lang es --model small --live-source system --no-speaker-labels --live-preset latency --live-translation-mode direct --live-save-transcript logs\live_english_transcript.txt
```

Accuracy-first live Spanish-to-English system-audio captions:

```powershell
.\live_translate_quality.bat
```

The quality launcher uses cascade mode so WhisperLiveKit keeps Spanish source text and English translation separately. It writes `logs\live_english_transcript.txt`, writes bilingual Spanish/English pairs to `logs\live_bilingual_transcript.txt`, and prints live audio diagnostics.

Quality launcher overrides:

```powershell
set LIVE_MODEL=large-v3
set LIVE_NLLB_SIZE=1.3B
set LIVE_BEAMS=3
```

Use `LIVE_BEAMS=3` first if lag grows during quality mode. If lag still grows, keep `LIVE_NLLB_SIZE=600M` and try `LIVE_MODEL=small`.

Live audio sanity check:

```powershell
python -m transcriber --live-loopback-test --seconds 20 --output loopback_test.wav --live-audio-diagnostics
```

The live stream sent to WhisperLiveKit is 16 kHz signed 16-bit mono PCM, which is 32,000 bytes per second. A 500 ms chunk is about 16,000 bytes, a 750 ms chunk is about 24,000 bytes, and a 1,000 ms chunk is about 32,000 bytes.

Evaluate a committed bilingual live transcript against reference text from the source checkout:

```powershell
python tools/evaluate_live_transcript.py --reference-es reference_es.txt --candidate-bilingual logs\live_bilingual_transcript.txt --reference-en reference_en.txt
```

The evaluator is a dependency-free developer tool that prints Spanish CER/WER and, when `--reference-en` is provided, English CER/WER. It does not require WhisperLiveKit, network, GPU, model downloads, audio hardware, or a Hugging Face token.

Live mode status: implemented and covered by unit tests for CLI dispatch, launcher contracts, PCM conversion, WhisperLiveKit message parsing, direct/cascade transcript semantics, bilingual transcript formatting, quality preset flags, mode-aware prompts, evaluator metrics, window text formatting, dependency diagnostics, executable resolution, and WLK startup cleanup. Startup smoke has passed with the `live` extra installed; full live audio validation still requires Windows audio hardware.

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

Auto-detect source language without forced English conversion:

```powershell
transcriber --input "C:\media\call.wav" --lang auto
```

Maximum-accuracy English output from Spanish or German source subtitles:

```powershell
transcriber --input "C:\media\call.wav" --lang auto --mode quality --english-output-mode post --translation-backend server
```

Reuse an already-running local EuroLLM server:

```powershell
vllm serve utter-project/EuroLLM-1.7B-Instruct
transcriber --input "C:\media\call.wav" --english-output-mode post --translation-backend server --translation-server-url http://localhost:8000/v1
```

If `--translation-server-url` is omitted, post-translation auto-starts `vllm serve <translation model>` on `127.0.0.1`, waits for the OpenAI-compatible `/v1/models` endpoint, and stops that child process when translation finishes. Native Windows `vllm` is tried first. If it is missing on Windows, the launcher checks the default WSL2 distribution for `vllm` and starts it with `wsl -e sh -lc ...`. If `--translation-server-url` is supplied, the launcher reuses that existing server and does not stop it.

The default post-translation model is `utter-project/EuroLLM-1.7B-Instruct`, a small European-language translation model chosen for local 12 GB GPU systems. Override it with `--translation-model` when you want to use another local server model.

For the WSL2 fallback, WSL must be able to start, the default distribution must have `vllm` on its Linux `PATH`, and Windows localhost forwarding must be working. Install vLLM inside WSL with a Linux Python environment; do not install a Linux NVIDIA display driver inside WSL.

`--translation-server-url` accepts localhost or loopback HTTP(S) URLs such as `http://localhost:8000/v1` or `http://127.0.0.1:8000/v1`. Remote URLs are rejected by default so transcript text is not accidentally sent off-machine.

Post-translation defaults are conservative because the local 1.7B model is more reliable with small requests:

```powershell
transcriber --input "C:\media\call.wav" --english-output-mode post --translation-batch-size 1 --translation-max-new-tokens 1024
```

Increasing `--translation-batch-size` can be faster but raises the chance of malformed JSON, dropped items, or untranslated runs. Increasing `--translation-max-new-tokens` gives the model more room to finish a response; it is a cap, not a guarantee that every request will use that many tokens.

German source audio:

```powershell
transcriber --input "C:\media\meeting.mp4" --lang de
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

Record a loopback test with levels:

```powershell
python -m transcriber --live-loopback-test --seconds 10 --output loopback_test.wav --live-audio-diagnostics
```

## Runtime behavior

- Output files are written next to the input media file.
- Logs are written under `<project>\logs\`.
- Before transcription, inputs are normalized into a temporary mono 16 kHz WAV with a light speech-band filter.
- If preprocessing fails, the launcher falls back to the original source file.
- WhisperX receives an initial prompt built from `--asr-prompt`, `--asr-prompt-file`, and glossary terms when present.
- Quality mode uses a fallback temperature schedule by default; fast mode uses a single-pass decode.
- Interactive one-off runs prompt for language, quality/fast mode, and speaker labels unless those choices are already provided with CLI arguments.
- Interactive one-off runs also expose `Convert generated output to English` with `off`, `post`, `direct`, and `auto` choices. Quality mode defaults to `auto`; fast mode defaults to `off`.
- Watch mode monitors the top level of the watched folder for supported media files.
- The default recordings watcher also monitors `%USERPROFILE%\Videos\escuela` for supported video files.
- Watch mode waits for a file to stop changing before transcription starts.
- Watch mode skips files that already have an up-to-date `.srt` next to them.
- When `%USERPROFILE%\Videos\escuela` already has both the video and `.srt`, watch mode will still try to move them to `\\BECKWITT-SERVER\Plex\TV\Escuela de Nada`.
- `%USERPROFILE%\Videos\escuela` files are renamed into `Escuela de Nada - s01e<next> - <translated title>` and tracked with `episode_counter.txt` in the destination folder.
- Default mode presets:
  - `quality`: model `large-v3`, speaker labels and diarization on by default
  - `fast`: model `medium`, speaker labels and diarization off by default
- Language is auto-detected unless you force `--lang en`, `--lang es`, or `--lang de`.
- `--english-output-mode off` keeps generated subtitles/transcripts in the detected or forced source language.
- `--english-output-mode direct` or `--translate-to-english` asks WhisperX to write English subtitle text directly.
- `--english-output-mode post` transcribes Spanish/German source first, writes `your_file.es.srt` or `your_file.de.srt`, then writes English `your_file.en.srt` and compatibility `your_file.srt`.
- In explicit `post` mode, the launcher auto-starts a local vLLM server when no server URL is supplied. On Windows it can start default-distro WSL2 vLLM when native vLLM is unavailable. Startup or translation failure marks the run failed instead of silently producing source-only output.
- `--english-output-mode auto` post-translates Spanish/German when a local server can be reused or auto-started, skips translation for English, and keeps unsupported or failed post-translation output in the source language with a warning.
- Post-translation writes `your_file.translation.json` with backend/model, cue count, batch size, token cap, warnings, selected English mode, and `quality_check` fields. It does not include the transcript text.
- The post-translation quality gate looks for obvious residual Spanish/German text and dropped cue content after the model returns valid JSON. Flagged cues are retried once individually. If they still fail, explicit `post` fails and `auto` keeps source-language output instead of promoting the bad English candidate.
- Local model JSON/index mistakes are recovered when the server still returns exactly one translated text item per input cue; recovery warnings are written to the translation report. Changed cue/item counts fail for single-cue requests and trigger per-cue retry for larger batches, so subtitle timing is kept one-to-one or the run fails.
- If speaker labels are disabled, the launcher skips diarization and writes SRT text without `SPEAKER_00:` prefixes.
- When diarization is enabled, short speaker blips are smoothed by default.
- Low-confidence words are italicized in the `.srt` output and shown with confidence percentages in `*_llm.txt`.
- If diarization fails due token/access issues, the launcher can retry without diarization and continue transcription.
- Live mode is separate from file/watch transcription. It captures Windows PC speaker output through WASAPI loopback, converts it to 16 kHz mono signed 16-bit PCM, streams it to a local WhisperLiveKit server at `/asr` with the LocalAgreement backend policy, and displays committed captions plus one replaceable partial line in a small always-on-top Tkinter window.
- `--live-translation-mode direct` uses WhisperLiveKit direct English translation for fastest captions. In this mode WLK committed `text` is already English, so bilingual transcript output is rejected to avoid labeling English as Spanish.
- `--live-translation-mode cascade` uses WhisperLiveKit `--target-language en` so committed `text` is source-language text and `translation` is English. Use this mode when a real source/English transcript is required.
- `--live-preset quality` defaults to cascade mode, model `medium`, the Faster-Whisper backend, beam decoding, CTranslate2 NLLB translation, and a Spanish-to-English static prompt for the known `casarse` / `hunt` failure mode. Explicit live flags still override preset defaults.
- `--live-static-prompt` is passed to WhisperLiveKit as a static prompt that does not scroll out of context. Existing `--asr-prompt`, `--asr-prompt-file`, and glossary flags still feed the regular initial prompt.
- `--live-audio-diagnostics` prints input sample rate, channel count, output chunk bytes, RMS level, peak level, queue depth, dropped chunk count, queue-delay estimate, and WhisperLiveKit lag when available.
- `--live-audio-min-len` and `--live-audio-max-len` are validated before WhisperLiveKit starts. Max length must be greater than zero, and min length cannot exceed max length.
- Live audio downmixing averages normal stereo channels, but falls back to the stronger channel when one channel is nearly silent or phase cancellation would erase speech. Resampling uses SciPy's polyphase resampler when the live extra is installed, with the previous linear interpolation kept as a fallback.
- The latency preset keeps a bounded queue and may drop the oldest chunk when behind. The quality preset uses an unbounded queue, does not intentionally drop audio, and may show growing lag when the machine cannot keep up.
- Live mode never enables diarization, never renders speaker labels, never loads Hugging Face tokens, and does not write SRT or `*_llm.txt` files.
- `--live-save-transcript` can write the current committed English caption lines to a text file while live mode runs.
- `--live-save-bilingual-transcript` writes committed source-language and English translation pairs to a text file, and requires `--live-translation-mode cascade`. The evaluator remains Spanish-specific and expects `ES:` / `EN:` pairs.
- Closing the caption window or pressing `Ctrl+C` stops capture and terminates the local WhisperLiveKit subprocess.

## CLI options

```text
--input, -i        Path to audio/video file
--lang             auto | en | es | de
--glossary         Glossary entry: 'source=target' or 'source' to preserve (repeatable)
--glossary-file    Glossary text file (one entry per line)
--asr-prompt       Optional text prompt to bias WhisperX toward names and jargon
--asr-prompt-file  Text file with extra ASR prompt lines
--english-output-mode  off | direct | post | auto
--post-translate-to-english  Shortcut for --english-output-mode post
--translation-backend  server
--translation-model    Model name for the local translation server (default: utter-project/EuroLLM-1.7B-Instruct)
--translation-server-url  Optional OpenAI-compatible localhost/loopback server base URL; omitted means auto-start vLLM
--translation-batch-size  Subtitle cues per post-translation request (default: 1)
--translation-max-new-tokens  Generated-token cap per post-translation request (default: 1024)
--save-source-srt / --no-save-source-srt
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
--live-chunk-ms            Override live audio chunk size in milliseconds
--live-translation-mode    direct | cascade
--live-preset              latency | quality | custom
--live-backend             auto | faster-whisper | whisper
--live-backend-policy      localagreement | simulstreaming
--live-frame-threshold     AlignAtt frame threshold
--live-beams               Beam search width
--live-decoder             auto | greedy | beam
--live-audio-min-len       Minimum audio length to process in seconds
--live-audio-max-len       Maximum audio buffer length in seconds
--live-nllb-backend        transformers | ctranslate2
--live-nllb-size           600M | 1.3B
--live-static-prompt       Static live prompt passed to WhisperLiveKit
--live-audio-diagnostics   Print live audio levels, queue state, and lag
--live-no-window           Run live mode without the caption popup
--live-save-transcript     Write committed live captions to a text file
--live-save-bilingual-transcript  Write committed source/English live caption pairs to a text file
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
