# ADR-003: Use WhisperLiveKit for live system-audio captions

## Status
Accepted

## Date
2026-05-15

## Context
The project needs an optional live mode that captures Windows PC speaker output, translates Spanish speech to English during live transcription, and displays updating captions without changing the existing file transcription or watch pipelines.

The live path has heavier runtime requirements than the base file transcription launcher:

- WhisperLiveKit currently requires Python 3.11+.
- Windows speaker capture requires WASAPI loopback support through PyAudioWPatch.
- Live captions need streaming updates, partial replacement, and a lightweight UI loop.

The base package still needs to stay compatible with its current Python 3.10 target and must not require Windows audio devices, model downloads, GPUs, or Hugging Face tokens in CI.

Live users also need a durable local record of committed Spanish source text next to the English translation so captions can be reviewed after a session.

## Decision
Use WhisperLiveKit as the live ASR and translation engine, launched as a local subprocess and reached through its documented `/asr` WebSocket endpoint with PCM input.

Keep live dependencies in the `live` optional dependency group. Keep the existing WhisperX file/watch pipeline isolated from live mode except for CLI entry points and shared prompt/glossary parsing.

Start WhisperLiveKit with the LocalAgreement backend policy. This keeps the repo on a documented WhisperLiveKit streaming policy while avoiding the current SimulStreaming encoder failure observed with CTranslate2 `StorageView` values on the local Windows CPU environment.

Preserve committed bilingual caption pairs in `CaptionState` and expose `--live-save-bilingual-transcript PATH` as an additive live-mode output. Keep `--live-save-transcript` as English-only, and have `live_translate.bat` write the default direct-mode English log to ignored `logs\live_english_transcript.txt`.

Make live translation mode explicit. Direct mode starts WhisperLiveKit with `--direct-english-translation`; in that mode committed `text` is already English, so bilingual transcript output is rejected. Cascade mode starts WhisperLiveKit with `--target-language en`; in that mode committed `text` is Spanish source text and `translation` is English, so bilingual transcript output is allowed.

The live runtime path is:

```text
WASAPI loopback audio
  -> 16 kHz mono signed 16-bit PCM chunks
  -> local WhisperLiveKit subprocess
  -> ws://127.0.0.1:<port>/asr?language=es&mode=full
  -> CaptionState updates
  -> Tkinter caption window and optional English-only or bilingual committed transcript text
```

## Alternatives Considered

### Direct streaming-policy integration

- Pros: Avoids a local server subprocess.
- Cons: Requires this repo to own more streaming ASR policy, protocol, and lifecycle logic.
- Rejected: WhisperLiveKit is the maintained integration target and already wraps the relevant streaming policies.

### Extending the existing WhisperX file pipeline

- Pros: Reuses existing transcription dependency and configuration concepts.
- Cons: The file pipeline is batch-oriented, writes SRT outputs, optionally uses alignment/diarization/token flows, and is not shaped around low-latency streaming updates.
- Rejected: Live mode should not destabilize file/watch transcription behavior.

### Requiring live dependencies in the base install

- Pros: Simpler installation story for live users.
- Cons: Forces heavier dependencies and Python 3.11+ constraints onto users who only need file transcription.
- Rejected: Live mode remains optional; base compatibility stays Python 3.10+.

### Replacing English-only transcript output

- Pros: Keeps one transcript flag.
- Cons: Breaks the existing `--live-save-transcript` meaning for users and scripts that expect translated English lines only.
- Rejected: The bilingual transcript is additive so the existing CLI output contract stays stable.

## Consequences

- Live mode is additive and does not deprecate existing CLI behavior.
- `--live-save-transcript` remains English-only; `--live-save-bilingual-transcript` writes numbered Spanish and English pairs only in cascade mode.
- `logs\live_english_transcript.txt` and `logs\live_bilingual_transcript.txt` are generated local output, ignored by git, and not part of release artifacts.
- The `live` extra and runtime guard are required for Python 3.11+ live use.
- CI covers live parsing, conversion, CLI, and lifecycle logic without Windows audio hardware or model downloads.
- No deprecation or migration path is required for this feature because the old transcript flag and file/watch pipelines remain unchanged.
- A bounded startup smoke has passed with the `live` extra installed, but manual Windows loopback audio validation is still required before treating live captions as fully released.
- Rollback is git-based: revert the live-mode commit and rerun the validation gate.
