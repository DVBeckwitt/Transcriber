# ADR-004: Use Local Server Post-Translation for File English Output

## Status
Accepted

## Date
2026-06-08

## Context
The file transcription pipeline needed a visible English conversion setting anywhere users choose normal transcription settings. The existing `--translate-to-english` path asks WhisperX to translate directly, which preserves compatibility but can be less accurate for Spanish/German subtitle workflows because source-language transcription and cleanup are skipped.

The new behavior must:
- Keep legacy direct WhisperX English output available.
- Preserve Spanish/German source subtitles when post-translation is selected.
- Support maximum-accuracy English output for generated Spanish and German SRTs.
- Avoid adding heavyweight translation model imports to the base package.
- Avoid accidentally sending transcript text to remote services.

## Decision
Add explicit file English output modes:

- `off`: keep generated files in the detected/source language.
- `direct`: use the legacy WhisperX translate task.
- `post`: transcribe Spanish/German source first, then translate generated SRT output to English.
- `auto`: post-translate Spanish/German when available, skip English, and keep unsupported or failed post-translation output in the source language with a warning.

Post-translation uses an OpenAI-compatible local server backend over Python standard-library HTTP. The server URL must be localhost or loopback by default. When no server URL is provided, the launcher starts `vllm serve <translation model>` on `127.0.0.1`, waits for the OpenAI-compatible `/v1/models` endpoint, uses that server for subtitle translation, and terminates only that owned child process when translation finishes. Native Windows `vllm` is tried first; on Windows, default-distro WSL2 `vllm` is the fallback when native vLLM is unavailable. The default local model is `utter-project/EuroLLM-1.7B-Instruct` so 12 GB local GPUs can run the automatic path; larger models remain opt-in through `--translation-model`.

Post-translation defaults to one subtitle cue per request and a 1024 generated-token cap because the local 1.7B model is more reliable with smaller requests and enough room to finish JSON output. `--translation-batch-size` and `--translation-max-new-tokens` remain tuning knobs for local hardware and model changes. Malformed model indexes can recover by response order when cue counts still match, and malformed multi-cue batches retry as single-cue requests before failing. After valid JSON parsing, the translator runs a quality gate for obvious residual Spanish/German text and dropped cue content. Flagged cues retry once individually. Explicit `post` fails if the quality gate still fails; `auto` preserves source-language output with a warning. Translation reports include backend/model, cue count, batch size, token cap, warnings, fallback count, selected English output mode, and quality-check metadata without transcript text.

Explicit `post` mode is strict: server startup failure or translation failure marks the run failed because the user requested English output. `auto` mode is fallback-friendly: unavailable post-translation keeps source output and warns.

## Alternatives Considered

### Keep Only WhisperX Direct Translation
- Pros: No new backend, dependency, or output naming.
- Cons: Does not preserve source-language subtitle generation before English conversion and does not meet maximum-accuracy Spanish/German workflow requirements.
- Rejected: Direct mode remains available, but it is not sufficient as the only English conversion mechanism.

### Use In-Process Transformers Post-Translation
- Pros: No server process.
- Cons: Adds heavy model imports and runtime dependencies to the file pipeline, repeats the old helper path, and makes German support harder to keep clean.
- Rejected: The base package should stay lightweight and unit-testable without model downloads.

### Allow Any OpenAI-Compatible URL
- Pros: Supports hosted translation providers.
- Cons: Transcript text can be sensitive, and a permissive URL would make accidental off-machine exfiltration easy.
- Rejected: Local-only is the default security posture. Remote providers require a future explicit opt-in decision.

## Consequences
- Users get a normal settings-flow English conversion choice instead of relying on hidden CLI-only flags.
- Spanish/German source SRTs are preserved beside English outputs in post mode.
- `--translate-to-english` remains backward compatible as direct WhisperX translation; no migration is required.
- Post-translation no longer requires an HTTP client extra. Automatic server startup requires a compatible `vllm` executable in the active Python environment, on `PATH`, or in the default WSL2 distribution on Windows, unless the user supplies `--translation-server-url`.
- The WSL2 fallback is additive and advisory: native vLLM remains preferred, existing `--translation-server-url` usage is unchanged, and no CLI migration or deprecation is required.
- Removing the `translation-server` extra is not a user migration because no replacement package install is required for stdlib HTTP; users only need vLLM or an existing local server for server-backed translation.
- Smaller default batches trade speed for reliability. Users who accept higher risk for faster local translation can raise `--translation-batch-size`; users with responses that hit the generation cap can raise `--translation-max-new-tokens`.
- Bad English candidates are no longer promoted when the quality gate detects unrecovered source-language or dropped-content cues.
- CI remains model-free: tests use fake backends and fake HTTP clients.
- Rollback is a normal git revert; no data migration is required.
