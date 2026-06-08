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

Post-translation uses an OpenAI-compatible local server backend through the optional `translation-server` extra. The server URL must be localhost or loopback by default. Subtitle cue requests are batched according to the translation batch size, and translation reports include backend/model, cue count, warnings, fallback count, and selected English output mode without transcript text.

Explicit `post` mode is strict: missing server configuration or translation failure marks the run failed because the user requested English output. `auto` mode is fallback-friendly: unavailable post-translation keeps source output and warns.

## Alternatives Considered

### Keep Only WhisperX Direct Translation
- Pros: No new backend, dependency, or output naming.
- Cons: Does not preserve source-language subtitle generation before English conversion and does not meet maximum-accuracy Spanish/German workflow requirements.
- Rejected: Direct mode remains available, but it is not sufficient as the only English conversion mechanism.

### Use In-Process Transformers Post-Translation
- Pros: No server URL configuration.
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
- Post-translation requires users to run a compatible local server and install the optional `translation-server` extra.
- CI remains model-free: tests use fake backends and fake HTTP clients.
- Rollback is a normal git revert; no data migration is required.
