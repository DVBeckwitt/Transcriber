# Security Policy

## Supported Scope

This repository ships a local command-line transcription launcher and helper scripts. Security support covers the current `main` branch and the latest release branch or commit used by the repository owner.

## Reporting A Vulnerability

Use GitHub private vulnerability reporting when it is available for this repository. If private reporting is unavailable, contact the repository owner directly before opening a public issue.

Do not include tokens, local media, logs, transcripts, or other private data in a public issue or pull request.

## Secrets And Local Data

Never commit:

- `HF_TOKEN.txt`, `.env`, or any real token file.
- `logs/`, `*.log`, `.srt`, `*_llm.txt`, or transcription outputs.
- Local input media, virtual environments, bytecode caches, `build/`, `dist/`, or `*.egg-info/`.

Use `HF_TOKEN.example.txt` as the safe template for local setup.

## Dependency And CI Checks

CI runs tests, lint, formatting, type checking, CLI startup, package build, and a Python dependency vulnerability audit. Dependency update pull requests are managed through Dependabot.

If a dependency audit fails:

1. Read the advisory and affected package.
2. Prefer upgrading through `uv lock --upgrade-package <name>` or the smallest safe lockfile update.
3. Rerun the full validation gate from `AGENTS.md`.
4. Document any deferred advisory and why it is not exploitable in this local CLI context.
