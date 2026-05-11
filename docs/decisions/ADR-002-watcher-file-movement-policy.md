# ADR-002: Watcher File Movement Policy

## Status
Accepted

## Date
2026-05-11

## Context
Watch mode can process files from multiple folders. The default recordings folder leaves outputs beside the source media. The Escuela folder uses a special policy: video files are treated as Spanish, translated to English `.srt` output without diarization labels, renamed as `Escuela de Nada - s01e<next> - <translated title>`, then moved with the `.srt` to the Plex destination.

The move step is observable behavior. If the source media is moved but the `.srt` is missing or locked, the watcher cannot safely retry from the source folder and may leave a partial result in the destination.

## Decision
Treat completed media and `.srt` files as a logical pair for watcher moves.

- Preflight that the `.srt` exists before moving media.
- Refuse to overwrite an existing destination media or `.srt`.
- Move media first, then `.srt`.
- If the `.srt` move fails after media moved, roll the media file back to the source folder.
- Keep the watcher retry contract: failed move attempts should leave the source media retryable whenever possible.
- Keep Escuela episode numbering in `episode_counter.txt`, while also scanning destination names to avoid already-used episode numbers.

## Alternatives Considered

### Move `.srt` first
- Pros: Avoids media rollback if subtitle move fails.
- Cons: A failed media move can leave subtitles in the destination without matching media.
- Rejected: Destination should not receive half-complete pairs.

### Move media first without rollback
- Pros: Simpler implementation.
- Cons: Breaks retry behavior when the `.srt` move fails.
- Rejected: The watcher tells users it will retry while the source remains.

### Copy then delete source files
- Pros: Easier all-or-nothing behavior across volumes.
- Cons: Higher I/O cost for large media and more cleanup cases.
- Rejected for now: current local workflow favors `shutil.move`; revisit if cross-volume failures become common.

## Consequences
- Failed subtitle moves preserve retryability when rollback succeeds.
- Destination collision checks happen before either file moves.
- Tests must cover happy path, missing `.srt`, failed `.srt` move rollback, Escuela rename, and episode collision behavior.
- Any future watcher destination policy should go through `WatchTarget` instead of adding folder-specific branches inside the loop.
