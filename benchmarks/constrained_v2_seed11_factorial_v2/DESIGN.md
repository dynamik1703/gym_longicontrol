# Design

## Intended scientific question

The preregistered single-seed 2×2 experiment would have separated the effects of
the exact reconstructed training-track stream and collector/update schedule under a
single current runtime and byte-identical learner initialization. It was not a
population-level algorithm comparison.

| ID | Frozen name | Track stream | Collector |
|---|---|---|---|
| A | `A_histTrack_histCollector_seed11` | `T_HIST` | `C_HIST` |
| B | `B_histTrack_currentCollector_seed11` | `T_HIST` | `C_CURRENT` |
| C | `C_currentTrack_histCollector_seed11` | `T_CURRENT` | `C_HIST` |
| D | `D_currentTrack_currentCollector_seed11` | `T_CURRENT` | `C_CURRENT` |

The primary endpoint would have been 300k diagnostic RSR on seeds
`9000000–9000017`, with six 50k-spaced diagnostic checkpoints and no legacy
Development, legacy Validation, or Paper evaluation.

## Exact streams

`T_HIST` uses continuous environment RNG seeded 11 and the historical used reset
draws `0, 2, 4, ...`. `T_CURRENT` uses
`default_rng(SeedSequence([11, 0x4D564531]))` and one explicit reset seed per
episode, rejecting the established legacy reserved numeric IDs. Neither stream is
filtered for content.

Both reconstructed first-100 prefixes match the completed reproducibility audit.
The ordered 865-row content-fingerprint hashes are frozen in `results.json`.

## Exact collector definitions

`C_HIST` would hold actor parameters fixed for the complete physical episode. Every
transition, including the terminal transition, enters replay during collection. On
return, the exact two-component undiscounted episodic cost is passed to the PID
update; all accumulated replay/gradient updates then execute; only then may the next
episode use the updated actor. Historical `FastCollector` reset consumption causes
the used continuous-RNG draws to be `0, 2, 4, ...`.

`C_CURRENT` would collect and insert one transition, perform the terminal PID update
when applicable, and then catch gradient updates up to
`floor(real_transitions × 0.1)`. The next transition may therefore use an updated
actor. A terminal transition is in replay before both PID handling and update
catch-up.

## Terminal design result

The route-clean prerequisite fails before collector implementation, learner
construction, runtime benchmarking, recovery setup, or training. No factorial
contrast exists.
