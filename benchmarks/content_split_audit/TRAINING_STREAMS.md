# Training-stream reconstruction

No stream below executes a simulator transition. `StochasticTrackGenerator` is called
directly with the exact NumPy generator state used by reset.

## Critical streams

`T_HIST` starts `default_rng(11)` and retains reset draws `0,2,4,...,998`. Only episode
0 has an explicit numeric reset seed (`11`); later resets pass `None`, so rows record
the RNG draw index rather than inventing a numeric seed.

`T_CURRENT` uses `default_rng(SeedSequence([11, 0x4D564531]))`, draws explicit integers
from `[0, 2**31-1)`, and rejects all reserved numeric split IDs. The first 500 accepted
seeds are stored in `results.json` with episode index and both hashes.

Five hundred tasks exceed the observed completed/current task counts for the stopped
Seed-11 streams (historical 249 completed episodes; current matched Real-Only 223
instantiated reset seeds). It also covers the known collision at current episode 242
and is the preregistered conservative prefix used by this audit.

## Completed studies

| Study | Reconstruction | Audited task counts | Basis / limitation |
|---|---:|---:|---|
| Historical Constrained V2 | PARTIAL | 249 / 240 / 245 | Source-exact current-runtime draws `0,2,4,...`; original run commit and NumPy/Gymnasium/Torch versions were not recorded, so these are not recovered historical bytes. |
| Current MVE matched Real-Only | EXACT | 223 / 231 / 241 | Explicit `SeedSequence([seed, 0x4D564531])`; final checkpoints retain reset seeds and audit evidence records counts. |
| H=1 Physics MBRL | EXACT | 250 / 234 / 239 | Explicit `SeedSequence([seed, 0x4D42524C])`; completed real-episode counts from final heartbeats. |
| Requirement-Conditioned RL | PARTIAL | 250 / 259 / 271 | Current-runtime used draws `1,4,7,...` and independent balanced margins; FSRL/Tianshou are recorded, but exact NumPy/Gymnasium versions and realized tracks are not. |
| Matched scalar SAC | PARTIAL | 185 / 170 / 178 | Continuous draws `0,1,2,...` through the final active episode; SB3 and counters exist, exact NumPy/Gymnasium provenance does not. |
| HER | EXACT | 170 / 170 / 167 | Frozen sources and complete dependency versions; continuous draws through recorded partial final episodes. |
| CRL | EXACT | 301/278/297 (depth 4), 291/281/262 (depth 16) | Explicit PCG64 `SeedSequence([seed, 0x43524C31])`; manifest counts and checkpoints retain track seeds. |
| Binary Success | PARTIAL | 207 / 179 / 218 | Completed episode counts exist; exact NumPy/Gymnasium runtime and realized tracks do not. |
| LLM Reward final | EXACT | 260 / 167 / 167 | Frozen reward/source and full runtime versions; saved SB3 episode counters include the active final task. |

Counts are tasks that received at least one training transition. A reset performed only
after the final transition is not treated as training exposure.

`EXACT` means the content stream and used length are deterministically recoverable from
frozen sources/runtime plus recorded counters or explicit seed streams. `PARTIAL` means
the current-source reconstruction is informative but is not asserted to be the exact
historical byte stream. No stream was labeled exact solely because its nominal seed was
known.
