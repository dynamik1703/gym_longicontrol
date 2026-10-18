# Constrained V2 seed-11 track × collector factorial

This branch prepared the requested single-seed mechanistic 2×2 study and stopped
before training because a hard split-protection gate failed.

The exact reconstructed historical stream and the exact current archived stream both
generate content identical to Development track 2006 before the minimum possible
number of episodes in a 300,000-transition run. The collision is not a reserved seed
ID leak. It is a collision in canonical task/track content caused by the finite
stochastic track generator:

| Stream | Episode index | Reset seed | Reserved match |
|---|---:|---:|---|
| `T_HIST` | 158 | continuous RNG, no episode seed | Development 2006 |
| `T_CURRENT` | 163 | 1418256283 | Development 2006 |
| `T_CURRENT` | 242 | 274583042 | Development 2006 |

At least 167 episodes are required to consume 300,000 transitions with the fixed
1,800-step cap. Therefore the episode-158 and episode-163 collisions are unavoidable
in every full policy, regardless of learned behavior. Filtering either stream would
violate the requirement to use the exact reconstructed stream. Proceeding would
violate the requirement that no Development track occur in training.

No main condition, resource pilot, Development evaluation, Validation evaluation, or
paper-track evaluation was run. LongiControl simulator transitions consumed by this
study: **0**.

Evidence:

- [DESIGN.md](DESIGN.md) defines the intended factorial.
- [PROTOCOL.md](PROTOCOL.md) records the frozen protocol and the failed gate.
- [RESOURCE_REPORT.md](RESOURCE_REPORT.md) records why runtime benchmarking did not
  proceed.
- [RESULTS.md](RESULTS.md) is the human-readable terminal report.
- [results.json](results.json) materializes both first-500 streams, every protected
  reference track, initialization hashes, archive linkage, and collisions.
- [stream_preflight.py](stream_preflight.py) deterministically regenerates the
  machine-readable evidence.

The historical stream remains a source-exact reconstruction under the current
runtime, not recovered historical bytes. The original historical runtime, initial
weights, and exact dependency set remain unavailable.
