# Preparation resource report

Measurements were made on macOS arm64 with Python 3.11.9 and PyTorch 2.14.1.
The frozen runner uses CPU even though the host exposes an MPS backend. The
probe performed 90 synthetic batch-256 optimizer updates and 900 fixed-action
transitions on Development seed 2000. The physical transitions were used only
for timing and never entered replay or learning. Validation and paper-final
were not opened.

## Measurements

- Trainable actor parameters: **270,338**.
- Batch-256 update: 5.812 ms mean over 89 timed post-warm-up updates
  (172.0 updates/s).
- Deterministic one-state actor inference: 0.0914 ms mean over 2,000 calls.
- Full-capacity replay allocation: 45,000,000 bytes (42.92 MiB), including the
  checkpointed eligible-source index.
- Vectorized full-replay tuple sampling: 0.145 ms per batch over 500 samples.
- Synthetic full-replay checkpoint: 46,097,296 bytes (43.96 MiB).
- Fixed-action simulator: 4,673.5 native transitions/s over the bounded
  900-transition probe.

## Extrapolation

At the measured rates, 300,000 actor inferences, 7,250 replay samples and 7,250
updates require about 70.6 seconds; 300,000 simulator steps require about 64.2
seconds. Their simple sum is **134.8 seconds per policy** and **404.4 seconds
(6.74 minutes) for three sequential policies**.

These are arithmetic extrapolations, not end-to-end training measurements.
They exclude Development evaluation, diagnostic intervals, checkpoint I/O,
process startup, OS contention and simulator/policy state-distribution effects.
Actual study time will be higher. `resource_measurements.json` preserves the
raw timings and explicitly records that no policy training occurred.
