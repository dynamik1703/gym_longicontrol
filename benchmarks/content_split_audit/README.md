# LongiControl content-split audit

This directory audits procedural task content without training or executing a policy.
It statically constructs tracks from RNG state, fingerprints them, and compares hashes.
No `env.step`, evaluation runner, optimizer, replay buffer, or policy is invoked.

The machine-readable record is [`results.json`](results.json). The static audit was
generated from commit `c728da04e9028a0cbb75d715050c688f19bae1e1` and consumed:

- 0 policy-training transitions;
- 0 LongiControl simulator transitions;
- 0 policy-evaluation episodes.

Repository verification additionally executed 476 simulator state advances in the
pre-existing generic environment/dynamics unit tests (210 contract, 200 seed replay,
60 energy/context, 3 finish/time-limit, 2 direct dynamics, 1 headless subprocess).
None used Development, Validation, or Paper content, and none trained or evaluated a
policy. The static content audit itself remained reset/track-construction only.

Development `2000..2008` and Validation `3000..3008` were instantiated only as static
track definitions. Paper-final `4000..4017` was never instantiated, hashed, or inspected.
Only membership of those numeric IDs in the rejection set was checked.

## Files

- [`FINGERPRINT_SPEC.md`](FINGERPRINT_SPEC.md): canonical route/full-task schemas.
- [`TRAINING_STREAMS.md`](TRAINING_STREAMS.md): reconstruction rules and provenance.
- [`COLLISIONS.md`](COLLISIONS.md): critical and completed-study collision findings.
- [`SPLIT_POLICY.md`](SPLIT_POLICY.md): policies S/F/R, recommendation, and repair design.
- [`RESULTS.md`](RESULTS.md): concise scientific interpretation and verification record.
- [`fingerprints.py`](fingerprints.py): reusable hashes and hard split gate.
- [`streams.py`](streams.py): reset-only stream construction and audit primitives.
- [`generate_results.py`](generate_results.py): deterministic audit generator.

## Reproduce

```bash
python -m pytest -q tests/benchmark/test_content_split_audit.py
python -m benchmarks.content_split_audit.generate_results
```

The generator deliberately contains no code path that constructs a Paper-final track.
Do not extend it to Paper-final seeds while that split remains sealed.
