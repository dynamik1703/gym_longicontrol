# Contrastive RL preparation

This directory prepares, but does not execute, a Contrastive Reinforcement
Learning (CRL) study for LongiControl. Here CRL means contrastive RL, not the
earlier constrained-RL benchmark.

The source-faithful core contains residual actor/state-action/goal networks,
negative-Euclidean association scores, diagonal-positive InfoNCE, entropy
temperature learning, episode-safe discounted-future sampling, and save/load.
LongiControl-specific physical outcomes live in a separate pure adapter.

The canonical task is a **first-arrival requirement set**:

```text
previous_position < 1000 m <= current_position
and absolute_episode_time <= 140 s
and cumulative_max_speed_violation <= 0 m/s
```

This is not a point state with time coordinate 140, and an association score is
not a calibrated success probability or safety guarantee. The exact outcome
representation and set-membership predicate are implemented and tested, but a
principled actor objective for the whole inequality-defined set is unresolved.
Consequently [PROTOCOL.md](PROTOCOL.md) is DRAFT and main training is disabled.

## What was checked

- Primary-source audit and pinned upstream revision: [SOURCE_AUDIT.md](SOURCE_AUDIT.md)
- Mapping decision and open issue: [DESIGN.md](DESIGN.md)
- Future matched-depth protocol: [PROTOCOL.md](PROTOCOL.md)
- Bounded CPU/JAX measurements: [RESOURCE_REPORT.md](RESOURCE_REPORT.md)
- Machine-readable readiness: [preparation_status.json](preparation_status.json)

The optional JAX stack is isolated in `requirements-reference.txt`; it is not a
dependency of the public environment package. External tracking and telemetry
are absent. No results file exists because no LongiControl policy was trained.

## Reproduce preparation checks

```bash
python3.10 -m venv /tmp/longicontrol-crl
/tmp/longicontrol-crl/bin/pip install -r benchmarks/contrastive_rl/requirements-reference.txt
PYTHONPATH=src /tmp/longicontrol-crl/bin/python -m pytest \
  tests/benchmark/test_contrastive_rl_core.py \
  tests/benchmark/test_contrastive_rl_sampling.py \
  tests/benchmark/test_contrastive_rl_goal_adapter.py
```

`resource_check.py` is deliberately bounded to at most 100 synthetic updates
per depth and 2,000 simulator transitions. It has no policy-training entrypoint.
