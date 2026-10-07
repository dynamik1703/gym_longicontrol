# Projected-goal Contrastive RL

This directory contains the frozen design and execution infrastructure for a
Contrastive Reinforcement Learning (CRL) depth study. Here CRL means
contrastive RL, not the earlier constrained-RL benchmark. The six main runs
remain disabled until a separate explicit authorization.

The source-faithful core contains residual actor/state-action/goal networks,
negative-Euclidean association scores, diagonal-positive InfoNCE, entropy
temperature learning, episode-safe discounted-future sampling, and save/load.
LongiControl-specific raw outcomes and their requirement-aware projection live
in separate adapter modules.

The canonical task is a **first-arrival requirement set**:

```text
previous_position < 1000 m <= current_position
and absolute_episode_time <= 140 s
and cumulative_max_speed_violation <= 0 m/s
```

The fixed network command is `[1,1,1]`: completed progress, within deadline,
and compliant so far. Raw values remain in replay/state, and domain validation
proves equivalence to canonical first arrival for valid physical transitions.
This projection is task engineering, not a scalar reward or an unchanged paper
representation. The score remains an uncalibrated association surrogate.

## What was checked

- Primary-source audit and pinned upstream revision: [SOURCE_AUDIT.md](SOURCE_AUDIT.md)
- Mapping decision and equivalence argument: [DESIGN.md](DESIGN.md)
- Frozen pre-training depth protocol: [PROTOCOL.md](PROTOCOL.md)
- Bounded CPU/JAX measurements: [RESOURCE_REPORT.md](RESOURCE_REPORT.md)
- Machine-readable readiness: [preparation_status.json](preparation_status.json)
- Execution/checkpoint contract: [EXECUTION.md](EXECUTION.md)
- Machine-readable run schema: [execution_schema.json](execution_schema.json)

The optional JAX stack is isolated in requirements-reference.txt; it is not a
dependency of the public environment package. External tracking and telemetry
are absent. runner.py now provides the exact collector/replay/update loop,
atomic full-state checkpoints, diagnostics, split gates, and duplicate-run
protection. Readiness does not authorize execution:
main_training_authorized and main_training_enabled remain false, and no
LongiControl policy was trained.

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
