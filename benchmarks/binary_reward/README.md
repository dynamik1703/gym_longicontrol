# Binary Success Reward V1

This separately preregistered benchmark tests the smallest task specification:
SB3 SAC receives zero reward everywhere except a terminal reward of one when the
existing external evaluator declares the fixed 140-second, zero-speed-excess
task feasible.

The public environment and observation are unchanged. SAC configuration, seeds,
splits, budget, and checkpoints match frozen Scalar SB3 SAC. See `PROTOCOL.md`
for the fixed design and decision gates.

```bash
PYTHONPATH=src:. python -m benchmarks.binary_reward.experiment \
  --output-dir runs/binary-reward-20261004 --workers 3 --device cpu
```

No result should be interpreted from binary training return. `results.json` is
produced from deterministic physical evaluation and `training-outcomes.json`
records when the sparse signal was actually observed.

## Frozen result

The completed study reaches **Gate D**: none of 604 completed training episodes
produces a positive reward, and the three final policies achieve 0/27 feasible
Validation episodes. All final policies respect the speed requirement, but none
completes the route within the unchanged 180-second environment horizon. See
[`RESULTS.md`](RESULTS.md) for the full analysis.

Reproduce the derived artifacts from an existing run with:

```bash
PYTHONPATH=src:. python -m benchmarks.binary_reward.analysis \
  runs/binary-reward-20261004 --output benchmarks/binary_reward/results.json
PYTHONPATH=src:. python -m benchmarks.binary_reward.trajectories \
  runs/binary-reward-20261004 \
  --output benchmarks/binary_reward/trajectories.json
PYTHONPATH=src:. python -m benchmarks.binary_reward.plot_results \
  benchmarks/binary_reward/results.json \
  --results-dir runs/binary-reward-20261004 \
  --trajectories benchmarks/binary_reward/trajectories.json \
  --output-dir benchmarks/binary_reward/plots
```
