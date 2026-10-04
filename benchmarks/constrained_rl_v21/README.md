# Constrained RL V2 post-hoc diagnosis

This directory diagnoses the six strict speed failures of the frozen
Constrained RL V2 study. It does not alter V2 and does not contain a newly
trained V2.1 policy. The result is Gate F: two different physical mechanisms
remain, so no single justified intervention follows.

The replay uses the existing final checkpoints and validates every episode
against the stored metrics before analysis:

```bash
PYTHONPATH=runs/constrained-rl-deps:src:. python -m \
  benchmarks.constrained_rl_v21.diagnosis \
  --results runs/constrained-rl-v2-20261002 \
  --output runs/constrained-rl-v21-diagnosis-20261002/frozen-v2-trajectories.json

PYTHONPATH=src:. python -m benchmarks.constrained_rl_v21.analysis \
  runs/constrained-rl-v21-diagnosis-20261002/frozen-v2-trajectories.json \
  --v2-results runs/constrained-rl-v2-20261002 \
  --output benchmarks/constrained_rl_v21/diagnosis.json

PYTHONPATH=src:. python -m benchmarks.constrained_rl_v21.plot_results \
  runs/constrained-rl-v21-diagnosis-20261002/frozen-v2-trajectories.json \
  benchmarks/constrained_rl_v21/diagnosis.json \
  --v2-results runs/constrained-rl-v2-20261002 \
  --output-dir benchmarks/constrained_rl_v21/plots
```

See [PROTOCOL.md](PROTOCOL.md) for the frozen gate and [RESULTS.md](RESULTS.md)
for the interpretation. `diagnosis.json` contains the full event-level record.

