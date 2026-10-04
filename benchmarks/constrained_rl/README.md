# Constrained RL benchmark

This study isolates task formulation: FSRL SAC-Lagrangian minimizes physical net
energy while two explicit costs represent integrated speed excess and failure to
finish within 140 seconds. The unchanged reward-independent evaluator determines
episode feasibility.

The frozen design and decision rules are in [`PROTOCOL.md`](PROTOCOL.md), and all
machine settings are in [`canonical.json`](canonical.json). Install the optional
research stack in an isolated environment:

```bash
python -m pip install -r benchmarks/constrained_rl/requirements.txt
```

Training artifacts and checkpoints belong under ignored `runs/`; only compact
results, intentional plots, and the final report are versioned. Exact commands are
shown below.

```bash
python -m benchmarks.constrained_rl.experiment \
  --output-dir runs/constrained-rl-20260927 \
  --device cpu --threads 2 --workers 3

python -m benchmarks.constrained_rl.analysis \
  runs/constrained-rl-20260927 \
  --output runs/constrained-rl-20260927/analysis.json

python -m benchmarks.constrained_rl.trajectories \
  runs/constrained-rl-20260927 \
  --output runs/constrained-rl-20260927/trajectories.json

python -m benchmarks.constrained_rl.plot_results \
  benchmarks/constrained_rl/results.json \
  --diagnostics-dir runs/constrained-rl-20260927 \
  --trajectories runs/constrained-rl-20260927/trajectories.json \
  --output-dir benchmarks/constrained_rl/plots
```

The completed study reaches preregistered **case D**: 0/27 validation episodes
are feasible or complete, while 27/27 respect the speed limit. All three policies
learn deterministic standstill. See [`RESULTS.md`](RESULTS.md) for the physical,
optimization, trajectory, and frozen-baseline analysis. This result does not
justify a 1M run or a transition to requirement-conditioned RL; the sparse
completion/deadline cost representation must be repaired first.
