# Constrained RL V2 benchmark

This versioned study isolates one intervention relative to frozen Constrained V1:
the binary terminal completion/deadline cost is replaced with a dense physical
deadline-deficit cost. Energy remains the only objective, and the speed integral,
FSRL SACLag implementation, hyperparameters, seeds, simulator, and external evaluator
remain unchanged.

The preregistered protocol is in [PROTOCOL.md](PROTOCOL.md). The completed study and
Decision Gate B interpretation are in [RESULTS.md](RESULTS.md). `results.json` contains
the machine-readable analysis; `trajectories.json` contains the reproducible native-step
representative rollouts.

## Frozen CMDP

```text
objective_t = -signed_step_energy_kwh / 0.25

speed_cost_t =
    max(0, velocity_m_s - speed_limit_m_s) * dt_s

optimistic_remaining_time_s(position) =
    integral over remaining route of dx / speed_limit(x)

deadline_deficit_s = max(
    0,
    elapsed_time_s + optimistic_remaining_time_s - 140 s,
)

deadline_cost_t = dt_s * deadline_deficit_s / 140 s

episodic cost limits = [0 m, 0 s]
```

The deadline cost is nonnegative and dense once the state falls outside an optimistic
speed-limit-respecting completion envelope. There is no terminal task-failure impulse.
The proxy is a training signal, not the external feasibility definition.

## Reproduction

Install the project and the isolated research dependencies. The exact FSRL revision is
pinned in `requirements.txt`.

```bash
python -m pip install -e '.[dev]'
python -m pip install -r benchmarks/constrained_rl_v2/requirements.txt
```

Run the registered implementation smoke:

```bash
python -m benchmarks.constrained_rl_v2.experiment \
  --smoke \
  --output-dir runs/constrained-rl-v2-smoke
```

Run all three main seeds. Parallel workers change wall time, not the physical budget:

```bash
python -m benchmarks.constrained_rl_v2.experiment \
  --output-dir runs/constrained-rl-v2 \
  --workers 3 \
  --threads 2
```

Analyze only Development 2000--2008 and Validation 3000--3008:

```bash
python -m benchmarks.constrained_rl_v2.analysis \
  runs/constrained-rl-v2 \
  --v1-runs runs/constrained-rl-20260927 \
  --output runs/constrained-rl-v2/analysis.json
```

Record deterministic representatives and generate plots:

```bash
python -m benchmarks.constrained_rl_v2.trajectories \
  runs/constrained-rl-v2 \
  --output runs/constrained-rl-v2/trajectories.json

python -m benchmarks.constrained_rl_v2.plot_results \
  runs/constrained-rl-v2/analysis.json \
  --diagnostics-dir runs/constrained-rl-v2 \
  --trajectories runs/constrained-rl-v2/trajectories.json \
  --output-dir benchmarks/constrained_rl_v2/plots
```

The experiment validates the canonical configuration hash, rejects incomplete result
sets, checks all three training seeds, and fails if sealed paper seeds 4000--4017 occur
in the results.

## Scope

This directory does not implement requirement-conditioned RL, goal-conditioned RL,
HER, CRL, a new optimizer, or another cost sweep. The evidence should determine the
next study; no next method is launched automatically.

