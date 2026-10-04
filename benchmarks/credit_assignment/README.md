# Scalar credit-assignment diagnostic

This directory contains the preregistered 2 x 2 SB3 SAC experiment described in
[`PROTOCOL.md`](PROTOCOL.md). It changes only gamma and the policy decision
frequency while preserving the V2-B reward, 0.1 s physics, task, metrics, and
data splits.

Run a smoke test:

```bash
python -m benchmarks.credit_assignment.experiment \
  --smoke --device cpu --output-dir runs/credit-assignment-smoke
```

Run the complete matrix (three workers keep one training seed per process):

```bash
python -m benchmarks.credit_assignment.experiment \
  --workers 3 --device cpu \
  --output-dir runs/credit-assignment-20260926
```

Generate the frozen analysis, plots, and native-resolution trajectories:

```bash
python -m benchmarks.credit_assignment.analysis \
  runs/credit-assignment-20260926 \
  --output runs/credit-assignment-20260926/analysis.json
python -m benchmarks.credit_assignment.plot_results \
  runs/credit-assignment-20260926/analysis.json \
  --output-dir runs/credit-assignment-20260926/plots
python -m benchmarks.credit_assignment.trajectories \
  runs/credit-assignment-20260926 \
  --trajectory-json \
    runs/credit-assignment-20260926/representative-trajectories.json \
  --output-dir runs/credit-assignment-20260926/plots
```

Raw checkpoints and logs remain under ignored `runs/`. Only the protocol,
configuration, reproducibility code, compact analysis summary, small plots, and
final written result belong in version control. Tracks 4000--4017 are not used.
