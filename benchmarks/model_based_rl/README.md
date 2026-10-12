# Model-Based Constrained RL V1

This benchmark prepares a matched comparison between learned vehicle dynamics and the
known LongiControl physics model under the frozen Constrained RL V2 task and SACLag
learner. It is Dyna/MBPO-style training augmentation, not MPC: evaluation calls only
the learned actor.

The mixed six-policy study was terminated incomplete on 2026-10-12 because measured
computational cost made the preregistered experiment operationally infeasible. This
decision was not triggered by Development performance. Validation and paper-test
tracks remained sealed. See `TERMINATION.md` and `termination.json`.

Read `SOURCE_AUDIT.md`, `DESIGN.md`, `PROTOCOL.md` and `RESOURCE_REPORT.md` before any
execution. `canonical.json` is the frozen machine-readable protocol.

Useful inert checks:

```bash
python -m benchmarks.model_based_rl.runner preflight
python -m benchmarks.model_based_rl.runner status
python -m benchmarks.model_based_rl.runner model-disabled-parity
```

The frozen protocol specified a six-policy matrix and one final Validation pass. The
terminated execution never reached that gate. Validation remains sealed and paper-test
tracks remain hard blocked.

Research-only dependencies are listed in `requirements.txt`. Raw replays, checkpoints,
models and run directories are intentionally ignored and are not benchmark artifacts.
