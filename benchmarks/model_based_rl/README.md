# Model-Based Constrained RL V1

This benchmark prepares a matched comparison between learned vehicle dynamics and the
known LongiControl physics model under the frozen Constrained RL V2 task and SACLag
learner. It is Dyna/MBPO-style training augmentation, not MPC: evaluation calls only
the learned actor.

Read `SOURCE_AUDIT.md`, `DESIGN.md`, `PROTOCOL.md` and `RESOURCE_REPORT.md` before any
execution. `canonical.json` is the frozen machine-readable protocol.

Useful inert checks:

```bash
python -m benchmarks.model_based_rl.runner preflight
python -m benchmarks.model_based_rl.runner status
python -m benchmarks.model_based_rl.runner model-disabled-parity
```

The checked-in execution state authorizes exactly the frozen six-policy matrix and its
single final Validation pass. Validation remains sealed until all six final policies
pass the completion gate. Paper-test tracks remain hard blocked.

Research-only dependencies are listed in `requirements.txt`. Raw replays, checkpoints,
models and run directories are intentionally ignored and are not benchmark artifacts.
