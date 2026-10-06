# Goal-conditioned study execution

`runner.py` executes the already frozen comparison in `PROTOCOL.md`. It is not
a configurable experiment framework: condition order, seeds, budget,
checkpoints and evaluation splits come only from `canonical.json`.

The main-study command is:

```bash
PYTHONPATH=src python -m benchmarks.goal_conditioned.runner \
  --output-root runs/goal-conditioned-her-v1 \
  --device cpu
```

The runner requires a clean `research/goal-conditioned-her` branch containing
the preparation commit and a later committed runner. Before creating the run
directory it verifies the scientific configuration and Goal/Environment/Replay
source hashes, installed dependency versions, the closed LLM study and its
frozen artifacts.

The output root is created exclusively. `ACTIVE.lock` and an atomic
`manifest.json` are written before training; any pre-existing root is treated
as a completed or partial study and is never overwritten. A technical failure
leaves the lock, manifest and all completed artifacts in place.

Each policy is trained with one `learn()` call. Intermediate checkpoint `N` is
captured at the next callback boundary, after update `N` and before transition
`N+1` is stored. The final 300k checkpoint is captured in `on_training_end`,
after its scheduled update. Development evaluation uses a separate environment
and restores Python, NumPy and Torch RNG state before training resumes.

Validation remains sealed until the manifest proves that all six policies have
exactly 300,000 transitions, six Development results and unchanged final-model
hashes. Only the final model for each condition/seed is then evaluated on
tracks 3000--3008. Tracks 4000--4017 are not accepted by any execution path.

Replay diagnostics subclass the frozen `GoalReplayBuffer`. Counters inspect
only real and virtual rows already sampled by SAC. They make no additional
replay draw and are tested to preserve both returned samples and NumPy RNG
state. Model ZIPs and raw run records remain under ignored `runs/`; compact
results are copied into the benchmark only after the completed run is audited.
