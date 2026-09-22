# Start with a trained LongiControl agent

For vector rewards, CAPQL and a multi-objective comparison, see the
[MORL guide](morl/README.md). The scalar examples below are unchanged.

These examples run from a **repository checkout or extracted source distribution**.
The wheel contains the environment, not the examples or demonstration policy.
Use Python 3.10 or newer and run all commands below from the checkout root.

## Install and try the bundled agent

Using a fresh virtual environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[examples]"
python -m examples.sb3_quickstart demo
```

No training or additional model download is necessary. The demo runs a small,
already trained Stable-Baselines3 SAC policy on five seeded stochastic tracks
and prints a JSON comparison with a random policy on the same tracks.
It uses CPU inference and does not open a GUI by default.

For plots and a guided walkthrough, launch the notebook in the same environment:

```bash
python -m jupyterlab examples/quickstart.ipynb
```

Choose **Run All**. The notebook does not install packages, start training, or
write files. It loads the model, checks the SB3 interface, plots one drive, and
evaluates both policies. If imports fail, check that the selected kernel uses
the Python environment where you installed the examples extra.

Optional CLI commands:

```bash
python -m examples.sb3_quickstart demo --episodes 1 --seed 1001
python -m examples.sb3_quickstart demo --report comparison.json
python -m examples.sb3_quickstart demo --episodes 1 --render
```

`--render` needs a desktop display. On headless machines, omit it and use the
notebook's inline plots. Report files are created exclusively; choose a new
filename instead of overwriting an earlier result.

## Train your own model

```bash
python -m examples.sb3_quickstart train --steps 2000 --output runs/my-first-sb3
python -m examples.sb3_quickstart demo --model runs/my-first-sb3/model.zip
```

Two thousand steps are a workflow smoke test, not a convergence target.
The default is 20,000 steps with seed 42, a `[64, 64]` network, and one CPU
thread. Runtime depends on the machine. Training uses the unmodified v1
environment with its default reward weights, normalized observations, and
1,800-step time limit. `--env DeterministicTrack-v1` selects the fixed track;
the default is `StochasticTrack-v1`. Evaluation reads the environment ID from
the checkpoint's metadata automatically.

SB3's checker recommends `float32` actions and may emit a warning: v1 deliberately
retains its `float64` action/observation contract. The integration tests exercise
training and inference with that existing contract; the warning is not an error.

Each **new** output directory receives:

- `model.zip`: the SB3 model and optimizer parameters;
- `metadata.json`: environment ID, training settings, software versions,
  source-script hash, model hash, training duration, and seed;
- `evaluation.json`: per-episode results and aggregate SAC/random comparisons.

Keep `model.zip` and its adjacent `metadata.json` together. The loader checks
the hash before deserialization. **Only load trusted SB3 checkpoints.** SB3's
[save format](https://stable-baselines3.readthedocs.io/en/master/guide/save_format.html)
can contain pickle-based metadata; a checksum is not a security sandbox.

This short example intentionally does not implement resuming training or saving
the replay buffer/RNG state. Use the separate `python -m training` workflow
for the existing full-state checkpoint functionality. The two trainers' checkpoint
formats are not interchangeable.

## Interpret the results

Read the [bundled model card](models/sac_demo/README.md) before interpreting its
performance. The report shows reward, completion rate, distance, elapsed time,
estimated net energy, mean absolute jerk, and time above the speed limit.

All averages include timed-out episodes. Compare completion rate, distance and
travel time before making energy-efficiency claims. A stationary or slow agent
may use less energy simply because it never finishes. Energy comes from the
historical BMW i3 estimator, not physical measurements. Overspeed is measured at
simulation step endpoints, not continuously.

One training seed and five evaluation tracks are an illustrative demonstration,
not a statistical benchmark. Different package versions or hardware can change
training outcomes. No safety, optimality, real-vehicle suitability, or general
energy-saving claim is made.

The integration uses SB3's documented
[custom-environment interface and checker](https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html).
The core package still needs only Gymnasium and NumPy; examples dependencies
remain optional.
