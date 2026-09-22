# SAC quickstart model card

## Intended use

This small model demonstrates loading, evaluating, plotting, and training a
Stable-Baselines3 agent in LongiControl. It is **not** an optimized controller,
a validated real-vehicle policy, or evidence that RL saves energy.

The model is bundled in the repository and source distribution so the notebook
can run without training or a separate model download. It is not included in the
core wheel.

## Training recipe

```bash
python -m pip install -e ".[examples]"
python -m examples.sb3_quickstart train --steps 20000 --seed 42 --output runs/reproduce-demo
```

- Environment: `StochasticTrack-v1`, default BMW i3 estimator, 1,000 m track,
  0.1 s timestep, 1,800-step time limit, default reward weights `[1, 0.5, 1, 1]`.
- Algorithm: Stable-Baselines3 SAC, `[64, 64]` hidden layers, CPU, one Torch thread.
- Budget: 20,000 interaction steps; 500 initial exploration steps; one training seed.
- Learning rate `0.0003`, batch size 64, replay capacity 50,000, gamma `0.99`,
  tau `0.005`, one gradient update per step after warmup, entropy `auto_0.1`.
- No reward shaping, observation normalization wrapper, or hyperparameter search.

Exact installed versions, measured training duration, script fingerprint and model
SHA-256 are recorded in [metadata.json](metadata.json). The source base revision
identifies the environment/refactoring commit; the script hash identifies the
example used to train it. Retraining is not guaranteed to produce bit-identical
weights across versions, operating systems, or devices.

## Recorded simulation results

Evaluation uses deterministic SAC predictions on five freshly reset tracks with
seeds **1001–1005**. A uniform-random action policy uses the same track seeds and
seeds its action space independently with the corresponding seed. The evaluation
seeds differ from the training seed, but this is not a formal held-out benchmark.
No test tracks were selected or excluded based on the results.

| Metric | SAC demo | Random policy |
|---|---:|---:|
| Completed episodes | 4 / 5 | 0 / 5 |
| Mean episode return | -1404.68 | -2144.41 |
| Mean distance [m] | 992.48 | 243.77 |
| Mean elapsed time [s] | 175.92 | 180.00 |
| Mean estimated net energy [kWh] | 0.5265 | 0.1498 |
| Mean absolute jerk [m/s³] | 1.70 | 16.26 |
| Mean time above speed limit [s] | 9.16 | 0.00 |

All means include incomplete episodes. The random agent's lower energy use is
**not an efficiency advantage**: it travels much less and never reaches the finish.
The SAC demo times out on seed 1005 and exceeds speed limits on other tracks.
The plotted trajectory also shows repeated acceleration/braking oscillations;
reaching the finish does not imply a comfortable or efficient driving policy.
Neither policy should be interpreted as safe or suitable for deployment.

The full per-episode measurements are in [evaluation.json](evaluation.json).
Re-evaluate the bundled weights with:

```bash
python -m examples.sb3_quickstart demo --report demo-recheck.json
```

## Limitations and loading safety

- One training seed and five evaluation seeds do not establish statistical
  superiority, robustness, convergence, or generalization.
- Energy is predicted by the historical vehicle model; it is not a measurement.
- There are no other vehicles, collisions, traffic lights, or road gradients.
- Speed-limit violations are counted at simulation step endpoints.
- The SB3 checkpoint is intended for inference here; replay/RNG state for exact
  continuation is not saved. It cannot be loaded by the separate legacy SAC trainer.
- Only load trusted archives: SB3's format may deserialize pickle-based metadata.
  The SHA-256 check detects corruption, not malicious code in an untrusted artifact.
