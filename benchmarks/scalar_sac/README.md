# Scalar SAC requirement-sensitivity benchmark

## Research question

This benchmark asks how sensitive successful longitudinal control is to a
hand-designed scalar reward. The operational task is fixed independently:

> Minimize signed net energy while completing the route within 140 seconds and
> never exceeding the applicable speed limit.

Mean episode reward is a training diagnostic, **not** a benchmark result. The
primary outcome is Requirement Satisfaction Rate (feasible episodes divided by
all evaluation episodes). Energy is summarized only for feasible episodes.
Future work can compare MORL, constrained RL, and requirement-conditioned RL
against this protocol; none of those approaches is implemented here.

The canonical configuration is [`canonical.json`](canonical.json). It uses
`StochasticTrack-v1`, training seeds 11/29/47, calibration tracks 2000–2008,
and a separate fixed evaluation set of track seeds 1000–1008. The adjacent
budgets 120/140/160 seconds are listed for later sensitivity analysis; 140
seconds is the canonical result.

## Why 140 seconds?

`python -m benchmarks.scalar_sac.calibrate` reproduces a pilot with two
physics-aware reference controllers on seeds 2000–2008. They are privileged
calibration tools that read the generated track, not learned-policy baselines.
The reporting tracks 1000–1008 are not used to choose the task. Both controllers
stay below the speed limits under the benchmark's end-of-step definition.

| controller | completed / clean | travel time min / median / max |
| --- | ---: | ---: |
| fast-safe (0.5 m/s margin) | 9/9 / 9/9 | 63.2 / 85.6 / 129.7 s |
| conservative (3.0 m/s margin) | 7/9 / 9/9 | 73.4 / 107.2 / 180.0 s |

Thus 140 seconds leaves headroom above the fast clean reference on every
calibration track, while conservative energy-oriented behavior can still miss
the requirement or fail to finish. This is empirical evidence for a useful
first task, not a claim of global physical optimality.

## Scalar reward

Only the training reward is replaced by a Gymnasium wrapper. Dynamics,
observations, actions, route termination, `EpisodeMetrics`, and the historical
`v1` reward implementation are unchanged. For transition `t`,

```text
r_t = w_progress * delta_position_m / 1000 m
    - w_energy   * step_energy_kwh / 0.25 kWh
    - w_time     * delta_time_s / 140 s
    - w_speed    * speed_excess_m_s * delta_time_s / 1 m
```

The energy term uses signed net energy, so regeneration has the same semantics
as the physical benchmark objective. The speed term is normalized integrated
speed excess. Every returned component is available under
`info["benchmark_reward_components"]`; the wrapped historical reward remains
under `info["historical_reward"]` for debugging.

The compact 2×2×2 grid fixes `w_progress=1` and varies:

- `w_energy` in `{0.5, 2.0}`;
- `w_speed` in `{0.0, 2.0}`;
- `w_time` in `{0.25, 1.0}`.

This deliberately tests low/high energy emphasis, absent/present speed
penalties, and relaxed/strong time pressure without turning the experiment into
a general hyperparameter search. Each run records all weights and
normalizations alongside the training seed.

## Running the experiment

Install the training and plotting dependencies, then run:

```bash
python -m pip install -e ".[train,render]"
python -m benchmarks.scalar_sac.experiment --output-dir runs/scalar-sac
```

The default launches eight reward configurations for three training seeds.
Restrict it while developing:

```bash
python -m benchmarks.scalar_sac.experiment \
  --reward-id e05-s2-t1 --training-seed 11 \
  --output-dir runs/scalar-sac

python -m benchmarks.scalar_sac.experiment \
  --smoke --device cpu --output-dir runs/scalar-sac-smoke
```

Independent policies can be trained concurrently with `--workers N`. This does
not change seeds or stored metadata. For example, the first complete experiment
used `--device cpu --workers 6`; each worker was limited to one BLAS thread via
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.

Each run directory contains `result.json` with raw episode records and aggregate
statistics, `run-config.json`, and an SAC checkpoint. `runs/` is git-ignored;
do not commit checkpoints or large result sets.

Generate three deterministic, headless diagnostic plots from those JSON files:

```bash
python -m benchmarks.scalar_sac.plot_results runs/scalar-sac \
  --output-dir runs/scalar-sac/plots
```

The plots show reward sensitivity, feasible energy versus satisfaction, and
whether failures came from incompletion, time, or speed. Evaluation never reads
the reward, so the same evaluator can later score policies produced by other
algorithms.

Evaluate and persist the three reference policies, then produce the track-aware
analysis:

```bash
python -m benchmarks.scalar_sac.references \
  --output benchmarks/scalar_sac/reference-results.json
python -m benchmarks.scalar_sac.analysis runs/scalar-sac \
  --references benchmarks/scalar_sac/reference-results.json \
  --output runs/scalar-sac/analysis.json
```

The fast and conservative references have privileged access to the future
track and are oracle-style physical anchors, not fair competitors. The random
policy does not use privileged information.

Representative trajectories are first replayed into a persisted JSON file;
the plot is then generated from that stored data rather than transient state:

```bash
python -m benchmarks.scalar_sac.trajectories runs/scalar-sac \
  --trajectory-json runs/scalar-sac/representative-trajectories.json \
  --plot runs/scalar-sac/plots/representative-trajectories.png
```

The selection rule is stored alongside the trajectories. See
[`RESULTS.md`](RESULTS.md) and [`results-140s.json`](results-140s.json) for the
first complete 140-second experiment. Large raw outputs and checkpoints remain
under ignored `runs/` directories.

## Scalar reward V2 baseline hardening

The V1 study above is preserved as the naive reward-specification artifact.
V2 separately asks whether explicit return-scale analysis, terminal task
signals, and longer training produce a fair scalar baseline. The formula,
candidate meanings, track split, and acceptance threshold were frozen before
evaluation in [`V2_PROTOCOL.md`](V2_PROTOCOL.md). A validation-driven scale
repair is separately recorded in [`V2B_PROTOCOL.md`](V2B_PROTOCOL.md); together
the two iterations contain six candidates, not a Cartesian search.

Run an iteration with continuous 100k/300k checkpoints and 50k validation
curves:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m benchmarks.scalar_sac.v2_experiment \
  --config benchmarks/scalar_sac/canonical_v2b.json \
  --device cpu --workers 6 \
  --output-dir runs/scalar-sac-v2b

python -m benchmarks.scalar_sac.v2_analysis runs/scalar-sac-v2b \
  --config benchmarks/scalar_sac/canonical_v2b.json \
  --references benchmarks/scalar_sac/reference-results.json \
  --v1-summary benchmarks/scalar_sac/results-140s.json \
  --output runs/scalar-sac-v2b/analysis.json
```

Plot learning curves and replay the deterministic representative checkpoints:

```bash
python -m benchmarks.scalar_sac.v2_plot_results \
  runs/scalar-sac-v2b/analysis.json \
  --output-dir runs/scalar-sac-v2b/plots

python -m benchmarks.scalar_sac.v2_trajectories runs/scalar-sac-v2b \
  --config benchmarks/scalar_sac/canonical_v2b.json \
  --trajectory-json runs/scalar-sac-v2b/representative-trajectories.json \
  --output-dir runs/scalar-sac-v2b/plots
```

The complete conclusion is in [`RESULTS_V2.md`](RESULTS_V2.md), with a compact
machine-readable summary in [`results-v2.json`](results-v2.json) and factual
tuning provenance in [`reward-development.json`](reward-development.json).
No candidate met the predefined all-seed reliability threshold. Seeds
4000–4017 remain reserved and uninspected for a future frozen paper protocol.
