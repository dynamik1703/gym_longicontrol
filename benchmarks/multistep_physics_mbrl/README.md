# Multi-Step Physics Model-Based Value Expansion

This benchmark prepares a matched six-policy study of exact-physics model-based value
expansion (MVE) for frozen LongiControl Constrained RL V2. It is intentionally a
training-time value-target method: the final controller is the deterministic learned
actor and never queries physics online.

The two conditions are:

- `matched_real_only`: untouched pinned FSRL SACLag with real `n_step=2` targets;
- `physics_mve`: the same learner and real samples, extended after the two real steps
  by 60 exact native physics steps and a target-critic bootstrap.

There is no synthetic replay buffer. Both conditions use the same real replay,
seeds (11, 29, 47), 300,000 real transitions and 30,000 optimizer updates per policy.
The actor update in both cases consumes only the sampled real states.

## Main-training result

The authorized six-policy main-training matrix is complete. On Development at the fixed
300k endpoint, matched real-only achieves pooled RSR 13/27 and Physics MVE achieves
0/27. Physics MVE seed 47 breaches the frozen numerical-stability gates, so the
preregistered classification is **E — unstable**. See
[`MAIN_TRAINING_REPORT.md`](MAIN_TRAINING_REPORT.md) and
[`main_training_results.json`](main_training_results.json) for the complete result and
execution provenance.

## Validation result

The single authorized Validation opening is complete. On tracks 3000--3008, matched
real-only reaches 10/27 pooled RSR and Physics MVE reaches 0/27. The frozen Validation
performance classification is **D -- degradation**; the overall study classification
remains **E -- unstable** because the Development instability override had already
triggered. See [`RESULTS.md`](RESULTS.md) and [`results.json`](results.json) for all 54
compact episode records, Development-to-Validation comparisons, stability diagnostics,
and provenance.

## Safety state

`canonical.json` retains the reviewed main-training and one-time Validation
authorizations as execution provenance. Paper-final authorization remains false. `runner.py` checks
both `main_training_authorized` and `main_training_enabled` before it can collect a
main-study transition. The completed launchd job exited successfully and was unloaded.

Preparation used 1,024 simulator transitions for two 512-transition collection
measurements, including the final explicitly seeded-0 benchmark, below the 5,000 cap.
All physics semantic tests use synthetic tracks and exact dynamics. Validation
3000--3008 is complete; paper-final 4000--4017 remains sealed and unopened.

## Execution commands

From the repository root with the pinned environment active:

```bash
python -m benchmarks.multistep_physics_mbrl.runner preflight
python -m benchmarks.multistep_physics_mbrl.runner status
python -m benchmarks.multistep_physics_mbrl.runner prepare-launchd
```

The last command only writes launch artifacts. Main training and the single Validation
opening are complete. Paper final still requires separate explicit authorization and
remains sealed.

See `DESIGN.md` for target equations, `PROTOCOL.md` for the preregistration,
`SOURCE_AUDIT.md` for upstream semantics, and `RESOURCE_REPORT.md` for the compute
gate.
