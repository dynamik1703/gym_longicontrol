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

## Safety state

`canonical.json` freezes the protocol with main training, Validation and paper-final
authorization all false. `runner.py` checks both `main_training_authorized` and
`main_training_enabled` before it can collect a main-study transition. The prepared
launchd file is never bootstrapped by preparation.

Preparation used 1,024 simulator transitions for two 512-transition collection
measurements, including the final explicitly seeded-0 benchmark, below the 5,000 cap.
All physics semantic tests use synthetic tracks and exact dynamics. Validation
3000–3008 and paper-final 4000–4017 remain sealed.

## Prepared commands

From the repository root with the pinned environment active:

```bash
python -m benchmarks.multistep_physics_mbrl.runner preflight
python -m benchmarks.multistep_physics_mbrl.runner status
python -m benchmarks.multistep_physics_mbrl.runner prepare-launchd
```

The last command only writes launch artifacts. `run`, `resume`, and `matrix` reject
execution in this frozen preparation snapshot. A later, explicit authorization must
change and review the canonical authorization fields before any main policy starts.

See `DESIGN.md` for target equations, `PROTOCOL.md` for the preregistration,
`SOURCE_AUDIT.md` for upstream semantics, and `RESOURCE_REPORT.md` for the compute
gate.
