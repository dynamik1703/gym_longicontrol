# Frozen protocol

## Matched matrix

| Property | Matched real-only | Physics MVE |
|---|---:|---:|
| Training seeds | 11, 29, 47 | 11, 29, 47 |
| Real transitions per policy | 300,000 | 300,000 |
| Optimizer updates per policy | 30,000 | 30,000 |
| Real replay batch | 256 | 256 |
| Real prefix | 2 steps | 2 steps |
| Exact model expansion | 0 | 60 steps |
| Synthetic replay rows | 0 | 0 |
| Actor-update states | sampled real states | sampled real states |

Both conditions share code, learner, track-seed stream, RNG construction, replay,
collection schedule, checkpoint schedule, and Development evaluation. They differ only
in critic target expansion. Total authorized-by-protocol (but not yet authorized for
execution) interaction is 1,800,000 real native transitions. `update_per_step=.1` and
there are no extra MVE optimizer updates.

Frozen learner parameters remain: FSRL commit `e056fc9498d5d037869533da7cf976acf462f918`,
Tianshou 0.5.1, gamma .99, 128x128 networks, actor LR .0005, critic LR .001,
automatic alpha with LR .0003 and initial effective alpha 1, tau .05, replay capacity
100,000, batch 256, PID `(0.05, 0.0005, 0.1)`, Lagrangian rescaling, zero cost limits,
native 10 Hz action rate, and deterministic evaluation. The sole condition parameter
difference is `model_expansion_steps: 0` versus `60`.

## Splits and evaluation

Training rejects all seeds 1000–1008, 2000–2008, 3000–3008 and 4000–4017.
Development uses 2000–2008 at 50k, 100k, 150k, 200k, 250k and the fixed final 300k
checkpoint. No best-checkpoint selection is allowed. Validation 3000–3008 is opened
once only after all six final policies are frozen and separately authorized.
Paper-final 4000–4017 stays sealed.

Development and later Validation use only:

```text
ordinary observation -> deterministic learned actor -> action
```

No physics query, lookahead, planning or model-derived action is permitted.

## Outcomes and interpretation

The primary outcome is final pooled Validation RSR, reported by policy seed and
condition. Secondary outcomes are completion, deadline compliance, speed compliance,
exclusive failure modes, Development curves, feasible-only energy, seed robustness,
and training stability. Energy is interpreted only among feasible episodes.

The frozen interpretation is:

- **A — robust improvement:** MVE RSR at least 24/27, completion 27/27, deadline
  27/27, speed at least 24/27, and RSR exceeds matched real-only.
- **B — material, not robust:** MVE exceeds matched real-only pooled RSR by at least
  3/27 but misses A.
- **C — no material improvement:** absolute pooled RSR difference is below 3/27.
- **D — degradation:** MVE is at least 3/27 below baseline or materially worsens
  completion/deadline.
- **E — unstable:** any non-finite monitored value, multiplier at least 1,000, critic
  loss at least 1,000,000, or target magnitude at least 1,000,000.

Late collapse is separately flagged when peak Development RSR exceeds final 300k RSR
by at least 3 tracks.

## Diagnostics

At every 50k scientific checkpoint, the runner records count-weighted mean, standard
deviation, minimum and maximum for real-prefix, model-expansion, terminal/bootstrap,
total target, and absolute model-expansion fraction, independently for all three
critics. It also records model steps, terminal/truncation counts, RSR, completion,
speed, deadline, both multipliers, alpha, actor loss, and all critic losses.

The non-controlling anticipatory diagnostic compares discounted predicted speed cost
from stored pre-drop real states after fixed first actions `accelerate`, `coast`, and
`brake`; later actions come from the current stochastic actor. Its results never select
an action or checkpoint.

## PID, recovery and authorization

Only a genuine completed real physical episode passes two finite episodic costs to
FSRL `pre_update_fn`; modeled completion is rejected by a hard source gate. Model costs
affect critics only.

Recovery checkpoints are atomic and verified every 10k real transitions. They contain
policy and target parameters, actor/critic/alpha optimizer state, PID state, real
replay, exact-physics sidecar, environment and collector state, training-track RNG,
Python/NumPy/Torch RNG, dedicated rollout RNG, counters, diagnostics, completed episode
state, source hashes, runtime versions, and immutable attempt identity. Resume requires
an interrupted attempt and its explicit verified checkpoint. Duplicate starts and
silent fresh restart are rejected.

Although the full runner and a launchd plist writer are prepared, all execution flags
remain false. Preparation does not bootstrap launchd or start any policy.
