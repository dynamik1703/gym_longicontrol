# Contextual Physical-Reference Protocol

Status: **preregistered before execution**.

> This is a post-hoc contextual physical-reference analysis on previously
> opened RL Validation tasks.

It is not a new independent held-out generalization benchmark. Its only purpose
is to determine whether failures already observed on the two frozen RL panels
occurred on tasks for which the already frozen physical references can produce
feasible witnesses. Contextual performance may not be used to tune either
physical controller or to select Hybrid RL design choices.

## Immutable inputs

| Input | Frozen revision / identity |
| --- | --- |
| Physical reference | `research/physical-feasibility-reference` at `3d9e8bcb72ecf4a01be192036df8f00c5ac9de8e` |
| Same-information panel | `research/same-information-scalar-vs-constrained` at `78a50b7ad866beb72a93f06cb1c5d6bcaa25cc12` |
| GCSL panel | `research/gcsl-canonical` at `5ada9402e3a2d627a7ea417cf1ffc23e1ab3789e` |

The physical controller source and configuration must match
`benchmarks/physical_feasibility_reference/freeze.json` exactly. In particular,
the causal cruise cap remains 60 km/h, the velocity margin remains `1e-6 m/s`,
the braking-position margin remains `0.5 m`, and the Oracle retains ten energy
refinement iterations. No controller source or setting may change after this
protocol is frozen.

The immutable RL `results.json` and `SPLIT_LEDGER.md` blobs are read directly
from the two named commits with `git show`; no RL checkpoint or actor is loaded.
The evaluator verifies their byte SHA-256, Git blob IDs, evaluation IDs, ordered
split hashes, complete 108-row episode matrices, and the 18 route/full-task
fingerprints before physical execution.

### Frozen panels

- `SAMEINFO_VALIDATION_V1`: seeds `9500001, 9500002, 9500003, 9500005,
  9500006, 9500007, 9500008, 9500011, 9500012, 9500014, 9500015,
  9500016, 9500017, 9500018, 9500019, 9500020, 9500021, 9500022`;
  ordered SHA-256 `d8a658e96b0090f8e0cdcdfdec3b291329045dc3dac9541e0603f34d916bf10a`.
- `GCSL_VALIDATION_V1`: seeds `9700000, 9700001, 9700002, 9700003,
  9700004, 9700005, 9700006, 9700007, 9700008, 9700009, 9700010,
  9700011, 9700012, 9700013, 9700014, 9700015, 9700017, 9700018`;
  ordered SHA-256 `d808fcc361be35f0c585003278f200d4013cc03718b0ceb318fbe6d55195a159`.

The route and full-task fingerprints are recovered from the immutable ledgers
and copied into `freeze.json`; runtime regeneration must equal every row.

## Execution

For each panel and task, execute exactly once as a scientific controller-task
pair:

1. the frozen full-route Oracle, including its frozen feasibility-first
   ten-iteration cruise-cap refinement;
2. the frozen observation-limited causal controller.

This gives exactly `2 panels × 18 tasks × 2 controllers = 72` scientific
controller-task records. Oracle search candidates are internal deterministic
solver trajectories, not additional reported task outcomes. Every selected
Oracle witness is independently replayed once; replay simulator steps and
search trajectory counts are reported separately.

Each completed controller-task record is written atomically beneath
`episodes/`. A normal invocation fails closed if any record already exists.
Recovery may evaluate only absent pairs; it must never repeat an existing pair.
`aggregate-only` performs no environment/controller execution and is the only
permitted reporting recovery after all immutable episode records exist.

Environment and success semantics are unchanged: `StochasticTrack-v1`, 1,000 m,
140 s, native 0.1 s dynamics, 1,800-step wrapper limit, exact zero speed
tolerance, production `EpisodeMetrics`, and production `is_feasible`.

## Frozen measurements

Every physical record contains feasibility, completion, deadline and speed
compliance, travel time, final position, maximum and integrated speed excess,
signed net energy, exclusive failure mode, solver time, simulator steps,
action/trace hashes, and Oracle replay status/steps. Energy summaries include
only feasible episodes.

For each panel the report contains Oracle and causal RSR `/18`, completion,
deadline and speed counts, feasible-only energy mean/median, paired task-level
reference outcomes, and paired feasible-only `causal - oracle` energy and travel
time differences. Percentage energy difference is defined per track as
`100 * (energy_causal - energy_oracle) / energy_oracle`; the aggregate percent
is the same expression applied to panel means.

Immutable RL rows are compared task-by-task. Per policy, per training seed, per
method, and pooled across the panel, report:

- failed policy-track-pair count and the subset with Oracle/causal witnesses;
- distinct failed-track count and the subset with Oracle/causal witnesses;
- speed-related, deadline-related and noncompletion counts plus the original
  exclusive failure modes;
- energy differences only for pairs where both RL and reference are feasible.

Policy-track pairs are the unit for policy results; repeated policies on one
track are never presented as independent track samples.

## Planned interpretation and limits

The analysis answers whether these observed RL-failed tasks have physical and
causal witnesses, whether limited preview is necessary to explain failures on
these panels, the observed reference energy gap, and whether an achieved energy
gap motivates a later learning study. It does not establish that all generated
tracks are solvable, that the Oracle is globally energy-optimal, or that future
hidden-route information is irrelevant in general.

No contextual result may change controller settings, Hybrid action-interface
selection, reward design, split construction, or training hyperparameters.
Paper seeds `4000–4017` remain numerically sealed and their content must never be
constructed. No RL run, training, adaptation, behavior cloning, demonstration,
or policy distillation is authorized.
