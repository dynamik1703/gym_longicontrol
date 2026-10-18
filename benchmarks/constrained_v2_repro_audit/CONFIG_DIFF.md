# Scientific configuration diff

The frozen Constrained RL V2 configuration SHA-256 is
`89d7142cc84ba60d98075313220b331b60c2c41997b6c65b11adeb1b08712060`.
The current MVE configuration explicitly references that exact V2 hash. The
historical run configs also record it. No scientific parameter difference was found.

## Parameter comparison

| Area | Historical Constrained V2 | Current Matched Real-Only | Result |
|---|---|---|---|
| Environment | `StochasticTrack-v1` | same | Identical |
| Objective | negative net energy, scale 0.25 | same | Identical |
| Speed cost | integrated speed violation in metres | same | Identical |
| Deadline cost | normalized optimistic deadline-deficit integral | same wrapper/config | Identical |
| Cost limits/order | `[speed_integral_m, deadline_deficit_integral_s]`, `[0, 0]` | same | Identical |
| Task | 1000 m, maximum time 140 s, zero permitted speed violation | same | Identical |
| Native action interval | 0.1 s, action repeat 1 | same | Identical |
| Episode limit | 1800 native steps | same | Identical |
| Gamma / n-step | 0.99 / 2 | same | Identical |
| Batch / replay | 256 / 100,000 | same Tianshou replay | Identical |
| Update ratio | 0.1 update per real transition | same nominal value | Parameter identical; realization differs |
| Warm-up | first update cannot occur before buffer size 256 | same effective buffer guard | Equivalent threshold; schedule differs |
| Actor | probabilistic 8D → 128 → 128 → 1 | same builder | Identical |
| Critics | three double critics, 9D → 128 → 128 → 1 | same builder | Identical |
| Actor/critic LR | `5e-4` / `1e-3` | same | Identical |
| Entropy alpha | automatic, initial 1.0, LR `3e-4`, target entropy -1 | same policy | Identical parameters |
| Target update | tau 0.05 | same | Identical |
| Action mapping | scaling enabled; `clip` bound method | same | Identical |
| PID | `(0.05, 0.0005, 0.1)`, rescaling enabled | same FSRL objects | Identical parameters; timing differs |
| Initial multipliers | speed 0, deadline 0 | same | Identical |
| Training seeds | 11, 29, 47 | same labels | Identical labels, different trajectories |
| Development | 2000–2008 | same | Identical |
| Validation | 3000–3008 | same | Identical |
| Paper | 4000–4017, sealed | same | Identical |
| Evaluation | deterministic actor, no observation/reward normalization | same shared evaluator | Identical |

The complete canonical configuration appears in `results.json` under
`configuration.canonical_configuration`. `scientific_parameter_differences` is an
empty machine-readable list.

## Effective differences outside the configuration object

Nominal values do not make the runners end-to-end equivalent:

- Historical `episodes_per_collect=1` is enacted by
  `FastCollector.collect(n_episode=1)`; the current runner directly calls
  `Collector.collect(n_step=1)`.
- Historical updates are `round(0.1 * episode_steps)` after each episode. Current
  updates target `floor(total_real_transitions * 0.1)` after every transition.
- Historical final update totals are 29,995 / 30,003 / 30,003; current totals are
  exactly 30,000 / 30,000 / 30,000.
- Historical training uses the base environment's continuous RNG stream. Current
  training uses a dedicated track-seed generator and re-seeds the base environment
  on every reset.
- Historical training evaluated Development and Validation at every checkpoint.
  Current training evaluated Development at checkpoints and opened Validation once
  after training. This protocol difference does not change deterministic final-policy
  evaluation outcomes but matters for provenance and sealing.

These are implementation/RNG mismatches, not evidence of a changed scientific task
or hyperparameter.

## Environment/task source diff

No source diff was found in dynamics, observation generation, speed-limit lookup,
task/deadline logic, energy model, termination, track generation, or metrics between
the earliest archived V2 snapshot and the audit-start commit. Classification by code
path is therefore:

| Code path | Diff | Classification |
|---|---|---|
| Dynamics and energy | none | no scientific change |
| Observation and speed-limit sensing | none | no scientific change |
| Track/task generation | none | no scientific change |
| Termination and dense deadline cost | none | no scientific change |
| External metrics/feasibility | none | no scientific change |
| Current MVE runner/recovery/sidecar | added after V2 | execution-only for M=0 except runner cadence/RNG |
| Documentation/tests | later additions | documentation/test-only |

## Evaluation parity

Both final result artifacts use Validation tracks 3000–3008, the shared deterministic
actor adapter/evaluator, the same 8D observations, no normalization, identical action
scaling/clipping, a maximum of 1800 steps, exact-zero speed compliance, the 140 s
deadline, and the same `EpisodeMetrics`/feasibility authority. No evaluation mismatch
was found that explains 21/27 versus 10/27. Validation was not rerun during this audit.
