# Scientific design

## Factorial treatments

| ID | Condition | Track stream | Collector |
|---|---|---|---|
| A | `A_histTrack_histCollector_seed11` | `T_HIST` | `C_HIST` |
| B | `B_histTrack_currentCollector_seed11` | `T_HIST` | `C_CURRENT` |
| C | `C_currentTrack_histCollector_seed11` | `T_CURRENT` | `C_HIST` |
| D | `D_currentTrack_currentCollector_seed11` | `T_CURRENT` | `C_CURRENT` |

`T_HIST` is the current-runtime reconstruction of continuous environment RNG seed
11 with used reset draws 0, 2, 4, … . `T_CURRENT` uses explicit reset seeds from
`SeedSequence([11, 0x4D564531])`.

`C_HIST` holds the actor fixed for a complete episode, inserts the terminal
transition, performs one real-episode PID update, then catches accumulated learner
updates up to `floor(real_transitions × 0.1)`. `C_CURRENT` applies the same terminal
PID rule but catches learner updates after every transition. Both receive exactly
300,000 real transitions and 30,000 RL updates.

## Evaluation and estimands

Every policy is evaluated deterministically and actor-only on the same 18 frozen
route-clean tasks at 50k, 100k, 150k, 200k, 250k, and 300k. The primary endpoint is
300k Diagnostic requirement-satisfaction count out of 18. Completion, deadline,
speed, failure modes, travel time, speed violation, and feasible-only energy are
secondary descriptive endpoints.

At 300k, the frozen contrasts are `C-A`, `D-B`, `B-A`, `D-C`, and interaction
`D-C-B+A`, reported as counts and rates for RSR, completion, deadline, and speed.
This is one seed, so no significance claim is permitted.

D is tested against exact archived Current Real-Only seed-11 dynamics, including
learned-state hashes and six checkpoint lambda/alpha trajectories. Its new
diagnostic outcome is not compared to archived Validation performance. A is only a
current-runtime historical-direction reconstruction, never an exact historical
reproduction.
