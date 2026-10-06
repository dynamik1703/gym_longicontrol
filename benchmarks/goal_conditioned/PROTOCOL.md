# Goal-conditioned SAC versus SAC+HER protocol

Status: **preregistered on 2026-10-06 before policy training**.

## Question and primary contrast

Does constraint-preserving hindsight replay improve acquisition of the
canonical LongiControl task relative to the same goal-conditioned SAC learner
without relabeling?

The two and only two primary conditions are:

- `sac-no-her`: SB3 SAC with `GoalReplayBuffer` and zero virtual goals;
- `sac-her`: the same SAC/replay implementation with four virtual goals.

Both receive identical physical rollouts, Dict observations, canonical real
goals, sparse first-arrival reward, terminal semantics, architecture,
preprocessing, warmup, budget, seeds and evaluations. The only intended
difference is future position relabeling in replay and the mechanically required
reward/terminal recomputation for those virtual transitions.

## Task and information

Canonical evaluation is unchanged: complete the original 1,000 m route by
140 seconds with maximum speed excess at most 0 m/s. External
`EpisodeMetrics`/`is_feasible` results, not replay reward, determine evaluation
success. Energy is evaluation-only.

The policy receives the original eight features plus normalized achieved and
desired goal vectors defined in `DESIGN.md`. Additional physical information is
current/previous position, elapsed time, cumulative maximum speed excess,
target position, deadline and speed tolerance. It receives no full track,
track ID, future action, optimal trajectory or oracle signal.

Every real rollout goal is `[1, 1, 140/180, 0]`. HER may replace only the two
target-position entries with an admissible strictly future achieved position.
Deadline and tolerance remain fixed. Reward is exactly 1 on valid first arrival
and 0 otherwise in both arms. The finite 180-second horizon and route end are
terminal failures when the requirements are not met.

## Learner and replay

The recorded implementation is Stable-Baselines3 2.9.0 SAC with
`MultiInputPolicy`, 256x256 networks, learning rate `3e-4`, replay capacity
1,000,000, batch 256, `tau=0.005`, `gamma=0.99`, train frequency 1, one gradient
step per interaction and automatic entropy tuning. No observation or reward
normalization is used.

Both conditions use `learning_starts=1800`, replacing the historical 100-step
warmup because HER requires a recorded completed episode. The warmup is counted
inside the fixed budget. Both arms use completed-episode sample eligibility and
set `handle_timeout_termination=False`. The control returns only real samples.
HER uses four virtual goals per real-goal share (SB3 ratio 4/5), inclusive
`future` selection, no copied `info`, and the benchmark-local position-only/
terminal-consistent adapter.

## Sampling, splits, seeds and budget

Training uses `StochasticTrack-v1`. A stochastic track is sampled on every reset
from the training environment's seeded RNG; Development is not a finite
training-track schedule.

- Conditions, in fixed reporting order: `sac-no-her`, `sac-her`.
- Training seeds per condition: 11, 29, 47.
- Budget: 300,000 native simulator transitions per seed and condition.
- Total planned policy-training budget: 1,800,000 transitions.
- Checkpoints: 50k, 100k, 150k, 200k, 250k, 300k.
- Development: tracks 2000--2008 at every checkpoint.
- Validation: tracks 3000--3008 once, using only each final 300k policy.
- Historical tracks 1000--1008: unused.
- Paper-test tracks 4000--4017: sealed and forbidden.

Development evaluation is deterministic, separate from replay and may diagnose
the preregistered run but cannot change reward, replay, hyperparameters,
checkpoint choice, budget or stopping. All six policies must finish before
Validation is opened. Validation results cannot alter another condition or
seed. There is no best-checkpoint selection or extension.

## Outcomes and analysis

The primary estimand is the paired final Validation RSR difference
`sac-her - sac-no-her` over the same 27 seed-track pairs. Report the underlying
success counts and each policy seed separately; 27 episodes are not treated as
27 independently trained policies.

Secondary outcomes are completion, deadline and strict speed compliance,
exclusive failure modes, Development RSR curves, successful training episodes,
first observed success, eligible/positive virtual-relabel rates, replay terminal
rates, and feasible-only energy. Missing feasible energy is `null`, never zero.
No unregistered favorable threshold or hyperparameter sweep is introduced.

## Stop and integrity rules

This preparation task stops before policy training. A future run must refuse an
existing or partial output directory, record actual transitions and dependency
versions, preserve both conditions after any technical interruption, and never
substitute another algorithm or reward after observing behavior. Contrastive
RL, dense shaping, demonstrations and paper-test evaluation are outside scope.
