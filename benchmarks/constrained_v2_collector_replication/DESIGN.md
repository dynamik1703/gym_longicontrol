# Design

## Research question

Under identical Constrained-V2 science, seed-paired learner initialization, and
route-clean training stream, is episode-wise or transition-wise collection more
robust across seeds 11, 29, and 47?

The only treatment is the collector/update schedule.

| ID | Nominal seed | Collector |
|---|---:|---|
| E11 | 11 | EPISODIC |
| T11 | 11 | TRANSITIONAL |
| E29 | 29 | EPISODIC |
| T29 | 29 | TRANSITIONAL |
| E47 | 47 | EPISODIC |
| T47 | 47 | TRANSITIONAL |

## Collector semantics

`EPISODIC` uses the exact verified V3 historical-collector ordering: one actor
state collects a complete real episode, including its terminal replay
transition; PID updates at terminal; the accumulated update budget then runs;
the next episode starts with the updated actor.

`TRANSITIONAL` inserts each real transition, updates PID first if the transition
terminates a real episode, catches RL updates up to
`floor(real_transitions × 0.1)`, and continues with the potentially updated
actor.

## Frozen science

The algorithm remains SACLag with gamma 0.99, two-step targets, batch 256,
replay 100,000, 128×128 actor and critics, actor LR 0.0005, critic LR 0.001,
automatic alpha initialized at 1.0, target entropy -1, alpha LR 0.0003, tau
0.05, PID `(0.05, 0.0005, 0.1)`, zero cost limits, dt 0.1 seconds, and a maximum
of 1800 native steps per episode. Objective and cost definitions are inherited
unchanged from Constrained V2. There is no model, synthetic replay, MVE,
physics augmentation, reward shaping, speed margin, progress reward, or altered
PID/alpha behavior.

## Primary outcome

Validation RSR is compared within each seed and pooled across 54 episodes per
collector. Completion, deadline compliance, and speed compliance receive the
same paired summaries. Interpretation uses only the preregistered four-category
rule in `PROTOCOL.md`; it is descriptive rather than a significance test.

Mechanism evidence records actor-version stationarity, PID timing, replay size
and transition number at update events, and movement among standstill,
crawling, timely-completion, and speed-violation states without changing
training.
