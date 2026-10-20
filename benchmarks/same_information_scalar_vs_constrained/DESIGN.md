# Frozen scientific design

## Question

Can a strong fixed scalarization of exactly the same task information match explicit
constrained learning?

This does not isolate every component of constrained optimization. The arms still
differ in adaptive versus fixed weights, decomposed versus scalar critics, entropy
placement, and PID feedback. A difference must not be described as proof that the
"constrained optimizer alone" caused the outcome.

## Information parity

Both methods receive the ordinary 8D actor observation and the same transition tuple:

```text
r_E       = -signed_step_energy_kwh / 0.25
c_speed   = 0.1 * max(0, velocity - speed_limit)
T_min(x)  = optimistic remaining speed-limit-respecting travel time
deficit   = max(0, -(140 - elapsed_time - T_min(x)))
c_deadline = 0.1 * deficit / 140
```

The Scalar wrapper subclasses the canonical Constrained-V2 signal wrapper. It changes
only the returned reward to the frozen scalar combination; it preserves observations,
dynamics, termination/truncation, and the exact three signals in `info`. No progress,
completion, terminal, time, margin, or bonus term is present.

The deterministic M=information fixture covers safe/on-time, future deadline deficit,
speed violation, and combined violation transitions. It compares the shared path to
the canonical objective, speed, and deadline functions value-for-value.

## Arms

Constrained V2 reuses immutable C11, C29, and C47. Each has 300,000 real transitions,
30,000 updates, and all six scientific checkpoints.

Scalar SAC will train S11, S29, and S47 using the identical per-seed ordered canonical
stream and TRANSITIONAL update timing. It has one double-Q reward critic, no cost
critics, no PID, no multiplier, and no constrained actor objective.

## Fairness

Reference-informed fixed scalarization is used as a strong control, not as a claim
that these weights would be available a priori in a new task. The medians use all
three prior immutable final Constrained policies with no seed selection, RSR weighting,
or new Diagnostic/Validation outcome.

## Interpretation gates

Let each paired difference and the pooled difference be Constrained minus Scalar RSR
counts.

- `CONSTRAINED_ROBUST_ADVANTAGE`: pooled `>= +6/54`, no seed `< -2/18`, and at
  least two seeds `>= +2/18`.
- `SCALAR_ROBUST_ADVANTAGE`: exact symmetric rule.
- `NO_MATERIAL_DIFFERENCE`: absolute pooled difference `< 6/54` and no severe
  contradiction. Severe contradiction is frozen as at least one seed `>= +3/18` and
  another `<= -3/18`.
- Otherwise: `SEED_HETEROGENEOUS`.

These are descriptive categories only. Raw counts are always primary; no inferential
significance claim is authorized.

Interpretation is frozen as follows:

- Scalar approximately equal: much of the earlier advantage was attributable to
  task/feasibility information; do not claim constrained optimization is useless.
- Constrained robustly greater: the same feasibility information is insufficient and
  explicit constrained learning adds material robustness in this benchmark.
- Scalar robustly greater: with the same information and strong fixed scalarization,
  explicit constrained optimization is not required for superior robustness here.
- Seed heterogeneous: do not force a winner.
