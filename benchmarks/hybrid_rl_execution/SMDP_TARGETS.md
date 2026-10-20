# Hybrid RL V1 SMDP Signals and Bellman Targets

Let `delta = 0.99^(1/10)`, let block `n` contain `k_n` native steps, and let
`D_n` be `0` at an episode/truncation boundary and `delta^k_n` otherwise.

## Native signals and macro aggregation

For native step `i` in a block:

```text
r_i = -1000 * signed_step_energy_kWh_i
c_speed_i = max(0, endpoint_velocity_i - endpoint_speed_limit_i)  [m/s]
c_terminal_i = 1 only on an episode-ending incomplete or >140 s step, else 0
c_intervention_i = abs(proposed_native_action_i - executed_native_action_i)
```

Both arguments of `c_intervention` are dimensionless native actuator commands
in `[-1,1]`. It never compares a normalized target speed with an actuator.

Each macro signal is the within-block discounted sum:

```text
R_n = sum delta^i r_i
C_speed_n = sum delta^i c_speed_i
C_terminal_n = sum delta^i c_terminal_i
C_intervention_n = sum delta^i c_intervention_i.
```

Energy remains the reward. Speed and terminal infeasibility remain separate
zero-limit constraints. Intervention is a separate fixed-weight regularizer;
it does not change feasibility classification.

## Critic targets

With `A' ~ pi(.|S')`, automatic temperature `alpha`, twin reward targets, and
one target critic for each cost:

```text
y_R = R + D * (min_j Qbar_R,j(S',A') - alpha log pi(A'|S'))
y_speed = C_speed + D * Qbar_speed(S',A')
y_terminal = C_terminal + D * Qbar_terminal(S',A')
y_intervention = C_intervention + D * Qbar_intervention(S',A').
```

Critics minimize ordinary mean squared Bellman error. There is no mid-block
actor resampling in any target. At termination/truncation `D=0`, so the target
is exactly the accumulated block signal.

## Actor and entropy losses

For `A ~ pi(.|S)`, nonnegative multipliers `lambda_speed` and
`lambda_terminal`, and frozen regularizer weight `beta=0.05`:

```text
L_actor = E[
    alpha log pi(A|S)
    - min_j Q_R,j(S,A)
    + lambda_speed Q_speed(S,A)
    + lambda_terminal Q_terminal(S,A)
    + beta Q_intervention(S,A)
]

L_alpha = -E[log_alpha * stop_gradient(log pi(A|S) + target_entropy)]
target_entropy = -1.
```

The intervention critic makes the regularizer differentiable with respect to
the proposed strategy action while preserving its observed physical quantity.
It is not a third hard constraint and has no learned multiplier.

## Multiplier updates

Both frozen limits are zero. Direct nonnegative multipliers use projected Adam
gradient ascent, implemented by minimizing

```text
L_lambda = -lambda_speed * E[Q_speed(S,A)]
           -lambda_terminal * E[Q_terminal(S,A)]
lambda <- max(0, AdamStep(lambda, grad L_lambda)).
```

The multiplier learning rate is `1e-3`. A learned finite multiplier is an
optimization device, not a safety certificate; the native physical shield and
external exact-zero feasibility check remain authoritative.
