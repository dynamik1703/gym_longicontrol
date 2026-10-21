# Research Question and Preregistered Interpretation

## Question

Can a causal RL policy learn a state-dependent time shadow price
`lambda_t >= 0` in kWh/s that improves feasible energy over a strong classical
AdaptivePrice MPC, when all downstream planning, dynamics, safety and
information are identical?

## Main hypothesis

A dynamic price may allocate scarce deadline reserve more effectively than a
constant price or a monotone hand-designed reserve law. The policy does not
choose target speed or actuator force; it chooses only the scalar price in
`E_100m + lambda_t T_100m + V_terminal`.

## Preregistered hierarchy

1. No loss of feasible coverage versus the stronger non-learning MPC baseline.
2. Paired feasible-only energy savings versus AdaptivePrice MPC: 3% is
   meaningful and 5% is strong, consistently across all three policy seeds.
3. The learned price must vary with causal state, materially change executed
   physical decisions, and not merely reproduce a constant price or fallback.

Feasibility dominates energy. No energy reduction can offset a new task
failure. The privileged full-route reference is contextual and is neither
information-matched nor a certified optimum.

## Preflight outcome

The price interface is non-degenerate, but the classical MPC foundation is not
strong enough: AdaptivePrice MPC underperformed TIME_AWARE by 9.301% energy on
the independent Development capability partition and was mostly
fallback-equivalent. Training would answer a different, weaker question than
the preregistered one. The study therefore stops before main RL training.
