# Implemented discrete SAC component; not scientifically released

The provisional implementation adapts corrected Hybrid constrained SAC to a
21-category native actor. Actor, twin reward critics and three cost critics
(speed, terminal failure, intervention) each have two 256-unit ReLU layers.
Reward targets use the smaller twin Q. Critics minimize squared Bellman error.
Adam learning rates are 3e-4 for actor, critics and temperature, and 1e-3 for
projected nonnegative speed/terminal multipliers. Target tau is 0.005. Replay
capacity is 300,000; batch size is 256; deterministic CPU execution is used.

The actor minimizes the exact categorical expectation of
`alpha log pi - min(Q1,Q2) + lambda_speed Q_speed + lambda_terminal Q_terminal
+ 0.05 Q_intervention`. Each cost critic has its own target network and no
entropy bonus. Both hard-constraint optimization limits remain zero. Finite
learned penalties provide no physical safety certificate.

All arms receive the same 10D partially observed causal base: original 8D
observation, propagated position/1000 and elapsed time/140. The persistent
Banker memory is external to the actor. No Markov sufficiency claim is made
for this observation. PRE additionally restricts the actor distribution by
the deterministic causal mask. The principal adaptation replaces continuous
Gaussian sampling and sampled-action expectations by exact finite sums.

The component's finite-update and distribution tests passed on synthetic
fixtures. Independent evidence verification failed; the algorithm/configuration
was never released by a successful source/protocol freeze for training.
