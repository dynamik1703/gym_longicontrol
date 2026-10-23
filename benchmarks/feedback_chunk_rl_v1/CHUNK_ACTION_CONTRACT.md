# Chunk Action Contract

At an actor boundary the policy observes one finite causal 16-vector and emits
the complete action. `CHUNK3_RESIDUAL_SAC` emits `(delta_0, delta_1, delta_2)`
in `[-1,1]^3`; `RESIDUAL1S_SAC` emits one value in `[-1,1]`.

For native step `n` in a chunk beginning at `n0`,
`j = floor((n - n0) / 10)` and the proposal is
`clip(u_mpc + 0.25 * delta_j, -1, 1)`. MPC replans every ten native steps and
the physical gate runs every native step. There is no actor resampling within a
chunk.

A newly visible transition or rejected residual cancels all later residuals in
the current chunk. The frozen MPC then controls the remainder; actor decisions
resume only at the scheduled boundary. A frozen-MPC fallback is never
overridden. True termination shortens the block. The final 300k dataset boundary
is a nonfailure truncation.

Replay stores the originally proposed actor vector. It never substitutes an
executed actuator command for that action.

