# Zero-Residual Parity

Status: **deterministic implementation test passed; full preflight artifact
pending**.

For `CHUNK3_RESIDUAL_SAC` with actor action `[0, 0, 0]`, the wrapper calls the
frozen AdaptivePrice MPC V2 at its original one-second boundaries, applies no
actuator change, and delegates executed-action propagation to the original MPC
method. The additional residual predicates are observational only in this case.

The deterministic test compares the baseline and wrapper with exact Python
numeric equality for every native actuator, environment position, velocity,
acceleration, cumulative signed energy, elapsed time, current speed limit, full
MPC decision (including replanning and every safety predicate), and terminal
metric. The initial suite covers constant high limit, maximum legal downward
change, consecutive downward changes, and a transition at the route endpoint.

The test command completed with 7/7 tests passing. The final preflight will
materialize per-fixture hashes and extend the physical gate report before main
training is enabled.

