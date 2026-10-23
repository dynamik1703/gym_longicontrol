# Zero-Residual Parity

Status: **full deterministic preflight passed**.

For `CHUNK3_RESIDUAL_SAC` with actor action `[0, 0, 0]`, the wrapper calls the
frozen AdaptivePrice MPC V2 at its original one-second boundaries, applies no
actuator change, and delegates executed-action propagation to the original MPC
method. The additional residual predicates are observational only in this case.

The deterministic test compares the baseline and wrapper with exact Python
numeric equality for every native actuator, environment position, velocity,
acceleration, cumulative signed energy, elapsed time, current speed limit, full
MPC decision (including replanning and every safety predicate), and terminal
metric. The final suite covers constant high limit, upward changes, maximum
legal downward change, consecutive downward changes, a short braking window,
and a transition at the route endpoint.

All six baseline/wrapper trace hashes matched across 5,366 native steps. The
complete per-fixture hashes are retained in `preflight_results.json`. The full
test command completed with 13/13 tests passing. Main training was enabled only
after parity and every other mandatory preflight/resource gate passed.

