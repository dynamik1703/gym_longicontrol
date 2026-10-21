# Counterfactual Option Value V1

Bounded internal pilot asking whether causal observations can identify beneficial
single START/CONTINUE deviations from frozen `CONTINUE_FEEDBACK`.

The package preserves `StochasticTrack-v1`, production BMW i3 dynamics, native
0.1-second feedback and the inherited physical Shield. It performs no full RL
training and does not interact with Paper or historical Validation content.

Execution is fail-closed: protocol and route-clean splits are frozen first;
unreserved fixtures must pass snapshot, injection, replay and resource gates;
each scientific branch is then exactly-once and hash-bound. Test predictions are
persisted and hashed before Test outcomes are scored.

