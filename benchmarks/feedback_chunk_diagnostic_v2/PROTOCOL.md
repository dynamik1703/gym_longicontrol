# Runtime-Isolated Diagnostic V2 protocol

The panel contains the first 36 route-clean Policy-R tasks found from numeric
seed 11,400,000. Selection uses only numeric identity and route/full-task
fingerprints. Historical opened content, registered unopened reservations,
training content, V1 Diagnostic/Validation content, internal duplicates, and
Paper numeric seeds are protected. Paper seeds 4000–4017 are never constructed,
reset, serialized, hashed, or fingerprinted.

The frozen schedule is 13 controllers by 36 tasks: CAUSAL60, TIME_AWARE,
FixedPrice MPC V2, AdaptivePrice MPC V2, H11/H29/H47, CHUNK3 seeds 11/29/47,
and RESIDUAL1S seeds 11/29/47. Only immutable final 300k endpoints are used.

All 468 pairs execute exactly once through one authoritative master ledger.
Fresh pairs become `STARTED` only after their native worker has loaded and
verified the appropriate checkpoint, and before the first reset. An ambiguous
`STARTED` interruption permanently stops execution. There is no whole-study
automatic restart.

Native evidence records actions, residuals, causal state, visible limits,
signed energy, all applicable safety checks, completion bound, viability,
interventions, cancellation, and the actual successor. A missing computation is
explicitly `NOT_APPLICABLE`; it is never converted to a successful zero. An
independent verifier reconstructs time, signed energy, right-endpoint speed
events, completion, and feasibility from evidence alone.

Raw feasibility is reported before energy. At method level, the three learned
seeds are grouped within each route. Energy comparisons use jointly feasible
routes and arithmetic means of per-route percentage savings, with medians,
absolute differences, and 10,000-resample descriptive route-bootstrap
intervals. The 108 learned outcomes are not treated as 108 independent routes.
Training-seed variability and shield/chunk behavior are descriptive secondary
outputs.

Classification is `CHUNK_V2_FEASIBILITY_REGRESSION` if a CHUNK seed fails where
its matched residual, AdaptivePrice, or FixedPrice is feasible. Otherwise,
`CHUNK_V2_PROMISING_FOR_VALIDATION` requires both frozen 1% and 3% targets;
failure to meet both is `CHUNK_V2_NO_INCREMENTAL_GAIN`. A preparation or
execution integrity failure yields `CHUNK_V2_INTEGRITY_STOP`.

Validation and Paper remain unauthorized and unopened regardless of the result.
