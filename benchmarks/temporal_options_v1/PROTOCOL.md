# Frozen Protocol and Authorization

1. Base commit is `c7c6175b22e373f476aa60c10a755cb7072d9d5f`.
2. Production dynamics/power, 1000 m route, 140 s deadline, exact zero speed
   tolerance, 0.1 s integration and 1800-step outer limit are unchanged.
3. The 17 inherited profiles and AdaptivePrice `(high=24 s, low=2 s)` are not
   retuned. The terminal heuristic and safety/viability formulas are inherited.
4. OPTIONS_CALIBRATION_V1 is the first 12 clean tasks from 10700000 upward;
   OPTIONS_CAPABILITY_V1 is the first 18 clean tasks from 10800000 upward, unless
   a pre-existing namespace forces a deterministic +1000000 shift.
5. Selection protects all 2595 reachable canonical training rows, historical and
   reserved panel fingerprints legitimately present, both new panels and internal
   duplicates. Paper 4000–4017 is numeric exclusion only.
6. Calibration is limited to 48 episodes. Capability opens only if projected
   active compute is at most 8 h and retained artifacts at most 4 GB.
   The compute projection uses measured process time; any pre-Capability metadata
   correction is aggregation-only from the retained 48 records and is recorded.
7. All 72 pair identities are frozen first. One exclusive writer persists
   `STARTED` before reset and writes each completed record atomically. Completed
   pairs cannot rerun; an uncertain STARTED pair stops unless its exact full-state
   checkpoint is verified.
8. Poor performance cannot stop, retune or replace an arm. An ordinary deadline
   failure is a controller outcome, not infrastructure failure.
9. Historical Validation and Paper evaluation are unauthorized. RL training and
   every gradient update are unauthorized.
10. The experiment stops after reporting. A learned selector requires a separate
    preregistration.

Full-state checkpoints contain the environment and RNG, causal controller memory,
option anchor/profile/phase, held strategy action/clock, partial transition sums
and counters. Focused tests resume across active option execution and require exact
subsequent actions, position, energy and controller state.
