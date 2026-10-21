# Frozen Comparison Design

The primary question is `CONTINUE_FEEDBACK` versus `RESET_FEEDBACK`. Both use the
same profiles, AdaptivePrice setting, terminal heuristic, native feedback and
physical authority. RESET always chooses among freshly anchored starts at each
one-second boundary. CONTINUE also scores the actual remaining suffix of the
active option. A continuation retains its anchor; a start never does.

Supporting comparisons are:

- `FROZEN_MPC_V2` versus RESET: effect of the new feedback executor and scoring
  path, with repeated one-second starts in both.
- `COMMIT_FEEDBACK` versus CONTINUE: effect of suppressing economic reselection
  until physical completion/interruption/timeout.

The identities cannot change after seeing performance. COMMIT cannot be promoted
to the primary treatment. Calibration runs exactly 48 performance episodes (the
same four arms on 12 fresh tasks) solely for semantic and resource measurement;
it performs no parameter or arm selection. Capability runs exactly 72 episodes.

For a continuation, scoring starts from current actual state, retains the original
anchor/phase and excludes incurred energy/time. A start receives a new anchor.
Both add the same inherited terminal continuation and registered lateness penalty:

```text
J = future option energy + terminal energy
    + lambda * (future option time + terminal time)
    + inherited terminal lateness penalty
```

Candidate rollout uses the executor's feedback/action-generation and Shield
semantics. Different candidate endpoints are compared only after adding the
continuation-to-task-end heuristic. These values are heuristic estimates, not
certified optima.

The preregistered label is `OPTIONS_VALID_PROMISING` only if semantic/integrity
checks pass, CONTINUE loses no feasible tasks versus RESET, and mean per-task
paired feasible-only energy saving is at least 1%. Other frozen labels are
`OPTIONS_VALID_EFFECT_UNCLEAR`, `OPTIONS_VALID_FEASIBILITY_REGRESSION` and
`SEMANTIC_OR_INTEGRITY_STOP`.

