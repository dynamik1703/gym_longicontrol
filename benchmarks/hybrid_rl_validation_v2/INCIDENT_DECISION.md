# Hybrid RL Validation Incident Decision

`HYBRID_RL_VALIDATION_V1` is permanently **INCIDENT_ABORTED**. It is rejected
as the final performance panel and replaced by `HYBRID_RL_VALIDATION_V2`
solely because a documented evaluation-infrastructure error opened two
baseline episodes before the required exactly-once ledger existed.

This amendment occurs after Hybrid training completed. No learned Hybrid
policy was evaluated on V1, and no numerical result from either accidental
baseline episode was persisted. Numerical performance must not be inferred or
reconstructed from those episodes, and the two pairs must not be replayed.

All five controller identities remain unchanged: `CAUSAL60`, `TIME_AWARE`,
`H11`, `H29`, and `H47`. The three learned actors remain frozen at 300,000
native transitions and 30,000 updates. The SMDP formulation, Shield, fallback,
training streams, metrics, paired-feasible analysis, and acceptance thresholds
remain unchanged.

V2 is selected deterministically from content fingerprints, without
performance information. Neither controller nor hyperparameter selection uses
the incident. Future reporting must call V2 an **amended post-incident
validation**, not the original untouched preregistration, and must disclose:

- V1: two completed unpersisted baseline episodes, one Hybrid reset, zero
  Hybrid actions, zero completed Hybrid episodes, zero registered pairs;
- V2: a new independently frozen 36-task panel, initially zero performance
  episodes, with separate authorization.

The original V1 ordered SHA-256 remains
`f3864fed8725874dc315004375af0d2850dc2d2d56957d52c64d043ca6b3352b`.
