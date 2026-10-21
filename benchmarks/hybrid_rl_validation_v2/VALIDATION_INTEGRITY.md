# Hybrid RL Validation V2 Integrity

V2 is an explicitly amended post-incident panel. V1 remains permanently `INCIDENT_ABORTED`; Paper remains `SEALED_UNINSPECTED`.

Transaction: `439c821bc40c0a4dd8962af0ce08ef050f3ca697af15d8960a7340c8f860ab76`
Split: `99a863fb423e2ceb5ec0641bb940fab21ff7eb86a1466886da8707862a015e74`
Result set: `735e379a05f43073c11ccf17f8bf397160183483527e9526c2b6f3d4ac9e4614`

## Final checks

- PASS — `exactly_180_unique_completed_pairs`
- PASS — `no_pending_started_or_interrupted_pairs`
- PASS — `ledger_completed_sealed`
- PASS — `lease_completed_sealed`
- PASS — `authorization_disabled_after_completion`
- PASS — `five_frozen_controller_identities`
- PASS — `split_unchanged`
- PASS — `transaction_id_unchanged`
- PASS — `checkpoint_hashes_unchanged`
- PASS — `actor_hashes_unchanged`
- PASS — `training_counters_unchanged`
- PASS — `scientific_sources_unchanged`
- PASS — `v1_remains_incident_aborted`
- PASS — `paper_sealed_uninspected`
- PASS — `result_hashes_valid`
- PASS — `zero_deadline_bound_falsifications`

The transaction used one durable, non-reclaimable exclusive lease. Every pair was registered before evaluator entry, durably marked `STARTED` before reset, completed once, read back, and sealed with its result SHA-256. The completed ledger and lease permanently block another execution.
