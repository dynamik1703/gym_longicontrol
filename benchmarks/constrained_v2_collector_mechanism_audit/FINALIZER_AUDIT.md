# Finalizer recovery audit

Classification: **`REPORTING_ONLY_RECOVERY_CONFIRMED`**.

All six final checkpoint files hash to their already-frozen manifest values, all
six Attempts ended before Validation opened, each Validation actor hash matches its
final/300k actor before and after evaluation, and each retained Validation artifact
contains exactly 18 episodes. The manifest records 108 completed episodes and Paper
remains `SEALED_UNINSPECTED`.

The recovery module calls deterministic `finalize.build_results` after replacing
only the known-bad in-memory object-order assertion. It contains no training or
evaluation call and reports no repeated training/evaluation or sealed-artifact
mutation. Source hashes are recorded in `results.json`.

- `six_final_checkpoint_hashes_verified`: `true`
- `all_attempts_ended_before_validation_opened`: `true`
- `one_attempt_per_policy`: `true`
- `no_resume_events`: `true`
- `validation_manifest_complete_108`: `true`
- `one_sealed_validation_file_per_policy`: `true`
- `all_validation_actor_hashes_unchanged`: `true`
- `recovery_source_has_no_training_or_evaluation_call`: `true`
- `recovery_provenance_says_no_repetition_or_mutation`: `true`
- `paper_still_sealed`: `true`

Qualification: The retained manifest is a single state record rather than an external append-only invocation log. Its timestamps, six one-pass artifacts, hashes, and aggregation-only recovery source are mutually consistent.

The mutually consistent immutable evidence supports reporting-only recovery; no
policy, split, sealed evaluation artifact, or scientific result changed.
