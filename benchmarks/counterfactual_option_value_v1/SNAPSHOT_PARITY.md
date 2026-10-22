# Snapshot Parity

Unreserved fixture seed: `12345678` at `430 m`.

- snapshot_serialization_deterministic: `True`
- snapshot_restore_exact: `True`
- two_reference_continuations_identical: `True`
- baseline_action_injection_exact: `True`
- alternative_injection_isolated: `True`
- option_memory_survives_restore: `True`
- all_native_audits_pass: `True`
- active_compute_projection_pass: `True`
- retained_artifact_projection_pass: `True`

The alternative branch began from identical physical, observation, RNG,
controller and option memory. Only its first high-level decision was
overridden; later decisions again used frozen causal CONTINUE_FEEDBACK.
