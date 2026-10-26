# BankerShield V1

Final classification: **BANKERS_PRE_SHIELD_INTEGRITY_STOP**. The original preflight readback checker failed on JSON tuple/list representation. The retained 1800 native records now pass independent replay; the original STARTED record remains and Development was never opened. See EXECUTION_INCIDENT.md, SCIENTIFIC_DECISION.md and DEVELOPMENT_RESULTS.md. The conditional theorem is in ROBUST_INVARIANCE_PROOF.md.

Only this package is changed. Frozen base 77e440f4a2387d5810251d7bc23fcc094aad9730; branch research/bankers-viability-shield-v1. No learner, gradient update, historical retraining, new Validation, or Paper evaluation is authorized.

Read-only verification commands from the repository root with the audited Python runtime:

```sh
PYTHONPATH=src:. python -m pytest benchmarks/bankers_viability_shield_v1/tests -q
ruff check benchmarks/bankers_viability_shield_v1
```

Scientific runs are immutable and exactly once. Do not rerun completed fixtures or Development pairs. An ambiguous STARTED pair stops execution. The preflight and Development entrypoints enforce source/configuration identity and gates; the published evidence is sufficient for offline verification without performance reruns. Model assumptions, grid limitations, deadline separation and future RL design are documented in the accompanying artifacts.

The evaluation entrypoints remain implemented but block this recorded stop. Do not invoke them to repair the result. Final code includes the readback correction and stop guards; retained certificates identify execution commit 973be83 separately.
