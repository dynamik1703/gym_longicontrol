# BankerShield V1

A new isolated simulator study of causal native-action admission with an exact maximum-braking continuation witness. Read SCIENTIFIC_DECISION.md and DEVELOPMENT_RESULTS.md for the completed outcome, and ROBUST_INVARIANCE_PROOF.md for the conditional theorem.

Only this package is changed. Frozen base 77e440f4a2387d5810251d7bc23fcc094aad9730; branch research/bankers-viability-shield-v1. No learner, gradient update, historical retraining, new Validation, or Paper evaluation is authorized.

Commands from the repository root with the audited Python runtime:

```sh
PYTHONPATH=src:. python -m pytest benchmarks/bankers_viability_shield_v1/tests -q
ruff check benchmarks/bankers_viability_shield_v1
PYTHONPATH=src:. python -m benchmarks.bankers_viability_shield_v1.evaluation preflight
PYTHONPATH=src:. python -m benchmarks.bankers_viability_shield_v1.evaluation develop
```

Scientific runs are immutable and exactly once. Do not rerun completed fixtures or Development pairs. An ambiguous STARTED pair stops execution. The preflight and Development entrypoints enforce source/configuration identity and gates; the published evidence is sufficient for offline verification without performance reruns. Model assumptions, grid limitations, deadline separation and future RL design are documented in the accompanying artifacts.
