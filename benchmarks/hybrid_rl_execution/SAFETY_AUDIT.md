# Hybrid RL V1 Safety and Viability Audit

The frozen causal controller and Hybrid shield were not modified. This
audit is deterministic simulator evidence, not a formal global safety proof.

- Extended adversarial cases: `7`.
- Exact right-endpoint speed compliance: `True`.
- Eventual completion: `True`.
- Deadline compliance: `True`.
- Deadline-bound falsifications: `0`.
- Stress coverage: repeated maximum legal reductions, slow sections,
  consecutive transitions, near-deadline/20 km/h proposals, gradual
  reserve consumption, fallback after accepted slow actions, low-speed
  acceleration, and an endpoint-interrupted macro block.

Frozen six-fixture shield reproduction:

- override fraction: `0.175975818110`;
- native-action influence fraction: `0.633854645814`;
- influenced fixtures: `6`;
- viability falsifications: `0`.

Any future falsification is a terminal gate failure; margins, deadline,
or zero speed tolerance may not be relaxed to rescue the study.
