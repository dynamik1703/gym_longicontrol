# Constrained RL V2 reproducibility audit

This directory contains a diagnostic-only forensic comparison of the historical
Constrained RL V2 result (Validation RSR 21/27) and the Multi-Step Physics MVE
`matched_real_only` control (10/27).

## Conclusion

The primary classification is **C — RNG/trajectory mismatch**, with high
confidence that the runs are not seed-paired reproductions. The first training
track already differs at episode index 0 for seeds 11, 29, and 47. A second,
independent high-confidence finding is **B — end-to-end implementation mismatch**:
the historical runner collects a whole episode under a fixed actor and then makes
bulk updates, whereas the current runner collects and updates transition by
transition. Their reset consumption and PID/update ordering also differ.

No scientific parameter or environment/task implementation mismatch was found.
The shared scientific source files are byte-identical between the earliest
historical archive commit and the audit start commit. Both result sets remain valid
observations of their recorded policies, but the current control is not an exact
end-to-end reproduction of the historical run.

Confidence that the identified differences fully explain the numerical 21/27 to
10/27 gap is **medium**, because no new full training was authorized and the exact
historical NumPy, Torch, and Gymnasium versions and initial RNG states were not
recorded.

## Scope and safety

- LongiControl simulator transitions consumed by the audit: **0**.
- New scientific training runs: **0**.
- Development, Validation, and paper-track evaluations: **0**.
- Historical run artifacts modified: **none**.
- Completed research branches modified: **none**.

The fingerprint ladder uses unreserved audit seed 0, confirms identical network
construction under one runtime, observes divergence at the first generated track,
and stops before an action or simulator step. Track-sequence reconstruction uses
environment resets only. Collector tests use a synthetic dummy environment.

## Evidence map

- [PROVENANCE.md](PROVENANCE.md): source, artifact, checkpoint, and runtime lineage.
- [CONFIG_DIFF.md](CONFIG_DIFF.md): scientific parameters and evaluation parity.
- [RNG_AUDIT.md](RNG_AUDIT.md): RNG ownership, track fingerprints, initialization,
  and earliest divergence.
- [IMPLEMENTATION_DIFF.md](IMPLEMENTATION_DIFF.md): collector, replay, n-step,
  PID, alpha, and scheduling semantics.
- [RESULTS.md](RESULTS.md): final A–F classification, confidence, and minimal future
  reproduction.
- [results.json](results.json): machine-readable evidence, including the first 100
  reconstructed track fingerprints per seed and raw artifact hashes.
- [harness.py](harness.py): bounded deterministic fixtures.
- [generate_results.py](generate_results.py): evidence materialization from retained
  artifacts.

## Reproduction of the audit artifact

From this worktree, using the retained MVE environment:

```bash
PYTHONPATH=. /Users/romanliessner/gitlab/gym_longicontrol_multistep_mbrl/.venv/bin/python \
  -m benchmarks.constrained_v2_repro_audit.generate_results \
  --repository /Users/romanliessner/gitlab/gym_longicontrol_repro_audit \
  --historical-runs /Users/romanliessner/gitlab/gym_longicontrol_task/runs/constrained-rl-v2-20261002 \
  --current-runs /Users/romanliessner/gitlab/gym_longicontrol_multistep_mbrl/runs/multistep-physics-mbrl-v1 \
  --output benchmarks/constrained_v2_repro_audit/results.json
```

The historical track fingerprints are explicitly labeled reconstructions. They are
source-exact under the installed current runtime, but cannot be claimed as recovered
historical bytes because the historical NumPy/Gymnasium versions and environment RNG
state were not persisted.
