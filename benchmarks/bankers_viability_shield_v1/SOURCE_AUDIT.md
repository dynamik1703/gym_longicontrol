# Frozen source audit

Base: `77e440f4a2387d5810251d7bc23fcc094aad9730`. Scope: every tracked file in the production package and the eight inherited research/audit packages listed in `source_audit.json`. 296 immutable files, including the original BMW-i3 artifact, task semantics, historical implementations and records. Ordered source digest: `21886a5650304452b00a7847667260ee0ba90fbad62cb24756924b0d614c8c85`. Before and after execution every file is rehashed and compared. No historical classification is rewritten.

Reviewed production `advance`, action interpolation, observation normalization, two-sign sensor, right-continuous limit sensing, endpoint metrics and exact task evaluator. Reviewed Hybrid endpoint/braking/preview/deadline checks; causal and oracle references; MPC V2 controller/design/safety and planner; Feedback Chunk native action contract/safety and diagnostic execution/split protocol; paper discrete infeasibility proof.

Findings: braking depends on speed, zero action coasts, maximum braking approaches rest asymptotically near zero. The generator minimum is binary64 `20/3.6`, sign spacing at least 100 m, and legal limits are 20–100 km/h in steps of 10. The sensor returns only two signs. At most two generator signs occur within 150 m. At exactly 150 m a differing limit distinguishes a sign from a placeholder. Normalization requires decoding legal limits to their exact generator values, with a checked decoding residual.

The inherited shield uses 60 km/h as its proposal cap, 0.5 m braking margin and a separate heuristic completion estimate. Its preview brake test starts from the successor but reuses the full 150 m; it does not explicitly subtract the just-executed displacement. None of these observations alone establishes a counterexample. No inherited global controlled-invariance proof is assumed.

Verified remote `gym_longicontrol_origin` = `https://github.com/dynamik1703/gym_longicontrol.git`. The main checkout contains unrelated Trading Dashboard changes; it remains untouched. Only the new isolated research package may change.
