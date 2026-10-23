# Physics and Safety

The baseline is the frozen AdaptivePrice MPC V2 with reserve law `24 s / 2 s`,
one-second replanning, and its inherited CAUSAL60 fallback. Residual execution
uses the same right-endpoint speed, remembered-transition braking,
unknown-preview braking, and pessimistic deadline-viability predicates. No
margin, tolerance, preview, deadline, vehicle model, or fallback behavior is
relaxed.

The residual gate is an additional wrapper. It first obtains the complete MPC
decision, refuses influence when MPC uses fallback, evaluates the combined
proposal, and propagates controller memory from the action actually executed.
Zero residual must reproduce every MPC action/state/metric exactly.

The shield is not claimed to prove global deadline feasibility. External task
success (`completed`, deadline at 140 s, exact zero speed excess) remains
authoritative.

Every native evidence row retains the MPC proposal and decision, requested
residual, combined proposal, executed action, state, energy, visible limits,
each gate outcome, `completion_bound_s`, acceptance/reason, next state, and
termination. Compressed evidence receives a SHA-256 digest.

