# Safety and Deadline Viability

Every proposed native action is checked using the same frozen Hybrid physical
logic:

1. exact next-state propagation and right-endpoint speed limit;
2. exact maximum-braking feasibility for every known lower transition;
3. braking protection against an unseen 20 km/h limit within the 150 m sensor
   envelope;
4. the existing pessimistic completion bound plus reserve;
5. native action bounds.

Failure replaces the proposal with the unchanged CAUSAL60 fallback. Controller
memory is propagated from the action actually executed. The fallback itself is
not weakened and no safety/deadline margin was tuned on Validation.

The completion bound remains an empirically tested causal recoverability
estimate, not a formal viability theorem. Any future counterexample is an
integrity stop.

The bounded adversarial test crossed a newly visible 60-to-20 km/h transition,
triggered replanning, completed within 140 s and recorded exactly zero maximum
speed violation. Capability evaluation had 20 deadline-recoverability
overrides and no optimizer infeasibility failures. Exact zero tolerance remains
authoritative through `is_feasible`.
