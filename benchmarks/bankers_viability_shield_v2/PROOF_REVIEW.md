# Conditional proof review
The frozen V1 ROBUST_INVARIANCE_PROOF.md, FORMAL_SAFETY_MODEL.md, BACKUP_POLICY.md,
SAFETY_COMPARISON.md, DEVELOPMENT_PROTOCOL.md, SCIENTIFIC_DECISION.md and their
executable certificate, safe set, propagation, evaluation and verification sources
are authoritative inherited inputs at 0162477848bcfef2d1acea3e38a47b8fbd539299.
V2 imports their safety algorithm unchanged. No new physical hypothesis or
numerical margin is introduced.

The exact production generator uses legal limits 20..100 km/h and transitions
at least 100 m apart. Two slots exhaust a 150 m preview. The sensor's right
continuity and normalization agree with the observed-limit inversion. Exact
state is propagated from reset and actual native actions. Model and power
artifact hashes, Python/NumPy runtime and source identity are part of the claim.

Each observation yields a conservative lower limit through the old absolute
frontier. The pointwise maximum of valid lower bounds remains a lower bound;
new observations can only increase it at future reachable positions. Enclose
sign-reconstruction roundoff by 1e-7 m and tighten velocity by 1e-6 m/s while
leaving task speed tolerance at zero. Nondecreasing position permits pruning
strictly past constraints. The proof assumes valid noiseless original sensors,
identical binary64 physics, finite state/energy and no unmodeled disturbance.

Define K by current compliance and the complete maximum-braking suffix until
v <= 20/3.6-1e-6. The negative-acceleration table is nonpositive on [0,37]
and strictly negative above that threshold. The finite suffix reaches the
universal safe tail; exact rest in finite time is unnecessary. Reset is in K.
Admission checks the exact successor and every suffix endpoint under the old
envelope. A refreshed observation preserves that suffix. Hence the successor
is in K and action -1 remains certified: its witness is the remaining suffix.

Classification: FORMALLY_SUPPORTED_UNDER_STATED_MODEL for this conditional
right-endpoint speed-invariance argument. Numerical fixtures and replay check
software behavior but do not constitute a machine-checked proof of Python or
IEEE-754 execution. There is no real-noise/model-mismatch guarantee, continuous
inter-step safety theorem, deadline guarantee, maximal-set characterization or
global energy optimality certificate.
