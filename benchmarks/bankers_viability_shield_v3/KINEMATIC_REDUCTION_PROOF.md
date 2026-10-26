# Conditional binary64 reduction

This proof precedes implementation and Development. Frozen source:
5030be8f9163b6909808626382f4a0f84c9b5d1e. The inherited conditional
simulator/sensor assumptions in V1 FORMAL_SAFETY_MODEL.md remain necessary.

In production advance, dt=0.1; v=state.velocity_m_s;
a=vehicle.acceleration_from_action(v,action), followed by exactly
min(max(a,(0-v)/dt),(37-v)/dt). Then v'=v+a*dt and
x'=state.position_m+v*dt+0.5*a*dt**2. Neither expression reads power,
energy, previous acceleration, jerk, or elapsed time. Preserve these Python
expressions, parentheses, binary64 input values, and interpolation implementation.
No reassociation, approximate stopping distance, altered time step, or monotonic
action pruning is permitted. The kernel calls the same acceleration method.
Thus structural induction over identical operations gives bitwise x/v parity
at each proposed/backup endpoint on the same audited runtime, conditional on
the omitted computations completing finitely. This is a source-level argument,
not a machine-checked Python, NumPy, BLAS or hardware theorem.

Finiteness is essential: frozen checked_advance also checks ALL auxiliary fields.
The fast domain is finite state, |x|,|time|,|energy|,|previous acceleration|,
|jerk| <= 1e100 and v in [0,37], with the frozen default BMW model. Larger,
nonfinite, custom-model, invalid action, or stale inputs use frozen certification.
This guard does not narrow admissions: it selects an implementation, not a
different safety predicate. All reachable 1800-step production states are inside
the domain by the bounds below. Hypothetical backups have at most 400 steps.

For the frozen acceleration arrays linear interpolation and the convex action
mapping keep acceleration in [-3,3], up to binary64 rounding well inside [-4,4].
With the unchanged clipping, checked production successors stay in [0,37];
the fast kernel retains its explicit x/v bounds checks. Power input absolute
bounds [38,4] cover all these calls. Each of four dense layers has width <=10.
For each layer use an outward bound B_next=2*(sum(B_i*abs(W_ij))+abs(b_j)+1).
The factor 2 and additive unit dominate binary64 multiply/add or FMA rounding
and subnormal errors for <=10 terms, including any standard BLAS summation
order; all partial sums are bounded by the absolute sum. ReLU cannot enlarge
an absolute bound. The actual frozen layers give maximum bounds approximately
67.57,309.78,2548.33,52135.26. The executable audit computes exact rational
sums with upward conversion; the theorem only needs the looser P<1e6 kW.
No overflow/NaN is possible in any dot product, bias, ReLU, or output conversion.

Energy per step is < 1e6*0.1/3600 < 28 kWh; 401 predicted steps add <11228 kWh.
Starting from |energy|<=1e100 cannot overflow (small increments may round away,
which cannot affect x/v). Jerk is <(1e100+4)/0.1 <2e101. Elapsed time adds
<=40.1 and x adds <=1484; both remain finite. Backup previous acceleration
is bounded by 4. The kernel need not compute these auxiliary values to establish
that frozen checked_advance would not reject them.

Maximum braking at v>=20/3.6-1e-6 is <=-1.4 m/s^2 on the frozen table and
interpolation, so from at most 37 m/s the threshold is reached in fewer than
226 native steps, including conservative rounding allowance; the original
400-step guard is retained. Below the threshold, repeated braking is
nonincreasing and below every legal speed limit.

Identical x/v paths, unchanged pre-action persistent envelope, bound comparisons,
frontier, numerical margins, loop termination and rejection precedence imply
identical endpoint, known-transition, unknown-preview, backup and acceptance
decisions. No newly visible observation enters the witness. Observation ingest
and sign memory are inherited without edits. The V1 invariant induction therefore
applies without a changed physical predicate. Deadline success does not follow.

Unsupported theorem extensions include altered models, runtime nondeterminism,
unmodeled state or sensing error, continuous-time speed safety and deadline
invariance. Those are not inferred from testing. Outside the reduction domain
the complete frozen implementation is retained, including its failures.
