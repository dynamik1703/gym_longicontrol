# Constructive robust continuation proof

Classification: FORMALLY_SUPPORTED_UNDER_STATED_MODEL for the conditional
simulator speed claim below. This is a mathematical proof with an executable
witness checker, not a machine-checked proof of Python or of IEEE-754 hardware.
The conditional assumptions are those in FORMAL_SAFETY_MODEL.md.

**Lower-envelope lemma.** Each complete two-sign observation specifies the
limit from the current position through its 150 m preview. Normalization is
inverted to the supported exact speed menu. Within the position enclosure of a
transition, the minimum of adjacent limits is a valid lower bound. Outside the
observed prefix use the universal minimum. Therefore every observation yields
a lower bound for every route in W. The pointwise maximum of all such bounds
is still a lower bound. Adding observations increases this envelope on every
future position. Past-position pruning is harmless because production position
is nondecreasing. Remembering only currently visible signs without retaining
past future constraints would not prove this monotonicity.

**Backup-tail lemma.** Production maximum braking is nonpositive at every
speed in [0,37]. This follows directly from its tabulated negative-acceleration
values and their linear interpolation, and from the within-step lower velocity
clip. For speeds >=20/3.6-1e-6 it is strictly negative and bounded away from
zero: below 1 m/s it is -3v; from 1 to 37 the tabulated negative accelerations
are at most -1.4 m/s² (the executable audit checks the actual extrema).
Hence it reaches the minimum threshold in finitely many steps. Once below
that threshold repeated braking is nonincreasing, nonnegative, and below every
legal limit. The theorem never requires reaching exact rest in finite time.
Signed power/energy and previous acceleration/jerk do not enter the next speed
or position dynamics. All computed state/energy values must remain finite.

**Definition of K.** The current endpoint and every maximum-braking endpoint
until that universal threshold obey the persistent lower envelope minus
1e-6 m/s. Lower bounds are no larger than the actual limit on every W route.
Thus the single policy pi_B(z)=-1 for all z is a speed-safe witness from K for
all those routes. It is causal even though it needs no new observations. It
continues to be safe when future signs become visible.

**Base.** Reset velocity is zero. Applying -1 leaves position and velocity at
zero. Every supported initial limit is >=20/3.6, so reset is in K. Stopped
states are safe even when they sacrifice completion.

**Admission.** For action a compute the exact production successor s+. Check
its right endpoint using the PRE-ACTION envelope and check every endpoint of
B(s+) until the universal threshold. The finite witness must be fully checked,
including a sign crossed inside a native integration step. This is stronger
than checking only a stopping distance or a single next speed. Accept exactly
when these checks pass; failed/nonfinite predictions are rejected.

**Induction.** The accepted successor's current endpoint and entire B suffix
were checked under the old envelope. Upon receiving the next observation, the
new envelope dominates the old one at every position still reachable from s+.
The same exact B suffix therefore proves s+ in K. In particular action -1 is
itself admitted there: its endpoint is the first B endpoint and its witness is
the remaining suffix. If already below the threshold, -1 remains below it.
Consequently every reachable certified state has at least the grid action -1.
Mask emptiness or final-verification disagreement is an integrity stop, never
an occasion to insert an unchecked fallback.

**Consequence and quantifiers.** From reset, arbitrary repeated choices among
certified actions preserve original exact right-endpoint speed compliance for
every continuation in W. The same single causal pi_B witnesses all of them.
The proof makes no claim of maximal action permissiveness, unconditional task
success, continuous-time speed safety, deadline satisfaction, physical model
robustness beyond the stated model, or an invariant for the inherited shield.
The 21 samples certify only those samples, not intervals between them.

**Latency and numerics.** The old preview ends at x_t+150, not x_next+150.
Every proposed step is checked before observing again and its backup uses that
same old absolute frontier. Therefore one-step sensing/reaction loss is already
paid in distance. Transition enclosures cover reconstruction roundoff and the
speed margin tightens the exact task bound. Physics propagation is exactly the
same production function on identical binary64 inputs, so witness suffixes
are reproduced by deterministic execution without a surrogate or an assumed
continuous braking-distance formula. A different runtime, noisy state estimate,
changed source/configuration or incomplete sensor model invalidates this claim.
