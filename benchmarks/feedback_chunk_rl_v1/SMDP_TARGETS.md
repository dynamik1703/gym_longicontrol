# SMDP Targets

Let `gamma_decision = 0.99` per second and
`gamma_native = 0.99 ** (1/10)`. For a macro block of `k` native steps and
native signal `z_i`, store `Z = sum(gamma_native**i * z_i)`.

For a valid nonterminal continuation the discount is `gamma_native**k`; for a
true episode termination or environment truncation it is zero. The exact final
training-budget boundary is recorded as dataset truncation, not terminal task
failure, and also has no bootstrap because no successor belongs to the frozen
dataset.

Reward target:

`R + D * (min(Qbar1, Qbar2) - alpha * log_pi)`.

Each cost target:

`C + D * Qbar_cost`.

CHUNK3 normally has `k=30`, RESIDUAL1S normally has `k=10`, and true final
episode blocks may be shorter. No CHUNK3 target bootstraps from a one-second
observation.

