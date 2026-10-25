# Runtime parity

Before authorization, both native workers execute the same four non-learning
controllers on two explicit, non-reserved fixed physical fixtures. Discrete
values and hashes must match exactly; floats use absolute tolerance `1e-12` and
zero relative tolerance. The tolerance is frozen in `canonical.json` before the
test and cannot be loosened after observing a discrepancy.

Production dynamics, task semantics, environment source, vehicle model, and
power artifact identities must also agree. Full machine-readable findings are
written to `runtime_parity.json`. Any failure is an integrity stop before a
scientific reset.

Preflight result: both complete trace hashes are
`43e9aa2fc4626897baad0f2af44d58cbefd4285f8361d2fd160f97d68eafcd13`.
The maximum absolute numeric difference is `0.0`; production identities and the
power-model SHA-256
`88a63fddfabf0f61123b3eee90fc84dba363911697d8e5cc10eb1685393d866d`
also match exactly.
