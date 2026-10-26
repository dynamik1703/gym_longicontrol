# Original preflight integrity event

Final study classification: BANKERS_PRE_SHIELD_INTEGRITY_STOP.

Execution source commit `973be83` ran the offline falsification search and the
constant-20 fixture. The first fixture was durably STARTED before reset. It
executed 1800 native steps, wrote atomic compressed full evidence, and passed
in-memory independent physics/mask/certificate replay. The next gate compared
JSON-decoded records directly with their Python originals. Tuples in the
lower-envelope/sign memory deserialize as lists, so Python equality failed even
though the canonical JSON content is identical. This is an implementation
error in the readback checker, not an observed physical safety counterexample.
The stack trace and original fail-closed stop are preserved in integrity_stop.json
and preflight_execution.log. No Development task was generated or evaluated.

The retained evidence SHA-256 is `a9e6d83a448f38bf85e0965302411a640a9e95391d9d52e96f7d6f41caaf6ea6`. Its complete
1800-row observation/action/physical-state chain and all 21-point masks have
subsequently been verified independently from disk. This verifies the retained
scope only. It does not retroactively declare the original scientific gate
passed. The original fixture ledger still says STARTED; the seven other
fixtures remain NOT_STARTED. No reset, controller execution or policy rollout
was repeated to repair the outcome.

The fix compares canonical content digests across JSON representation changes.
The independent verifier uses the same representation-stable comparison for
reconstructed memory. Added regression tests reject missing native rows,
missing applicable deadline bounds, tampered physical successors, and attempts
to open Development after the recorded stop. The first integrity incident and
falsification artifact cannot be overwritten by a later blocked invocation.
The fixture guard runs before any repeated search or reset.

Execution source/configuration identity is preserved in every selected
certificate and retained_fixture_verification.json. The final implementation
identity differs because the readback/verifier and stop guards were corrected.
Physical admission, production dynamics, generator, power model, causal
lower-envelope construction and action grid are unchanged. results.json retains
both identities rather than relabeling old evidence as final-code execution.

This work does not provide an approved resume/recovery protocol. A separate,
explicitly reviewed scientific continuation would have to account for this
ledger and source boundary before filling the missing fixtures or opening a
fresh Development study. No future training or evaluation is authorized here.
