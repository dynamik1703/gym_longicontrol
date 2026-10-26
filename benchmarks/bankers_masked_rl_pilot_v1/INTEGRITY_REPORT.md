# Failed preflight and containment

The original preflight result is 12 tests: 9 passed and 3 failed. The three exact
failure identities are:

- `test_native_evidence_and_recovery[RAW_DISCRETE_SAC]`
- `test_native_evidence_and_recovery[BANKER_POST_SAC]`
- `test_native_evidence_and_recovery[BANKER_PRE_SAC]`

All failed at `verification.py:40`, where the pilot omitted the previous memory
argument required by the unchanged inherited envelope verifier. No independent
native evidence row was accepted by these tests. Checkpoint continuation
assertions earlier in these failed tests do not turn them into gate passes.

`preflight_tests.xml`, `preflight_results.json` and `tested_source_manifest.json`
bind the exception and tested source bytes. The failed implementation is retained
without correction. Separate post-stop containment checks may verify blocked
entry points and preservation of these bytes; they never rerun the failed gate.

There were zero scientific Attempts, training transitions, training optimizer
updates or Diagnostic pairs. Complete resource measurement and source/protocol
freeze were not reached. All scientific gates are closed after this event.
No complete end-to-end recovery, scientific publication transaction, independent
training audit or Diagnostic verifier is claimed to be implemented.

The inherited 265 hashes and V4's 397 manifest files matched during the audit.
Changes are confined to the new study package. V1–V4 results, classifications
and historical incidents are unchanged. Historical Validation was not executed
or reconstructed by this pilot, and Paper 4000–4017 remains SEALED_UNINSPECTED.

Post-stop containment: two checks passed, verifying fail-closed entry points
and preservation of the original failed source/XML. These checks do not
change the scientific preflight result of 9 passed / 3 failed.
