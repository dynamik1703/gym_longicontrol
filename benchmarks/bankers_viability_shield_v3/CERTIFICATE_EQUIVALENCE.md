# Exact safety-decision equivalence
The reference is V2's import of V1 BankersShield at frozen source commit
5030be8f9163b6909808626382f4a0f84c9b5d1e. V3's safety_view compares every
safety field, including rejection precedence, margins, old-frontier availability,
input identity, exact x/v successor and terminal, backup existence and length.
No numerical tolerance is used. Bitwise x/v tests additionally compare float.hex.
Energy-only hypothetical states and their historical digests are intentionally
absent from schema bankers-speed-certificate-v3.0. There is no full-byte claim.
The compact transcript is network-order binary64 (position, velocity) per backup
endpoint, base64 encoded with SHA-256 over the decoded bytes. Actual native state,
power, signed increment and full selected production certificate remain recorded.

KINEMATIC_REDUCTION_PROOF.md establishes the conditional source-level reduction.
Finite differential testing is supporting evidence, not a universal implementation
theorem or exhaustive binary64-domain check. No finite-domain exhaustiveness is
claimed. Differential/adversarial JSON files declare the exact empirical domain.
The first mismatch must be saved as first_equivalence_counterexample.json and
classification BANKERS_V3_EQUIVALENCE_STOP; thresholds must not be altered.
Unsupported reduction cases retain the complete frozen certifier.
