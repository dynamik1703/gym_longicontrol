# Preserved conditional safety meaning

Reference commit b60549e95e31601039329ca8de041977da0155fe, tree
41ec3d0f89bde5b2968cec909cb84aed05bb01d8. V3 fast_certificate.py,
fast_kinematics.py, safe_action_mask.py, all original physical production sources,
causal ingest/envelope and controller selection laws remain untouched. V4 calls
the frozen V3 certifier once per candidate and applies a storage projection.
Therefore every retained V3 safety predicate, numerical value, reason and input
identity is unchanged by construction. The mask is computed once per native
decision; the off-grid continuous-proposal diagnostic and full selected-action
verification retain the V3 execution contract and never recompute a full mask.

Determinism of the frozen binary64 production operations, original action, full
causal initial state, observation and pre-action memory determine the proposed
successor and every endpoint under action -1. V3's KINEMATIC_REDUCTION_PROOF.md
establishes conditional exact position/velocity parity between its reduced
kinematics and full dynamics. Hence dropping explicit transcript storage loses
no reconstructive input. SHA-256 binds the original ordered binary endpoints,
subject to the usual cryptographic collision assumption; physical reconstruction
and exact semantic checks are required independently of that binding.

The V4 verifier never invokes FastBankersShield, CompactShield or safe_action_mask.
It independently runs unchanged full production advance for each candidate and
every braking endpoint. Its physical certificate reconstruction reuses V2's
independent lower-envelope predicate and source-level physical reconstruction,
retaining full states transiently to also reconstruct the frozen selected
certificate's full-state JSON digest. It checks endpoint, known/unknown preview,
backup convergence, finite states, margins, minimum slack, distances, terminal,
reason, final classification, action order and source/input identities, then the
binary trajectory hash and valid prefix. Invalid/out-of-domain inputs use the
original full-production BankersShield fallback, never the fast classifier.
Importing V3 safety_view only projects comparison fields; it makes no decision.

Identical masks, proposals and frozen deterministic tie-breaking imply identical
executed physical actions. Full production replay additionally checks actual
successors, native signed energy/power, memory, original lifecycle, deadline
diagnostics, EpisodeMetrics and original is_feasible with 140 s and zero excess.

The inherited conditional robust speed-invariance argument is preserved: the
envelope is a lower bound on every compatible route; accepted successors and all
backup endpoints satisfy that bound with the original velocity margin. Unknown
road beyond the pre-action frontier uses the minimum legal speed. Maximum braking
reaches that speed within the original finite bound and remains below every legal
limit thereafter. Refreshed preview cannot justify the current decision. The
original assumptions about sensing, generator spacing, frozen model, floating
runtime and causal memory remain necessary. This is a conditional source-level
argument with supporting finite tests, not a machine-checked universal theorem,
continuous-time speed guarantee, deadline proof or robustness to unmodeled error.
