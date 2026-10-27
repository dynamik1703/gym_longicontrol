# V3 prospective reuse

This V3 contract preserves the repaired V2 scientific method. All execution identities are new; V2 remains immutable.

# Exactly-once training transaction and recovery

An exclusive owner and immutable STARTED exist before the first reset. Initial
technical checkpoint captures the complete pre-reset runner. Every chosen native
action has a durable immutable write-ahead intent containing task/cursor, native
ID, current state/mask binding, proposal, actor-update version and post-sampling
policy RNG digest, bound to the preceding journal head. Actual action, replay
insertion, scheduled updates and independent transition audit precede row commit.
Hash-chained JSON rows are fsynced, then the intent is removed. Every 1000 rows,
an immutable deterministic-gzip segment is published and a technical checkpoint
atomically replaces its predecessor. Rotation duplicates must be byte-identical.

Checkpoints preserve actor/reward/cost/target networks, all optimizers, temperature,
multipliers, full replay and cursor, environment and RNG, causal memory, masks and
bindings, task cursor, partial episode metrics, global and explicit RNGs, counters,
auditor-owned state and journal-prefix binding. Checkpoint representation is a
gzip-1 compressed full pickle with an embedded hash of the uncompressed payload.
This storage adaptation changes no learner state. CPU threads=1 and deterministic
Torch execution are restored explicitly in a new process.

Same-Attempt recovery verifies source/runtime/Attempt identity, checkpoint digest
and journal prefix. Completed rows after the checkpoint are deterministically
reconstructed offline with exact proposal/RNG/row checks and the same optimizer
updates. This is recovery reconstruction of existing logical events, not new
scientific transitions or a fresh training Attempt. A pending intent is completed
only when the reconstructed prepared action matches all persisted intent fields.
A committed-but-not-cleared intent is recognized without duplicate insertion.
Exactly one logical native/replay ID is retained. Completed Attempts cannot restart.
Torn, conflicting or unavailable complete state fails closed and preserves evidence.

Technical retention=one until the run is independently completed; after immutable
final-50000 checkpoint and COMPLETED seal, the technical copy may be removed.
Scientific final checkpoints remain immutable. Diagnostic partial pairs have no
automatic recovery: STARTED without COMPLETED stops rather than rerunning a pair.

Engineering tests include mid-episode, post-action, pre/post replay, pre/post update,
near checkpoint publication, task termination/reset and real os._exit process crash.
All optimizer moments, RNGs, networks, replay arrays and native row identities must
match uninterrupted execution. Full native replay is tested after recovery in all arms.
