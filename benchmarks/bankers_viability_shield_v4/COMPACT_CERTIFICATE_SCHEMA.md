# bankers-evidence-v4.0 (frozen before preflight)

Canonical JSON retains V2's finite binary64 round trip, sorted ASCII keys,
duplicate-key rejection, explicit nulls, typed booleans and required fields.
Gzip mtime is zero. Immutable publication uses fsynced temporary files, exclusive
hard-link publication and directory fsync. Each native journal row uses this same
compact format and is flushed and fsynced before proceeding. No candidate path is
written into a secondary log. Temporary V3 transcripts exist only inside the
unchanged online reference call and are discarded immediately after projection.

Every row includes its Attempt/source identity, sequential step, exact native
simulation timestamp, controller, complete pre-state, ordinary eight-vector,
persistent causal envelope and remembered signs, CAUSAL60 memory, proposal,
selected index/action, intervention information, full predicted and actual native
successors, actual power and signed energy increment, physical endpoint speed
limit/excess, completion/termination/truncation, explicit deadline diagnostic or
NOT_APPLICABLE, sampled inherited predicates, authority and measured timing.

Banker rows contain exactly 21 ordered candidate certificates. Non-Banker rows
have an empty grid and explicit null selected certificate, proposal certificate
and authority, following the unchanged V3 method-specific execution contract.
Each candidate retains every V3 safety_view field: action, admission, endpoint,
known-transition and unknown-preview predicates, backup identity/existence/length,
minimum slack, recovery/preview/active-constraint distances, exact margins,
rejection reason, classification, input hash and prediction count. Predicted and
terminal hypothetical states retain exact position and velocity. Additional V4
fields are schema, action_index, source_config_identity, reference_kernel,
backup_record_status and backup_valid_prefix_steps. The continuous CAUSAL60
proposal diagnostic has action_index=null when it lies off grid.

backup_trajectory_sha256 hashes the concatenation, in native step order, of
IEEE-754 binary64 network-order position then velocity, struct.pack("!dd",x,v),
16 bytes per maximum-braking endpoint AFTER the proposed action. The proposed
successor is retained separately and is not part of this digest. Empty paths
hash the zero-length byte string. This is exactly V3's decoded transcript digest;
the base64 transcript is absent from every durable V4 row.

VALID and REJECTED describe completely computed backups; their valid-prefix count
is the number of consecutive initial backup endpoints satisfying the unchanged
envelope predicate before the first violation (zero is meaningful).
NOT_APPLICABLE_ENDPOINT_REJECTED and ABSENT_FAIL_CLOSED retain null valid-prefix
counts and terminal states. The original backup_steps and rejection reason are
always preserved; missing records cannot silently become inapplicable records.
No safety field is defaulted when absent. Invalid input retains V3's frozen
fallback semantics and exact fail-closed reason.

The selected Banker action additionally retains the complete frozen production
certificate, including all native physical fields and its original full-state
trajectory digest. Actual executed native state/energy records are never reduced.
V4 is a new representation with equivalent reconstructible safety meaning; it is
not a byte-for-byte copy of V3 evidence.
