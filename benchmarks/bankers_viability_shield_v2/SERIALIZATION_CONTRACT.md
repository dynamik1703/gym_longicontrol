# Canonical evidence contract bankers-evidence-v2.0
Normalize before comparison, hashing or serialization. Dataclasses become maps;
NumPy arrays and nested tuple/list sequences become lists; NumPy bool/integer
and finite binary64-or-smaller reals become native JSON scalars without loss.
Integers remain integers; floats remain floats, with Python's round-trip binary64
representation, including signed zero. Mapping keys must be strings. Sort keys
recursively; ASCII escaped JSON, compact separators, no NaN/Infinity, one LF per
record. Reject unsupported objects, extended precision and duplicate JSON keys.
This is a versioned runtime-bound contract, not an RFC 8785 interoperability claim.

A gzip file has a canonical header followed by canonical native-step JSON lines.
The header binds contract, V2 source/config identity, unique Attempt ID and count.
Persist compressed and uncompressed SHA-256, both byte counts and record count.
The gzip timestamp and filename are fixed. Write a unique temporary file in the
same directory, flush and fsync; publish with exclusive hard-link creation,
remove the temporary link, then fsync the directory. Existing evidence cannot
be overwritten. Mutable ledgers use atomic replace and directory fsync.
A publication failure before linking leaves no final evidence; a crash after
publication still leaves STARTED ambiguous and forbids restarting that pair.

Read from disk, verify hashes/counts/schema and exact canonical bytes, then run
independent physical replay. Recomputed valid hashes never excuse semantic
corruption. Missing safety fields, undefined deadline applicability, wrong counts,
duplicates, changed numeric fields and stale identities fail closed.

The V1 certificate's own source/configuration identity and hash algorithm are
preserved inside each record. The V2 outer identity binds its adapter, controllers,
serializer, replay, tests, fixed protocol and the entire inherited frozen tree.
All 21 full certificates are persisted, including successor, causal input digest,
backup policy identity/path hash, slack, distances, margins and rejection flags.
The selected action is freshly recertified immediately before execution.
Offline replay rebuilds every certificate from production advance without calling
the admission function, verifies actual observation/state/energy/limit/lifecycle,
checks the envelope against physical infrastructure, and replays policy selection.
