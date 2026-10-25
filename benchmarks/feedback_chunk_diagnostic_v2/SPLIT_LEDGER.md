# Diagnostic V2 split ledger

`split_ledger.json` is the authoritative performance-blind scan and rejection
ledger. It freezes ordered numeric identities, canonical task serializations,
route hashes, full-task hashes, historical protection sources, and an ordered
panel hash. Runtime fingerprints must reproduce exactly under both native
workers before authorization.

The scan considered seeds 11,400,000 through 11,400,039. It accepted 36 tasks
and rejected 11,400,005, 11,400,011, 11,400,029, and 11,400,033 because both
their route and full-task fingerprints were already reserved. The ordered panel
hash is
`d39fa351907c570fb47c3ba19805a434d480c32d5947a5b801364fc7a8175d6a`.
Both native runtimes reproduced every accepted route/full-task fingerprint
exactly.
