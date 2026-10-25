# Source and checkpoint audit

The study is rooted at commit
`39e7b3c5edb435af2540a912633a966222bf96be` on
`research/feedback-chunk-diagnostic-v2`. Only the remote named
`gym_longicontrol_origin`, whose URL is
`https://github.com/dynamik1703/gym_longicontrol.git`, may receive this branch.

Six Feedback-Chunk V1 endpoints and three historical Hybrid endpoints are
loaded read-only by their strict original loaders. SHA-256, training Attempt ID,
source provenance, 300,000 native transitions, 30,000 gradient updates, and
deterministic actor probes are mandatory preflight checks. No loader bypass,
metadata edit, weight migration, retraining, or checkpoint selection is
permitted.

The complete machine-readable checkpoint records are frozen in
`execution_manifest.json`. Diagnostic V2 itself authorizes zero training
transitions and zero gradient updates.

Final checkpoint SHA-256 values:

- H11 `a16cd663a4cd046532490b74a414e37acee9b37ca0c0b886753fa8d767f4f7ed`
- H29 `b4c6f5f484cc4a68b044b39f70db8d5f8245098729b3bb06749dbb46bdb9af55`
- H47 `3a67097e07360a07c8d5e5a276cf388434476943941085aea98b4a85254d82cf`
- CHUNK3 S11 `de59e6435e93f81b08931f414b8615226c424e025c44090e31d107d2e61770e0`
- CHUNK3 S29 `7b71c9c2206e0e3a5e96486812a8a85925b4ceef73bc7bdfce7a00482410939d`
- CHUNK3 S47 `207a5cc0feae49bab0a2027fcf1ab8432a073c7776a468f7d1af5f2d73ed813f`
- RESIDUAL1S S11 `90a832a9839a6a7f22298c91e45d61d8e2d3f47a508dcd6f4a1475cd9a056591`
- RESIDUAL1S S29 `46e5aa9ce633a88dbca012db352f1d923daf68de7ced29c2c0a2aa1d8167bead`
- RESIDUAL1S S47 `282544146f0ece5a6b37b9032907149fe808100d5955bb933129538ecd363424`

All nine passed strict native load and repeated deterministic inference.
