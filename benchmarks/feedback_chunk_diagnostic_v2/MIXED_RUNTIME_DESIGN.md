# Mixed-runtime design

Worker C uses the Feedback-Chunk V1 environment (Python 3.11.13, NumPy 2.4.6)
and exclusively loads CHUNK3/RESIDUAL1S checkpoints. It is also the
preregistered primary runtime for CAUSAL60, TIME_AWARE, FixedPrice MPC V2, and
AdaptivePrice MPC V2.

Worker H uses the historical Hybrid environment (Python 3.13.16, NumPy 2.5.4)
and exclusively loads H11/H29/H47. Both workers must match their complete
`runtime_manifests.json` records and the frozen production/source bundle.

Workers are persistent JSON-lines processes. `PREPARE` verifies the frozen pair,
task fingerprints, runtime, source, and already-loaded checkpoint. Only then may
the master persist `STARTED`; `EXECUTE` rejects the pair unless that durable
state and ownership token are visible. Workers never mutate a scientific
manifest or ledger.
