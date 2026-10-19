# Constrained V2 seed-11 track × collector factorial V3

V3 is the authorized repair of the stopped V2 study. It preserves the frozen
Constrained V2 learner and 2×2 track-stream/collector treatments while replacing
the invalid old diagnostic split with a deterministic, fully route-clean namespace.

The reachable training horizon is derived as `floor(300000 / 347) + 1 = 865`
tasks. The extra task covers an episode that may be active when the lifetime
transition budget ends. Both exact 865-task streams were reconstructed and match
their expected ordered fingerprints.

Starting at seed `9100000`, candidates are scanned in ascending numeric order and
the first 18 passing every frozen numeric, training-content, opened-split-content,
and internal-uniqueness rule are accepted. The full evidence is in
`split_ledger.json`; `SPLIT_LEDGER.md` is its compact human-readable view.

All static, initialization, collector, recovery, evaluation non-interference,
runtime, checkpoint, and disk gates passed. The serial A/B/C/D matrix is therefore
authorized. Training output is written beneath
`runs/constrained-v2-seed11-factorial-v3/`; `finalize.py` validates a completed
matrix and materializes `results.json` and `RESULTS.md` without further policy
evaluation.

Paper seeds `4000–4017` remain sealed and were never instantiated, fingerprinted,
hashed, inspected, or evaluated. Legacy Development and Validation were statically
fingerprinted solely for split separation and must not receive policy evaluation.
