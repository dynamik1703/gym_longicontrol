# Constrained V2 collector mechanism audit

This directory is a deterministic, read-only mechanistic audit of the completed
three-seed collector replication at
`38a8c7cc6789ae22d44de2b6f6cd3c84352c4b62`. It performs no training, RL update,
environment rollout, Development/Validation/Paper evaluation, or policy forward
pass. Loading retained checkpoint tensors is the only neural-network operation.

The frozen `SEED_HETEROGENEOUS` classification and `DO_NOT_FREEZE_COLLECTOR`
decision are unchanged.

## Reproduction

From this audit worktree, using the study's pinned environment:

```bash
/Users/romanliessner/gitlab/gym_longicontrol_multistep_mbrl/.venv/bin/python \
  -m benchmarks.constrained_v2_collector_mechanism_audit.analyze \
  --source-root /Users/romanliessner/gitlab/gym_longicontrol_collector_replication
```

Verify byte-deterministic regeneration without writing:

```bash
/Users/romanliessner/gitlab/gym_longicontrol_multistep_mbrl/.venv/bin/python \
  -m benchmarks.constrained_v2_collector_mechanism_audit.analyze --verify \
  --source-root /Users/romanliessner/gitlab/gym_longicontrol_collector_replication
```

`results.json` contains every checkpoint timeline, every retained episode event,
all paired E−T curves, input hashes, the missing-field ledger, and scorecards.
`MECHANISMS.md` is the synthesis; `SEED11.md`, `SEED29.md`, and `SEED47.md`
provide pair-specific evidence. `FINALIZER_AUDIT.md` covers recovery integrity.
