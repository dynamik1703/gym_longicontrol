# V2 Evaluation Orchestration

The V2 runner is intentionally unusable under the committed preparation
contract. Authorization is checked before split verification, policy loading,
environment construction, or reset.

A later, explicit authorization amendment must remove the preparation lock and
set both V2 authorization booleans. That amendment must be committed before
the first performance episode. It may not modify the split, pair order,
controllers, checkpoints, metrics, or thresholds.

For each future pair, the runner performs this order:

1. validate the separate V2 authorization contract;
2. verify deterministic split regeneration and the complete pair ledger;
3. load and hash all three frozen checkpoints read-only before any reset;
4. select the next `PENDING` identity;
5. atomically persist and fsync `STARTED`;
6. evaluate exactly that identity using the single-arm evaluator;
7. atomically persist `COMPLETED` and its result hash, or persist
   `INTERRUPTED_UNRECOVERABLE` and stop.

The old V1 three-arm convenience function is never called. Tests use synthetic
ledgers, callbacks, and failure injection only. The real V2 authorization file
is never changed by the suite.
