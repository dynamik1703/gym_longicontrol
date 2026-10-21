# Hybrid RL V1 Validation Integrity Stop

State: **INTEGRITY_STOPPED_BEFORE_REGISTERED_TRANSACTION**

On 21 October 2026, during pre-execution testing, the historical fail-closed
test called the old **evaluate_validation(None)** convenience function after the
Validation authorization flag had temporarily been enabled.

That convenience function evaluates its three arms sequentially. On the first
frozen Validation task (numeric seed 10100000):

1. **CAUSAL60** completed one episode;
2. **TIME_AWARE** completed one episode;
3. the generic Hybrid arm reset the environment and then raised
   **Hybrid RL evaluation requires an agent** before its first action.

The two completed baseline results were held only inside the failed function
call and were not returned or persisted. No registered pair ledger existed yet.
No learned actor was evaluated, no training or gradient update occurred, and
Paper remained untouched.

The authorized protocol states that a completed controller–task pair must never
be evaluated again and that an unrecorded partial execution must stop for
investigation. Reconstructing the lost results would require repeating the
CAUSAL60 and TIME_AWARE pairs for seed 10100000. The remaining 180-pair
Validation therefore cannot be represented as an exactly-once transaction
without a new explicit protocol decision.

Containment applied:

- Validation authorization returned to false;
- Main Training remains disabled and unauthorized;
- Paper remains sealed with zero episodes and zero fingerprints;
- no further Validation pair was executed;
- no Validation ledger, results aggregation, completion commit, or push was
  performed.

Machine-readable evidence is retained in
**validation_integrity_incident.json**. Further execution requires explicit
user and scientific-protocol direction.
