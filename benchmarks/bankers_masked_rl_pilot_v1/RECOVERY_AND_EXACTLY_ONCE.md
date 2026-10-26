# Recovery status and exactly-once requirements

The checkpoint component atomically serializes the complete runner, environment,
causal memory, current mask/certificates, task cursor, partial episodes, replay,
agent/targets/optimizers/temperature/multipliers, explicit policy/replay RNGs,
global RNG states and native/update counters. Source/config/runtime and Attempt
identity are checked on load. An exclusive ownership primitive is provided.

In each of the three failed treatment tests, checkpoint save/load and equality
assertions for continued synthetic rows, network parameters, temperature,
multipliers and replay arrays were reached successfully before the independent
audit raised TypeError. This is partial component evidence only: the combined
test failed. It does not establish exact scientific crash recovery.

A durable STARTED-before-reset transaction, journal/checkpoint reconciliation,
uncertain-action crash handling and immutable scientific final checkpoint sealing
were not completed. Therefore gate L is not accepted as a complete pipeline pass.
No scientific Attempt exists to resume or restart. A future repair must retain
this negative record and obtain a new explicit protocol decision rather than
silently converting it to a successful Attempt.
