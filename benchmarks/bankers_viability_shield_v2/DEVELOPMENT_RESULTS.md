# Development results — NOT_EXECUTED_RESOURCE_GATE_CLOSED
The resource gate failed before constructing any task. There is no ordered panel
hash, no scientific pair execution and no Development performance sample.

| Controller | Executed / planned | Raw RSR | Completion / deadline / speed | Feasible-only energy / time |
|---|---:|---|---|---|
| CAUSAL60 | 0 / 18 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| LEGACY_NATIVE_POST_GREEDY | 0 / 18 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| BANKERS_GREEDY | 0 / 18 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |
| BANKERS_MASKED_CAUSAL60 | 0 / 18 | NOT_MEASURED | NOT_MEASURED | NOT_MEASURED |

The original target raw RSR denominator is 18 per arm; zero executed tasks are
not reported as 0/18 success or as zero failures. Failure identities, deadline
failures, exact speed compliance, energy means/medians and travel times are
undefined for this unopened panel, with explicit NOT_EXECUTED states in results.json.

CAUSAL60 versus BANKERS_MASKED_CAUSAL60 matched feasibility, action differences,
trajectory changes and masked deadline-failure IDs: NOT_MEASURED. Legacy-post
versus Banker-pre matched Development certificates/outcomes: NOT_MEASURED.
The synthetic certificate comparisons are retained in fixture records and cannot
substitute for task-paired Development conclusions.

No Development task receives FEASIBLE_WITNESS, PROVEN_INFEASIBLE or UNRESOLVED
content classification because no task content was opened. Controller failure
would not establish physical impossibility. No denominator correction or
post-hoc route removal occurred. See ADVERSARIAL_FIXTURES.md for the separate
synthetic results, including speed-safe greedy deadline failures.
