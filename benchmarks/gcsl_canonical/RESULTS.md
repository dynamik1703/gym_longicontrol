# Canonical GCSL Validation results

## Outcome

The final held-out `GCSL_VALIDATION_V1` result is **GCSL_ZERO**: all three frozen final GCSL actors scored 0/18, for pooled RSR 0/54. The preregistered thresholds were applied unchanged. Despite self-generated hindsight supervision and observed real canonical training successes, GCSL did not transfer reliably to the held-out canonical task. The limiting factor was not simply absence of successful hindsight labels. This conclusion is specific to the tested projected-goal, self-generated protocol.

| Method | RSR | Completion | Deadline | Speed |
| --- | --- | --- | --- | --- |
| G11 | 0/18 (0.0%) | 18/18 (100.0%) | 18/18 (100.0%) | 0/18 (0.0%) |
| G29 | 0/18 (0.0%) | 18/18 (100.0%) | 18/18 (100.0%) | 0/18 (0.0%) |
| G47 | 0/18 (0.0%) | 16/18 (88.9%) | 16/18 (88.9%) | 1/18 (5.6%) |
| C11 | 10/18 (55.6%) | 17/18 (94.4%) | 11/18 (61.1%) | 15/18 (83.3%) |
| C29 | 6/18 (33.3%) | 11/18 (61.1%) | 6/18 (33.3%) | 15/18 (83.3%) |
| C47 | 6/18 (33.3%) | 10/18 (55.6%) | 6/18 (33.3%) | 16/18 (88.9%) |
| GCSL pooled | 0/54 (0.0%) | 52/54 (96.3%) | 52/54 (96.3%) | 1/54 (1.9%) |
| Constrained pooled | 22/54 (40.7%) | 38/54 (70.4%) | 23/54 (42.6%) | 46/54 (85.2%) |

The Constrained rows are a **contextual comparison — not fully matched algorithmically**. GCSL uses a 12D actor state, direct hindsight action likelihood, and 7,250 optimizer steps; Constrained uses an 8D actor state, reward/cost critics plus adaptive constrained optimization, and 30,000 optimizer steps. The table is not a strict algorithm-only causal comparison, and Constrained performance does not define GCSL success.

## GCSL classification and seed behavior

`GCSL_ZERO` is selected because pooled RSR is exactly 0/54. The alternatives remained frozen: `GCSL_ROBUST` required at least 36/54 pooled and at least 8/18 for every seed; `GCSL_PARTIAL` required at least 6/54; `GCSL_WEAK` required 1–5/54.

There is no RSR heterogeneity: G11, G29, and G47 are all 0/18. Secondary behavior does vary. G11 and G29 completed all 18 tracks within the deadline but violated the exact speed requirement on all 18. G47 completed and met the deadline on 16/18, and met the speed requirement on only 1/18. Thus GCSL learned task-completion behavior but failed to convert hindsight safety information into robust canonical speed compliance. Availability of successful hindsight labels was insufficient for reliable closed-loop canonical transfer.

| Policy | Mean final progress | Exclusive failure modes |
| --- | --- | --- |
| G11 | 1.0000 | `{"speed": 18}` |
| G29 | 1.0000 | `{"speed": 18}` |
| G47 | 0.9306 | `{"incomplete+deadline": 1, "incomplete+deadline+speed": 1, "speed": 16}` |
| C11 | 0.9961 | `{"deadline": 4, "deadline+speed": 2, "feasible": 10, "incomplete+deadline": 1, "speed": 1}` |
| C29 | 0.8250 | `{"deadline": 3, "deadline+speed": 2, "feasible": 6, "incomplete+deadline": 6, "incomplete+deadline+speed": 1}` |
| C47 | 0.9348 | `{"deadline": 3, "deadline+speed": 1, "feasible": 6, "incomplete+deadline": 7, "incomplete+deadline+speed": 1}` |

Pooled GCSL exclusive failure modes are `{"incomplete+deadline": 1, "incomplete+deadline+speed": 1, "speed": 52}`. Pooled Constrained exclusive failure modes are `{"deadline": 10, "deadline+speed": 5, "feasible": 22, "incomplete+deadline": 14, "incomplete+deadline+speed": 2, "speed": 1}`.

## Feasible-only energy

Energy is interpreted only for feasible episodes.

| Method | n | Mean | Median | Min | Max |
| --- | --- | --- | --- | --- | --- |
| GCSL | 0 | n/a | n/a | n/a | n/a |
| Constrained | 22 | 0.189386 | 0.177438 | 0.164647 | 0.274946 |

No GCSL feasible-energy comparison is defined. The 22 feasible Constrained episodes have mean signed/net energy 0.189386 kWh, median 0.177438 kWh, and range 0.164647–0.274946 kWh.

## Diagnostic-to-Validation consistency

| Policy | Diagnostic RSR | Validation RSR | Diagnostic C/D/S | Validation C/D/S | Mean progress D → V |
| --- | --- | --- | --- | --- | --- |
| G11 | 1/18 | 0/18 | 18/18/1 | 18/18/0 | 1.0000 → 1.0000 |
| G29 | 0/18 | 0/18 | 18/18/0 | 18/18/0 | 1.0000 → 1.0000 |
| G47 | 0/18 | 0/18 | 16/16/1 | 16/16/1 | 0.9412 → 0.9306 |

G29's failure regime reproduced exactly: full completion and deadline compliance with zero speed compliance. G47 reproduced the same RSR, completion, deadline, speed, and exclusive failure-mode counts, with similar mean progress. G11 lost its single Diagnostic feasible episode while retaining full completion/deadline performance; its earlier `SUCCESSFUL_BOOTSTRAP` description remains accurate for the observed training/Diagnostic event but is not supported as held-out robust transfer. Overall, the Diagnostic warning generalized and the held-out result is weaker by one feasible episode.

## Canonical support and supervised fit

| Policy | First success | Real successes | Canonical tuples | Validation | Interpretation |
| --- | --- | --- | --- | --- | --- |
| G11 | 147,496 | 4 | 69 | 0/18 | CANONICAL_SUPPORT_NO_TRANSFER |
| G29 | 10,614 | 13 | 476 | 0/18 | CANONICAL_SUPPORT_NO_TRANSFER |
| G47 | 29,655 | 5 | 158 | 0/18 | CANONICAL_SUPPORT_NO_TRANSFER |

Every seed had genuine canonical support but no held-out feasible episode, so the complete evidence supports `CANONICAL_SUPPORT_NO_TRANSFER` for G11, G29, and G47. The coarse projected goal `[progress, timely, safe]` collapses physically different trajectories. Projected-goal aliasing and self-imitation ambiguity are plausible mechanisms, but the retained diagnostics do not establish either as the cause of the held-out failures.

| Policy | Final transformed-action NLL | Validation RSR |
| --- | --- | --- |
| G11 | -4.559745 | 0/18 |
| G29 | -0.433832 | 0/18 |
| G47 | -0.286829 | 0/18 |

Low supervised NLL did not imply successful closed-loop control. G11's much lower final NLL did not yield a held-out feasible episode, and G29/G47 likewise scored zero. This is descriptive; it does not claim that NLL caused performance.

## Provenance and execution integrity

- Pre-Validation commit: `08c1d63f925c307168935c086d40dfc5f31ab3a9`.
- Scientific configuration SHA-256: `c8807fd21cc2335b87bc9d1391970f89b7a2d7af7d385bb1253888185dcfedad`.
- Frozen ordered Validation SHA-256: `d808fcc361be35f0c585003278f200d4013cc03718b0ceb318fbe6d55195a159`.
- Exactly 54 unique GCSL pairs plus 54 unique Constrained pairs were evaluated: 108 total, with every expected pair exactly once and no duplicate.
- Only final 300k actors G11/G29/G47 and immutable C11/C29/C47 references were used; no earlier checkpoint or best-checkpoint selection was used.
- All checkpoint, actor, learner/policy-state, optimizer, replay, task-stream, counter, and evaluation RNG checks remained unchanged where applicable. No training, adaptation, hindsight/replay update, critic update, PID/Lagrange update, planning, or model use occurred.
- Validation open count is 1. The first process completed the 54 GCSL pairs and then stopped before C11 because that interpreter lacked FSRL. Those artifacts were preserved; a pinned-dependency continuation evaluated only the 54 untouched Constrained pairs. Zero pairs were repeated and Validation was not reopened.
- Complete compact metrics for all 108 episodes and their immutable artifact hashes are in `results.json`.
- Paper is `SEALED_UNINSPECTED`: 0 episodes and 0 constructed fingerprints.
