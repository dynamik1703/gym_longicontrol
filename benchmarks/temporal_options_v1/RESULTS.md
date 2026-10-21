# Temporal Options V1 — Final Results

| Arm | RSR | Complete | Deadline | Speed | Energy feasible kWh | Time s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COMMIT_FEEDBACK | 18/18 | 18 | 18 | 18 | 0.176739850 | 97.244 |
| CONTINUE_FEEDBACK | 18/18 | 18 | 18 | 18 | 0.172722352 | 97.200 |
| FROZEN_MPC_V2 | 18/18 | 18 | 18 | 18 | 0.171096061 | 97.228 |
| RESET_FEEDBACK | 18/18 | 18 | 18 | 18 | 0.171765449 | 98.550 |

## Answers to the registered questions

1. Systematic phase postponement was demonstrated in deterministic fixtures for
   inherited distance-triggered 30%, 40%, 50% and 82% phases under repeated
   sub-threshold one-second re-anchoring. It was not universal: velocity-triggered
   pulse coasting remained reachable.
2. Affected fixtures were traction-then-coast, traction-coast-brake and
   coast-then-partial-traction. Speed clipping, safety intervention and short final
   horizons qualify the result.
3. RESET versus frozen MPC mean paired feasible-only saving was
   -0.40490711539569835%; this is the native-feedback/scoring
   contrast.
4. CONTINUE executed coast/brake/partial phases with counts
   `{'brake': 117, 'coast': 1809, 'fallback': 1153, 'partial_traction': 134, 'pulse_traction': 372, 'regulated_feedback': 10347, 'traction': 3564}` and retained anchors across boundaries.
   RESET planned 3330 brake and 1983 partial-traction native steps but executed
   neither phase; CONTINUE planned 3426 brake and 2865 partial-traction steps and
   executed 117 and 134 respectively. RESET recorded 1586 strategic resets below
   30% progress versus 769 for CONTINUE; COMMIT recorded none.
5. COMMIT versus CONTINUE mean paired saving was
   -2.3780106472573683%; COMMIT RSR was 18 versus
   CONTINUE 18.
6. Feasibility/completion/deadline/speed counts for all arms appear above.
7. CONTINUE versus RESET mean paired energy saving was
   -0.5620832885179694% and mean travel-time delta was
   -1.3499999999999224 s. CONTINUE versus frozen MPC saving was
   -0.9624538816570833%.
8. CONTINUE made 985 starts, 771
   continuations and 811 switches; mean option duration was
   1.6458883248730736 s and distance 16.79734161493474 m.
9. CONTINUE safety interruptions numbered 131; reasons
   are `{'DEADLINE_RECOVERABILITY': 116, 'KNOWN_TRANSITION_BRAKING': 11, 'RIGHT_ENDPOINT_SPEED': 4}`.
10. Every observed gain/loss is architectural and non-learning. Gradient updates
    were zero.
11. CONTINUE retained 15 distinct episode
    action trajectories and selected 15 distinct high-level actions (CONTINUE
    plus 14 of 17 START profiles). All 18 actions were still scored where valid
    and remain memory-distinct even when first native actions match.
12. The final interface is 70D with 18 actions and a valid-action mask; transition
    records use actual duration and `gamma_native ** k`.
13. Focused exact-recovery tests passed and all pair records are hash-bound and
    exactly-once. Post-run integrity audit found that per-step completion bounds
    were not retained, so deadline-bound falsifications cannot be independently
    recomputed; see `INTEGRITY_INCIDENT.md`.
14. Runtime, simulator steps, planner calls and physics-model calls are in
    `RESOURCE_REPORT.md` and machine-readable episode records.
15. Fresh panel hashes are recorded in `split_ledger.json`.
16. Classification: `SEMANTIC_OR_INTEGRITY_STOP` because of the retention incident.
    Before that correction the automatic performance label was
    `OPTIONS_VALID_EFFECT_UNCLEAR`: CONTINUE lost no feasibility but missed the
    1% saving gate. Other uncertainty includes heuristic candidate values,
    empirical Shield coverage and only 18 distinct Capability routes.
17. Commit and remote synchronization are reported at handoff after the final
    immutable results commit.
18. RL training episodes and gradient updates: zero.
19. Old Validation episodes: zero.
20. Paper episodes/fingerprints: zero; Paper stayed sealed and uninspected.

## Recommendation

Do not proceed until the semantic or integrity failure is resolved.
