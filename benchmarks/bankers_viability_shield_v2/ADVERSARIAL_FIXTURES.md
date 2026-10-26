# V2 synthetic/adversarial coverage — complete within declared scope
Every listed scenario passed under new V2 identities. Unit IDs below prefix the
new V2 test node names; these are not original V1 scientific pair identities.

| Required case | V2 evidence |
|---|---|
| constant 20 / constant 100 | V2_FIXTURE__constant_20 / constant_100 |
| rest / initial certification | V2_UNIT__test_v2_initial_certified; full fixture resets |
| maximum legal speed | V2_UNIT__test_v2_rounding_and_maximum_legal_speed |
| visible 60→20 | V2_FIXTURE__60_to_20 |
| maximum consecutive reductions | V2_FIXTURE__consecutive_40_reductions |
| multiple visible transitions | V2_FIXTURE__two_preview_signs; V2_UNIT__test_v2_multiple_signs_and_retention |
| exact 150 m entry | V2_FIXTURE__exact_150_preview; V2_UNIT__test_v2_exact_preview_boundary |
| within-step sign crossing | V2_UNIT__test_v2_within_step_transition_and_unsafe_acceleration |
| near 1000 m | V2_FIXTURE__final_near_1000 |
| high-speed frontier braking | V2_RESOURCE probes 20/25/25.5 m/s; V2 negative control |
| repeated maximal admitted action | all eight V2_FIXTURE full episodes |
| repeated minimal admitted action | V2_FIXTURE__repeated_minimal (120-step brake prefix) |
| floating-point boundaries | V2_UNIT__test_v2_rounding_and_maximum_legal_speed |
| retained lower limits | V2_UNIT__test_v2_multiple_signs_and_retention |
| attempted unsafe acceleration | V2_UNIT__test_v2_within_step_transition_and_unsafe_acceleration |
| empty-mask fail closed | V2_UNIT__test_v2_empty_mask_and_stale_observation_fail_closed |
| exact backup after admitted action | V2_UNIT__test_v2_full_backup_after_admitted_action; 6,425 search witnesses |
| native zero/coasting | V2_UNIT__test_v2_zero_is_coasting |

| V2 fixture | Steps | Completed | Time s | Exact speed excess | Feasible-only energy kWh |
|---|---:|---|---:|---:|---:|
| V2_FIXTURE__60_to_20 | 1487 | True | 148.7 | 0 | — |
| V2_FIXTURE__consecutive_40_reductions | 1185 | True | 118.5 | 0 | 0.209254311 |
| V2_FIXTURE__constant_100 | 435 | True | 43.5 | 0 | 0.299339781 |
| V2_FIXTURE__constant_20 | 1800 | False | 180.0 | 0 | — |
| V2_FIXTURE__exact_150_preview | 1668 | True | 166.8 | 0 | — |
| V2_FIXTURE__final_near_1000 | 645 | True | 64.5 | 0 | 0.158969461 |
| V2_FIXTURE__repeated_adversarial | 1344 | True | 134.4 | 0 | 0.295844239 |
| V2_FIXTURE__repeated_minimal | 120 | False | 12.0 | 0 | — |
| V2_FIXTURE__two_preview_signs | 1427 | True | 142.7 | 0 | — |

The minimal-action test is an intentional 120-step braking prefix, not a claimed
completed original task. Seven resource probes add 35 physical records. All
10,146 retained native steps and 213,066 masks/candidates passed physical
reconstruction; maximum endpoint speed excess is exactly 0 m/s. A state at a
legal maximum need not be in the conservative robust inner set; rejection there
is recorded, not misreported as backup preservation from an uncertified state.

The deterministic adversarial search tested 320 states and 6,720 proposals,
including 6,425 admitted exact braking paths and successor backup checks, finding
zero core counterexamples. Seed and counts are retained in counterexample_search.json.
The separate naive refreshed-preview negative control reproduces excess
0.08740041837874557 m/s at x=150.02593711687808 m; the intended certificate rejects
its proposal. It is not a core-invariant falsification. No tolerance or margin
was changed in response to these results.
