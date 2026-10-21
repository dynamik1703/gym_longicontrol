# Hybrid RL V1 Main-Training Report

## Outcome

`HYBRID_RL_MAIN_V1` Revision 3 completed successfully for H11, H29, and H47. Each policy reached 300,000 native transitions and 30,000 SAC updates. The final 300k policies are frozen; no best-checkpoint selection, early stopping, seed replacement, or performance-based hyperparameter change occurred.

Only the authorized 18-task `HYBRID_RL_DIAGNOSTIC_V1` ran at the six registered checkpoints. Final Validation remained unopened and Paper remained sealed.

## Authorization and frozen provenance

- Verified Revision-3 preparation commit: `e652f547e77753ef7dd5eb0a4b9550f32975bdc2`.
- Frozen pre-authorization configuration SHA-256: `54ac0314e49d125b984fa4d95ac4ae75d87bacba3ad517a425d4c4ef43c02127`.
- Main-training authorization commit: `94beb9a140199ba0baaee214f6ab1c273f2f6a19`.
- Execution configuration SHA-256: `73e2bd6d960504eff120a37aaba60f9c3e17cb353408a67f7feddb97bf33253e`.
- Diagnostic ordered SHA-256: `c8fe41d3ae100d108a6c5175ee180e023cbd431ab3a2e709d815fef66547b784`.
- Authorization changed execution metadata only. SMDP semantics, learner parameters, physical fallback, shield, streams, and split definitions remained frozen.
- `validation_authorized=false`; `paper_authorized=false`.

## Completion accounting

| Seed | Attempt ID | Resumes | Native | Macro | Updates | Replay | Cursor |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H11 | `5f5801d9-d98c-4ace-9b1a-34476759f60a` | 0 | 300,000 | 30,108 | 30,000 | 30,108 | 241 |
| H29 | `97a40dfa-c8ed-4334-b2c3-0351e71697a8` | 0 | 300,000 | 30,097 | 30,000 | 30,097 | 240 |
| H47 | `4553b298-5797-493f-88fe-075de3786dff` | 0 | 300,000 | 30,110 | 30,000 | 30,110 | 238 |
| **Total** | — | **0** | **900,000** | **90,315** | **90,000** | **90,315** | — |

All 18 scientific checkpoint/Diagnostic combinations and exactly 324 Diagnostic episodes are present.

## Exact Revision-3 update accounting

```python
0 if N <= 10000 else min(30000, ((N - 10000) * 30000) // 290000)
```

| Native | Required | H11 | H29 | H47 |
| ---: | ---: | ---: | ---: | ---: |
| 10,000 | 0 | 0 | 0 | 0 |
| 10,010 | 1 | 1 | 1 | 1 |
| 50,000 | 4,137 | 4,137 | 4,137 | 4,137 |
| 100,000 | 9,310 | 9,310 | 9,310 | 9,310 |
| 150,000 | 14,482 | 14,482 | 14,482 | 14,482 |
| 200,000 | 19,655 | 19,655 | 19,655 | 19,655 |
| 250,000 | 24,827 | 24,827 | 24,827 | 24,827 |
| 300,000 | 30,000 | 30,000 | 30,000 | 30,000 |

Every seed recorded update 1 at native transition 10,010 and update 30,000 at transition 300,000. No 1,000-update warm-up burst occurred; every registered count matches the cumulative target.

## Diagnostic and training curves

Diagnostic and real training behavior are kept separate below. Energy is the feasible-episode arithmetic mean in kWh. C/D/S means completion/deadline/right-endpoint-speed successes out of 18; D/S/I means cumulative training deadline/speed/incomplete failures.

### H11: held-out Diagnostic

| Native | Macro | Updates | RSR | C/D/S | Energy kWh | Time s | Target speed min/mean/max m/s | Override | Influence |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 50k | 5,017 | 4,137 | 18/18 | 18/18/18 | 0.180829 | 125.389 | 5.714 / 9.506 / 16.564 | 11.861% | 52.889% |
| 100k | 10,033 | 9,310 | 18/18 | 18/18/18 | 0.180553 | 127.044 | 5.616 / 9.196 / 16.640 | 11.982% | 55.339% |
| 150k | 15,053 | 14,482 | 18/18 | 18/18/18 | 0.178810 | 126.850 | 5.666 / 8.880 / 16.636 | 11.273% | 54.434% |
| 200k | 20,069 | 19,655 | 18/18 | 18/18/18 | 0.179000 | 126.228 | 5.700 / 8.794 / 16.492 | 11.676% | 53.866% |
| 250k | 25,090 | 24,827 | 18/18 | 18/18/18 | 0.178122 | 126.472 | 5.674 / 8.932 / 16.346 | 11.096% | 53.622% |
| 300k | 30,108 | 30,000 | 18/18 | 18/18/18 | 0.177847 | 126.772 | 5.659 / 8.958 / 16.585 | 11.065% | 53.889% |

### H11: learner and real training experience

| Native | Alpha | Speed multiplier | Terminal multiplier | Reward loss | Cost loss | Actor loss | Episodes | Feasible | D/S/I | Energy kWh |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50k | 0.350354 | 4.129758 | 3.958078 | 13.803730 | 28.940981 | 31.180515 | 42 | 41 | 1/0/0 | 0.237351 |
| 100k | 0.189720 | 8.910696 | 8.723926 | 5.026230 | 20.333334 | 52.763531 | 82 | 78 | 4/0/0 | 0.218132 |
| 150k | 0.195014 | 13.526958 | 13.408808 | 4.410473 | 13.687967 | 62.862770 | 121 | 116 | 5/0/0 | 0.209089 |
| 200k | 0.197935 | 17.889212 | 17.997356 | 8.337448 | 26.928143 | 64.780869 | 160 | 153 | 7/0/0 | 0.204352 |
| 250k | 0.198150 | 21.622314 | 22.455719 | 1.762692 | 9.434363 | 67.684875 | 200 | 193 | 7/0/0 | 0.201321 |
| 300k | 0.193039 | 22.874964 | 26.862995 | 4.028174 | 15.358876 | 65.773834 | 240 | 231 | 9/0/0 | 0.199085 |

### H29: held-out Diagnostic

| Native | Macro | Updates | RSR | C/D/S | Energy kWh | Time s | Target speed min/mean/max m/s | Override | Influence |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 50k | 5,018 | 4,137 | 18/18 | 18/18/18 | 0.180334 | 124.650 | 5.597 / 8.662 / 15.429 | 17.297% | 50.354% |
| 100k | 10,034 | 9,310 | 18/18 | 18/18/18 | 0.179878 | 126.506 | 5.660 / 9.387 / 16.649 | 11.581% | 51.970% |
| 150k | 15,047 | 14,482 | 18/18 | 18/18/18 | 0.179965 | 124.839 | 5.645 / 9.513 / 16.296 | 12.069% | 51.119% |
| 200k | 20,063 | 19,655 | 18/18 | 18/18/18 | 0.178292 | 126.156 | 5.690 / 9.522 / 16.447 | 11.304% | 52.092% |
| 250k | 25,079 | 24,827 | 18/18 | 18/18/18 | 0.178023 | 126.700 | 5.673 / 9.407 / 16.517 | 11.633% | 52.030% |
| 300k | 30,097 | 30,000 | 18/18 | 18/18/18 | 0.177834 | 126.133 | 5.697 / 9.357 / 16.314 | 11.901% | 51.788% |

### H29: learner and real training experience

| Native | Alpha | Speed multiplier | Terminal multiplier | Reward loss | Cost loss | Actor loss | Episodes | Feasible | D/S/I | Energy kWh |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50k | 0.323959 | 0.000000 | 0.000000 | 7.059653 | 22.741165 | 28.511520 | 42 | 41 | 1/0/0 | 0.239830 |
| 100k | 0.165336 | 0.000000 | 0.000000 | 3.832492 | 9.931006 | 48.508274 | 81 | 80 | 1/0/0 | 0.216553 |
| 150k | 0.193334 | 0.000000 | 0.000000 | 3.846311 | 15.147330 | 61.645821 | 120 | 117 | 3/0/0 | 0.207080 |
| 200k | 0.194908 | 0.000000 | 1.657717 | 6.121083 | 26.689913 | 69.335556 | 159 | 155 | 4/0/0 | 0.202185 |
| 250k | 0.200775 | 0.000000 | 6.648282 | 4.102942 | 18.914412 | 67.401192 | 199 | 194 | 5/0/0 | 0.199190 |
| 300k | 0.208638 | 0.000000 | 11.848896 | 3.748310 | 13.586204 | 65.304222 | 239 | 233 | 6/0/0 | 0.196630 |

### H47: held-out Diagnostic

| Native | Macro | Updates | RSR | C/D/S | Energy kWh | Time s | Target speed min/mean/max m/s | Override | Influence |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 50k | 5,019 | 4,137 | 18/18 | 18/18/18 | 0.180030 | 124.194 | 5.617 / 9.308 / 15.763 | 11.595% | 49.309% |
| 100k | 10,036 | 9,310 | 18/18 | 18/18/18 | 0.178102 | 130.044 | 5.622 / 8.525 / 16.506 | 13.162% | 56.348% |
| 150k | 15,054 | 14,482 | 18/18 | 18/18/18 | 0.177225 | 129.528 | 5.630 / 8.214 / 16.334 | 16.041% | 55.544% |
| 200k | 20,072 | 19,655 | 18/18 | 18/18/18 | 0.176807 | 127.961 | 5.630 / 9.016 / 16.579 | 11.427% | 54.096% |
| 250k | 25,093 | 24,827 | 18/18 | 18/18/18 | 0.177921 | 127.217 | 5.617 / 9.252 / 16.590 | 9.979% | 53.107% |
| 300k | 30,110 | 30,000 | 18/18 | 18/18/18 | 0.177694 | 127.228 | 5.672 / 9.105 / 16.157 | 10.270% | 52.509% |

### H47: learner and real training experience

| Native | Alpha | Speed multiplier | Terminal multiplier | Reward loss | Cost loss | Actor loss | Episodes | Feasible | D/S/I | Energy kWh |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50k | 0.324802 | 0.000000 | 3.832105 | 13.100458 | 16.863417 | 30.054344 | 42 | 42 | 0/0/0 | 0.232758 |
| 100k | 0.186238 | 0.000000 | 8.753930 | 8.360901 | 24.863276 | 50.735092 | 81 | 80 | 1/0/0 | 0.213319 |
| 150k | 0.166849 | 0.000000 | 13.802545 | 6.624300 | 7.909324 | 59.775974 | 119 | 115 | 4/0/0 | 0.205298 |
| 200k | 0.175977 | 0.000000 | 18.300749 | 4.706827 | 10.246434 | 66.167320 | 158 | 152 | 6/0/0 | 0.200664 |
| 250k | 0.183412 | 0.000000 | 22.976294 | 3.239197 | 8.136145 | 66.686722 | 197 | 189 | 8/0/0 | 0.197613 |
| 300k | 0.196658 | 0.407787 | 27.990887 | 2.245012 | 6.732901 | 62.869431 | 237 | 229 | 8/0/0 | 0.195614 |

All 18 Diagnostics achieved RSR, completion, deadline compliance, and right-endpoint speed compliance of 18/18. There were no Diagnostic failure modes, nonfinite values, or deadline-bound falsifications.

Real training experience comprised 716 completed episodes: 693 feasible, 23 deadline failures, 0 speed failures, and 0 incomplete episodes. These outcomes were preserved without intervention.

## Shield effectiveness

| Seed | Scope | Proposals | Overrides | Override | Influenced | Influence | Accepted | Fallback-identical | Deadline | Known braking | Endpoint speed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H11 | training | 300,000 | 42,974 | 14.325% | 166,658 | 55.553% | 166,658 | 90,368 | 40,952 | 1,940 | 82 |
| H11 | 300k Diagnostic | 22,819 | 2,525 | 11.065% | 12,297 | 53.889% | 12,297 | 7,997 | 2,354 | 165 | 6 |
| H29 | training | 300,000 | 43,289 | 14.430% | 161,963 | 53.988% | 161,963 | 94,748 | 41,284 | 1,916 | 89 |
| H29 | 300k Diagnostic | 22,704 | 2,702 | 11.901% | 11,758 | 51.788% | 11,758 | 8,244 | 2,528 | 165 | 9 |
| H47 | training | 300,000 | 43,711 | 14.570% | 155,927 | 51.976% | 155,927 | 100,362 | 41,800 | 1,830 | 81 |
| H47 | 300k Diagnostic | 22,901 | 2,352 | 10.270% | 12,025 | 52.509% | 12,025 | 8,524 | 2,242 | 107 | 3 |

The shield remained active and auditable. No override condition or causal fallback behavior changed after performance was observed.

## Frozen final policies and streams

| Seed | Final checkpoint SHA-256 | Actor parameter SHA-256 | Stream SHA-256 | Rows |
| --- | --- | --- | --- | ---: |
| H11 | `a16cd663a4cd046532490b74a414e37acee9b37ca0c0b886753fa8d767f4f7ed` | `2f574832349948dae731b57e4e85dca5ee797665865c5f332e7a7feb3612488e` | `08e8e5b0f03dd7caae95b3f1175edaa22a29608cbc9a554fea6e3be3e24cfe3c` | 865 |
| H29 | `b4c6f5f484cc4a68b044b39f70db8d5f8245098729b3bb06749dbb46bdb9af55` | `bd1de85fc0fe586e39831f2dea917fa5dd131530a625a58dd393f1e2094ace65` | `ca8912845c1293e86fd326a7415901c9beb0c70ae6a32db5e78263fceb355312` | 865 |
| H47 | `3a67097e07360a07c8d5e5a276cf388434476943941085aea98b4a85254d82cf` | `135486355eb36def746f38ef20638a0fdd9aab02370e392a108cda66f87dfad6` | `2b1323d664f8ca470975ce4238f33a1883bf6d88eb43d035681b5c1520b15255` | 865 |

Actor fingerprints hash lexicographically ordered actor-state entries including parameter name, dtype, shape, and contiguous tensor bytes. The full checkpoint hash is the canonical artifact identity. Runtime task/reset content was verified by the frozen runner against each canonical 865-row stream; no stream was regenerated, filtered, or reordered.

## Runtime, disk, and checkpoint overhead

- Matrix interval: 25255.510 s (7.015 h), within 24 h.
- Summed active runner walltime: 1010.761 s (16.846 min).
- Retained artifacts: 1,128,646,434 bytes in 82 files (1.129 GB), within 12 GB.
- Scientific checkpoints: 962,548,055 bytes; aggregate write time 1.933 s.
- A host/session suspension during H47 enlarged elapsed wall-clock time. The same foreground matrix process continued; no recovery event or duplicate Attempt occurred.

## Integrity and recovery

- PASS — Matrix manifest completed.
- PASS — All checkpoint hashes verified.
- PASS — All diagnostics finite.
- PASS — No deadline-bound falsification.
- PASS — Update schedule exact.
- PASS — Replay equals completed macro count.
- PASS — Lease released.
- PASS — No duplicate Attempt.
- PASS — Source hashes match execution freeze.
- PASS — Preparation artifact hashes match.
- PASS — Exact recovery preflight passed.
- PASS — Physical fallback verified.
- PASS — Shield safety audit passed.

No interruption occurred: every seed has resume count 0 and the recovery-event ledger is empty. Exact mid-block recovery parity remains established by the pre-authorized probe.

## Descriptive interpretation

`FEASIBILITY_AND_ENERGY_PROGRESS`

Feasibility was preserved at every Diagnostic checkpoint and mean feasible-only Diagnostic energy decreased from 50k to 300k for every seed. Losses, alpha, and multiplier trajectories remained finite. This is descriptive only: it performs no model selection, assigns no final acceptance label, and does not authorize Validation.

## Downstream seal

- `HYBRID_RL_VALIDATION_V1`: `FROZEN_UNOPENED`, authorized=false, episodes=0.
- Paper: `SEALED_UNINSPECTED`, authorized=false, episodes=0, content fingerprints=0.

The causal controller, non-learning baseline, and Hybrid policies were not evaluated on Validation. Paper seeds 4000–4017 were not instantiated, fingerprinted, hashed, inspected, or evaluated.

## Machine-readable record

The complete checkpoint-level record is `main_training_results.json` in this directory.
