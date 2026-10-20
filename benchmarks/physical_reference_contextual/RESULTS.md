# Contextual Physical-Reference Results

Status: **complete**. This is a post-hoc contextual analysis on two
previously opened RL Validation panels, not a new independent held-out
generalization benchmark. No RL policy was loaded, rerun or adapted.

## Physical outcomes

| Panel | Reference | RSR | Completion | Deadline | Speed | Feasible energy mean / median (kWh) | Mean travel time (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `GCSL_VALIDATION_V1` | oracle | 18/18 | 18/18 | 18/18 | 18/18 | 0.152418295 / 0.149577518 | 130.583 |
| `GCSL_VALIDATION_V1` | causal | 18/18 | 18/18 | 18/18 | 18/18 | 0.199774676 / 0.198297557 | 95.922 |
| `SAMEINFO_VALIDATION_V1` | oracle | 18/18 | 18/18 | 18/18 | 18/18 | 0.152067763 / 0.149577518 | 130.750 |
| `SAMEINFO_VALIDATION_V1` | causal | 18/18 | 18/18 | 18/18 | 18/18 | 0.194927079 / 0.189381393 | 90.150 |

Both references achieved full feasibility on both panels. This is an
achieved result for these 36 tasks, not a guarantee for all generated
tracks. Every selected Oracle trajectory has exact replay parity.

## Paired feasible-only reference differences

| Panel | Pairs | Causal−Oracle energy mean / median / range (kWh) | Aggregate mean difference | Travel-time difference mean |
| --- | ---: | ---: | ---: | ---: |
| `GCSL_VALIDATION_V1` | 18 | 0.047356381 / 0.048720039 / [0.013860184, 0.099120016] | 31.070% | -34.661 s |
| `SAMEINFO_VALIDATION_V1` | 18 | 0.042859317 / 0.039803876 / [0.013110873, 0.099760093] | 28.184% | -40.600 s |

## RL failures with feasible physical witnesses

### GCSL_VALIDATION_V1

| Policy/method | Failed pairs | Oracle / causal witnesses | Distinct failed tracks | Oracle / causal solvable tracks | Speed / deadline / noncompletion |
| --- | ---: | ---: | ---: | ---: | ---: |
| C11 (seed 11) | 8 | 8 / 8 | 8 | 8 / 8 | 3 / 7 / 1 |
| C29 (seed 29) | 12 | 12 / 12 | 12 | 12 / 12 | 3 / 12 / 7 |
| C47 (seed 47) | 12 | 12 / 12 | 12 | 12 / 12 | 2 / 12 / 8 |
| G11 (seed 11) | 18 | 18 / 18 | 18 | 18 / 18 | 18 / 0 / 0 |
| G29 (seed 29) | 18 | 18 / 18 | 18 | 18 / 18 | 18 / 0 / 0 |
| G47 (seed 47) | 18 | 18 / 18 | 18 | 18 / 18 | 17 / 2 / 2 |
| **constrained pooled** | 32 | 32 / 32 | 13 | 13 / 13 | 8 / 31 / 16 |
| **gcsl pooled** | 54 | 54 / 54 | 18 | 18 / 18 | 53 / 2 / 2 |

### SAMEINFO_VALIDATION_V1

| Policy/method | Failed pairs | Oracle / causal witnesses | Distinct failed tracks | Oracle / causal solvable tracks | Speed / deadline / noncompletion |
| --- | ---: | ---: | ---: | ---: | ---: |
| C11 (seed 11) | 10 | 10 / 10 | 10 | 10 / 10 | 3 / 9 / 1 |
| C29 (seed 29) | 11 | 11 / 11 | 11 | 11 / 11 | 3 / 9 / 4 |
| C47 (seed 47) | 10 | 10 / 10 | 10 | 10 / 10 | 2 / 10 / 5 |
| S11 (seed 11) | 18 | 18 / 18 | 18 | 18 / 18 | 0 / 18 / 18 |
| S29 (seed 29) | 18 | 18 / 18 | 18 | 18 / 18 | 0 / 18 / 18 |
| S47 (seed 47) | 18 | 18 / 18 | 18 | 18 / 18 | 1 / 18 / 18 |
| **constrained pooled** | 31 | 31 / 31 | 12 | 12 / 12 | 8 / 28 / 10 |
| **scalar pooled** | 54 | 54 / 54 | 18 | 18 / 18 | 1 / 54 / 54 |

Counts in the last column can overlap; the immutable exclusive modes
and all per-task comparisons are retained in `results.json`. Policy-
track pairs and distinct tracks are kept separate. SAMEINFO Constrained
failures are primarily deadline-related (28/31); Scalar failures are
noncompletion/deadline on all 54 pairs. GCSL Constrained failures are
primarily deadline-related (31/32); GCSL failures are primarily speed-
related (53/54).

RL/reference energy differences are reported only for jointly feasible
policy-track pairs in `paired_analysis.json`. No infeasible RL energy is
interpreted as an optimization result.

## Contextual interpretation

1. **Did the Oracle solve RL-failed tracks?** Yes: every distinct failed
   track and every failed policy-track pair in both panels has an Oracle
   feasible witness.
2. **Did the causal controller solve them?** Yes, with the same complete
   witness coverage and 18/18 panel RSR.
3. **Was hidden future-route information necessary here?** No for these
   observed tasks: the ordinary-observation causal controller also solved
   all tasks. This does not make hidden route information irrelevant in
   general.
4. **What drove RL failure?** SAMEINFO Scalar was crawling/noncompletion;
   GCSL was speed violation; both Constrained panels were predominantly
   deadline-limited with smaller speed/noncompletion components.
5. **How large was the physical energy gap?** Causal minus Oracle mean
   was 0.042859 kWh (28.184%) on SAMEINFO and 0.047356 kWh
   (31.070%) on GCSL. The causal controller was respectively 40.600 s
   and 34.661 s faster on average, leaving visible deadline reserve.
6. **Is an RL energy opportunity established?** The achieved gap
   motivates a feasibility-first energy study, but does not prove that
   RL can capture it. The Oracle is privileged and not globally optimal;
   a causal non-
   learning energy-aware control is required to identify learning value.

Limited preview therefore does not plausibly explain the observed
failures on these panels by necessity: a controller with the same 8D
sensor plus disclosed causal memory succeeded everywhere. It may still
matter for other controllers or generated tracks.

## Execution and integrity

- Scientific controller-task records: 72, all unique; duplicates: 0.
- Oracle internal search trajectories: 396 (11 frozen candidates/task).
- Independent Oracle replay checks: 36; replay steps: 47,040.
- RL policy episodes executed: 0.
- Frozen physical sources/configuration: verified.
- Immutable RL results/splits: read from exact terminal Git commits.
- Controller retuning after contextual results: none.
- Paper: `SEALED_UNINSPECTED`; content fingerprints constructed: 0.
