# Hybrid RL Validation V2 Results

> **Amended post-incident Validation.** V1 remains permanently `INCIDENT_ABORTED`: one CAUSAL60 episode and one TIME_AWARE episode completed without persisted results, and one Hybrid environment was reset without an action. V1 was not replayed or relabeled.

Transaction: `439c821bc40c0a4dd8962af0ce08ef050f3ca697af15d8960a7340c8f860ab76`
Ordered split SHA-256: `99a863fb423e2ceb5ec0641bb940fab21ff7eb86a1466886da8707862a015e74`
Ledger: `COMPLETED_SEALED` — 180 completed, 0 pending, 0 started, 0 interrupted.

## Feasibility and physical outcomes

| Controller | RSR | Completion | Deadline | Speed | Mean travel time (s) | Mean feasible energy (kWh) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CAUSAL60 | 35/36 | 36/36 | 35/36 | 36/36 | 85.875000 | 0.193872666 |
| TIME_AWARE | 35/36 | 36/36 | 35/36 | 36/36 | 128.425000 | 0.177960483 |
| H11 | 35/36 | 36/36 | 35/36 | 36/36 | 129.391667 | 0.176148263 |
| H29 | 35/36 | 36/36 | 35/36 | 36/36 | 128.919444 | 0.175807674 |
| H47 | 35/36 | 36/36 | 35/36 | 36/36 | 129.869444 | 0.174965380 |

## Paired feasible-only energy

TIME_AWARE versus CAUSAL60: n=35, mean saving 7.674981%, median 7.836873%, range [-6.851282, 20.111698]%.

| Hybrid | vs CAUSAL60 n / mean saving | vs TIME_AWARE n / mean saving |
| --- | ---: | ---: |
| H11 | 35 / 8.643602% | 35 / 1.035175% |
| H29 | 35 / 8.825079% | 35 / 1.245039% |
| H47 | 35 / 9.247409% | 35 / 1.700130% |

Full medians, ranges, absolute-unit energy differences, per-track values, and deterministic descriptive 95% bootstrap intervals are in `validation_pairwise_analysis.json`.

## Strategy, Shield, and decision

- H11: target speed mean 8.691343 m/s; override fraction 0.116228; action-influence fraction 0.562590; influence gate PASS.
- H29: target speed mean 9.183625 m/s; override fraction 0.114025; action-influence fraction 0.546034; influence gate PASS.
- H47: target speed mean 8.805968 m/s; override fraction 0.104378; action-influence fraction 0.562103; influence gate PASS.

Frozen classification: **`FEASIBILITY_PRESERVED_WITHOUT_TARGET_ENERGY_GAIN`**.

No training, optimizer update, checkpoint selection, reward change, Shield change, split change, post-hoc tuning, V1 replay, or Paper interaction occurred. Diagnostic-to-Validation comparisons and complete intervention/failure details are retained in the machine-readable artifacts.
