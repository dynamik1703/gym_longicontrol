# Feedback-Corrected Residual Action Chunking RL V1

Status: **protocol frozen; implementation/preflight in progress**.

This isolated study compares a jointly generated three-second residual action
chunk (`CHUNK3_RESIDUAL_SAC`) with an equally trained one-second residual SAC
(`RESIDUAL1S_SAC`). Both act only through the frozen AdaptivePrice MPC V2 and
its unchanged native physical requirements. The actor receives the causal 16D
MPC strategy observation and never receives route identity or hidden future
state.

The immutable starting point is commit `b1613e62ec381932cca3fb43ae698d2f75ffaa5e`
(tree `c1bc3f4eab0eb3a6d96966df25fe99a518493967`). Historical study files are
read-only inputs. Main training is allowed only after all preflight and resource
gates pass. Diagnostic evaluation is exploratory. Validation is
`FROZEN_UNOPENED`; Paper seeds 4000–4017 remain `SEALED_UNINSPECTED`.

The authoritative machine-readable protocol is `canonical.json`; split
identities are frozen in `split_ledger.json`.

