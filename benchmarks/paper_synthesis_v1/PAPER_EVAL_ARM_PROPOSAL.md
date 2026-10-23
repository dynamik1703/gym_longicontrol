# Draft Paper Controller-Arm Proposal

This is a recommendation, not execution authorization. The final inclusion
decision, checkpoint staging, and opening of Paper seeds require a subsequent
human approval.

## Recommended primary set

| Arm | Role | Frozen artifact status | Proposed status |
| --- | --- | --- | --- |
| CAUSAL60 | observation-limited causal physical anchor used in Hybrid V2 | source/config provenance recorded; no learned checkpoint | primary |
| TIME_AWARE | historical non-learning reference and attribution bridge | source/config provenance recorded; no learned checkpoint | primary |
| FixedPrice MPC V2 | strong richer-profile non-learning comparator | source/config provenance recorded; no learned checkpoint | primary |
| AdaptivePrice MPC V2 | strongest observed matched non-learning energy baseline | source/config provenance recorded; no learned checkpoint | primary energy reference |
| Hybrid H11 | frozen learned residual, seed 11 | checkpoint SHA-256 `a16cd663a4cd046532490b74a414e37acee9b37ca0c0b886753fa8d767f4f7ed` recorded by Hybrid V2 | primary |
| Hybrid H29 | frozen learned residual, seed 29 | checkpoint SHA-256 `b4c6f5f484cc4a68b044b39f70db8d5f8245098729b3bb06749dbb46bdb9af55` recorded by Hybrid V2 | primary |
| Hybrid H47 | frozen learned residual, seed 47 | checkpoint SHA-256 `3a67097e07360a07c8d5e5a276cf388434476943941085aea98b4a85254d82cf` recorded by Hybrid V2 | primary |

All three Hybrid seeds are included. No seed is selected using its Validation
energy result. Seven arms keep the comparison manageable while allowing the
historical architecture changes to be decomposed and learned residuals to be
tested against AdaptivePrice rather than only against TIME_AWARE.

## Conditional learning comparator

Constrained RL V2 has locally available final model files for seeds 11, 29,
and 47. A read-only audit measured these candidate SHA-256 values:

| Seed | Candidate local checkpoint SHA-256 |
| ---: | --- |
| 11 | `0fd0984dc49e26d159e1a251b84d400eb420285676fe6978365740796668dfc5` |
| 29 | `97c42e52044f8313a05542a655fced6f47a50d0f9b5465452765b39a42fe20fa` |
| 47 | `98b67166bd95b309330b96d32184fcb7c8b2c07c33fec2d2f4e8c3bc1062bbf1` |

The original Constrained V2 result record did not embed these file hashes.
Therefore they are **not automatically primary**. Inclusion requires a human
provenance decision, a frozen read-only artifact location, reconciliation of
checkpoint metadata with the original configuration, and recording of hashes
before Paper authorization. No replacement training is allowed.

If accepted, include all three seeds as a conditional secondary comparator;
never choose only the best historical seed.

## Optional contextual arms

- The privileged full-route physical reference may be included only as a
  separately labeled contextual ceiling after explicit authorization. It may
  not enter an information-matched ranking.
- Same-Information Scalar and GCSL checkpoints exist in historical local run
  material, but adding weak representatives expands cost and does not sharpen
  the primary learning-versus-strong-MPC question. Include at most one
  predeclared representative family, and only after immutable checkpoint and
  configuration provenance are accepted.
- Goal-SAC, contrastive depths, reward-search variants, terminated learned
  MBRL, SlackTrader, Temporal Options, and Counterfactual predictor are not
  proposed as final primary arms. Their completed negative/stopped evidence
  remains in the paper rather than being erased.

## Exclusions and rationale

- Missing checkpoints are `NOT_AVAILABLE_UNDER_CURRENT_AUTHORIZATION`; they
  must not be silently retrained.
- Diagnostic predictors are not controllers.
- Studies stopped before main training are not candidate learned policies.
- Cross-panel historical RSR is not a basis for choosing a single “winner.”
- Paper task content must not be opened to reduce or revise this set.

## Human decisions required

1. Approve or reject the seven-arm primary set.
2. Decide whether the newly audited Constrained V2 hashes meet the provenance
   threshold for secondary inclusion.
3. Decide whether the privileged reference or one weak learning comparator is
   worth the additional Paper pairs.
4. Approve exact checkpoint locations, source/config hashes, and evaluation
   ownership in a future execution authorization.
