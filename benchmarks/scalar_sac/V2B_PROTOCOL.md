# Scalar Reward Study V2-B protocol (frozen before evaluation)

## Why a second V2 iteration exists

V2-A (`canonical_v2.json`) corrected the *episode-level* V1 scale inversion,
but its dense speed cost was still poorly scaled for learning. On validation
tracks 3000–3008, five of six balanced/energy seed curves remained at zero
completion from 50k through 300k. `v2-balanced/seed47` briefly reached 4/9 RSR
at 150k, then collapsed to zero by 300k. All three V2-A candidates had 0/27
validation RSR at 300k. These observations were made without inspecting the
reserved paper-test tracks.

A 1M extension is not justified for flat or collapsing curves. V2-B is the
second and final planned reward-scaling iteration in this task. V2-A artifacts
remain unchanged.

## Change from V2-A

The reward formula and four conceptual components are unchanged. V2-B:

1. increases dense progress weight from 1 to 2, making normal forward movement
   locally rewarding after energy and time costs;
2. reduces integrated speed weights from 1–2 to 0.1–0.25, so small exploratory
   excess does not dominate many metres of compliant progress;
3. retains the terminal any-violation penalties of 2–3, preserving an explicit
   preference for exact compliance;
4. retains energy, time, and completion scales and all SAC hyperparameters.

This is scale repair, not a new shaping term or a Cartesian search.

## Frozen episode-level comparison

Using the same representative behaviors as `V2_PROTOCOL.md`, approximate
undiscounted returns are:

| candidate | standstill | feasible efficient | feasible inefficient | fast violating | slow completion |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2b-balanced | -0.402 | 3.335 | 3.195 | 0.779 | 1.254 |
| v2b-energy-oriented | -0.483 | 3.875 | 3.595 | 1.219 | 0.794 |
| v2b-strong-compliance | -0.402 | 3.335 | 3.195 | -0.971 | 1.254 |

Every candidate satisfies:

```text
feasible efficient > feasible inefficient > representative infeasible behavior
feasible completion > standstill
slow completion > standstill
```

V2-B uses the already frozen acceptance criteria, splits, 100k/300k comparison,
and 50k validation curve from `V2_PROTOCOL.md`. Seeds 4000–4017 remain reserved
and must not be evaluated in this study.
