# Requirement-Conditioned RL V1 results

Status: **complete; Gate D (tight requirements fail systematically)**.

The preregistered claim is not supported at the required reliability. A single
FSRL SACLag policy does respond strongly and usually smoothly to the requested
deadline, including at unseen intermediate margins, but pooled Requirement
Satisfaction Rate (RSR) is only 48/135 (35.6%). The response is primarily a
time shift, not a credible energy-time trade-off, and it is highly dependent on
the training seed.

## Scope and integrity

- Training seeds: 11, 29, 47; exactly 300,000 native simulator transitions each.
- Development tracks: 2000--2008; Validation tracks: 3000--3008.
- Primary matrix: 3 policies x 9 tracks x 5 requirements = 135 episodes.
- Training margins: 20/40/60 s; unseen interpolation margins: 30/50 s.
- Objective, two costs, network, optimizer, replay, entropy, PID settings, and
  transition budget are identical to frozen Constrained V2.
- Completed training-episode exposure was balanced to within one episode:
  84/83/83, 86/87/86, and 91/90/90 for the three seeds.
- All configuration hashes and 39 raw result files passed validation.
- The reserved 4000--4017 paper-test tracks were not generated, inspected,
  evaluated, or plotted.

An initial execution exposed that FSRL's collector performs unused internal
resets which advanced the first version of the requirement sampler. That run
was invalidated. The corrected runner draws one outer-loop requirement per
actually collected episode and pins it across collector resets. No scientific
parameter was changed, and only the clean rerun is analyzed here. The amendment
is recorded in `PROTOCOL.md`.

## Physics-only requirement range

On Development, optimistic `T_min(start)` spans 55.686--117.000 s. The
speed-compliant reference needs 4.914--19.914 s beyond that bound. The frozen
selection rule rounds the maximum reference gap upward to a 20-s tight margin,
then uses:

```text
seen anchors:          20, 40, 60 s
unseen interpolation: 30, 50 s
T_max = T_min(start) + margin
```

Development absolute deadlines span 75.686--177.000 s. Validation was checked
only after selection: all tight requirements are reached by the reference, and
the complete primary Validation range is 71.786--177.000 s. Thus no primary RL
failure is charged to a requirement known to be physically impossible.

## Policy input and evaluation

The public eight-dimensional observation remains unchanged. The benchmark-only
wrapper appends:

```text
[T_max / 180 s, elapsed_time / 180 s]
```

Elapsed time makes the variable-deadline state Markov without exposing
`T_min`, slack, deficit, or an oracle trajectory. External feasibility remains
strict and reward-independent:

```text
completed
and travel_time_s <= episode T_max
and max_speed_violation_m_s <= 0
```

## Requirement satisfaction

Final 300k Validation results pooled over the three policies:

| Margin | Status | Feasible | RSR | Completion | Deadline met | Speed compliant | Mean feasible energy |
|---:|:---|---:|---:|---:|---:|---:|---:|
| 20 s | seen | 5/27 | 18.5% | 96.3% | 25.9% | 63.0% | 0.2113 kWh |
| 30 s | unseen | 8/27 | 29.6% | 96.3% | 37.0% | 66.7% | 0.2058 kWh |
| 40 s | seen | 10/27 | 37.0% | 92.6% | 44.4% | 70.4% | 0.2125 kWh |
| 50 s | unseen | 11/27 | 40.7% | 81.5% | 48.1% | 70.4% | 0.2126 kWh |
| 60 s | seen | 14/27 | 51.9% | 77.8% | 55.6% | 77.8% | 0.2149 kWh |
| **All** | mixed | **48/135** | **35.6%** | **88.9%** | **42.2%** | **69.6%** | **0.2120 kWh** |

RSR by policy seed and margin:

| Training seed | 20 s | 30 s | 40 s | 50 s | 60 s | Overall |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 0/9 | 0/9 | 0/9 | 0/9 | 3/9 | 3/45 (6.7%) |
| 29 | 2/9 | 2/9 | 3/9 | 3/9 | 3/9 | 13/45 (28.9%) |
| 47 | 3/9 | 6/9 | 7/9 | 8/9 | 8/9 | 32/45 (71.1%) |

The 70% pooled criterion, 50% per-requirement criterion, and 50% per-seed
criterion all fail. Seed 47 demonstrates that the formulation can be learned in
one run, but 3/45 for seed 11 and 13/45 for seed 29 make it unreliable.

## Controllability and requirement ignoring

The controller does not simply drive at one always-fast operating point:

| Preregistered diagnostic | Result | Criterion | Pass |
|:---|---:|---:|:---:|
| Tight-to-loose travel-time pair monotonicity | 262/270 (97.0%) | >=70% | yes |
| Mean within-group Spearman(`margin`, `travel time`) | 0.924 | descriptive | -- |
| Median ordered-pair travel-time change | +7.2 s | descriptive | -- |
| Median 20-to-60-s travel-time change | +20.6 s | descriptive | -- |
| Requirement-sensitive policy/track groups | 26/27 (96.3%) | >=66.7% | yes |
| Feasible energy pair monotonicity | 37/71 (52.1%) | >=60% | **no** |
| Mean feasible-energy Spearman correlation | +0.082 | expected negative | **no** |
| Median feasible ordered-pair energy change | +0.00085 kWh | expected <=0 | **no** |
| Median feasible 20-to-60-s energy change | +0.00527 kWh | expected <=0 | **no** |

The travel-time response is real and operationally large. It nevertheless often
overshoots the useful response: looser requests reduce completion from 96.3% at
20/30 s to 77.8% at 60 s because some policies slow down until the environment's
180-s horizon. More time also does not systematically save energy. The observed
behavior is therefore requirement-sensitive but not a meaningful energy-time
trade-off.

## Seen versus unseen interpolation

Seen anchors achieve 29/81 (35.8%); unseen intermediate margins achieve 19/54
(35.2%). The interpolation gap is only 0.6 percentage points and passes the
15-point criterion. Intermediate achieved times lie directionally between their
neighbors in 52/54 comparisons (96.3%), also passing. Feasible-energy direction
holds in only 8/14 eligible interpolation triplets (57.1%).

The policy therefore interpolates its *time response* without retraining, but
does not reliably satisfy either seen or unseen requirements. The small RSR gap
must not be mistaken for strong generalization when both absolute rates are low.

## Canonical 140-s cost of conditioning

The same final checkpoints were evaluated separately at absolute `T_max = 140
s` on all Validation tracks:

| Method / training seed | Feasible |
|:---|---:|
| Requirement-Conditioned seed 11 | 3/9 |
| Requirement-Conditioned seed 29 | 2/9 |
| Requirement-Conditioned seed 47 | 7/9 |
| **Requirement-Conditioned pooled** | **12/27 (44.4%)** |
| **Frozen Constrained V2** | **21/27 (77.8%)** |

Supporting multiple deadlines costs 9 feasible episodes, or 33.3 percentage
points, relative to the specialized frozen V2 baseline. Constrained V2 was not
retrained.

## Speed, deadline costs, and Lagrange dynamics

Tight requests are harder on both deadline and speed compliance. Pooled speed
compliance rises from 63.0% at 20 s to 77.8% at 60 s. Mean training deadline
cost return decreases coherently with slack: 21.00, 15.36, and 7.41 for the
20/40/60-s training margins. Median deadline return likewise falls from 1.575
to zero. Mean speed-cost return is 18.51, 17.87, and 22.54; it is not ordered by
deadline, so tight deadlines do not uniquely dominate speed violations.

There is one shared speed and one shared deadline multiplier rather than a
separate multiplier per requirement. Final `(lambda_speed, lambda_deadline)` is:

```text
seed 11: (3.312, 2.185)
seed 29: (2.469, 2.122)
seed 47: (2.985, 1.488)
```

Early peaks are much larger: speed 58.95--78.68 and deadline 23.00--27.56.
All tracked optimizer diagnostics remain finite. The cost split shows that tight
episodes contribute the greatest deadline pressure, while the single shared
dual state and highly nonstationary curves do not produce consistent all-seed
control.

## Learning stability

Seed 11 and seed 29 peak at 200k (37.8% and 48.9%) and finish 31.1 and 20.0
points lower. Seed 47 finishes at its peak of 71.1%. All three rise from 250k to
300k, but the complete curves oscillate sharply rather than showing uniform
convergence. The preregistered budget was not extended and no favorable earlier
checkpoint was selected.

## Answers to the research questions

1. **Physically feasible range:** Development supports the selected 20--60-s
   margins; primary absolute deadlines are at most 177 s and every Validation
   tight case is reached by the speed-compliant reference.
2. **Training requirements:** 20/40/60 s beyond `T_min(start)`.
3. **Reserved interpolation:** unseen 30/50-s margins.
4. **Observation:** normalized absolute `T_max` and elapsed time; no privileged
   `T_min`, slack, deficit, or oracle action.
5. **One policy, multiple deadlines:** one seed is useful, but the three-seed
   result is not reliable enough to support the claim.
6. **RSR:** 18.5/29.6/37.0/40.7/51.9% by ascending requirement; 6.7/28.9/71.1%
   by seed.
7. **Travel-time response:** yes; 97.0% pair monotonicity, rho 0.924, and a
   median +20.6 s tight-to-loose shift.
8. **Energy response:** no coherent saving; 52.1% pair monotonicity and positive
   median deltas.
9. **Always fast:** no. It changes behavior substantially, sometimes slowing so
   much that it fails completion/deadline requirements.
10. **Interpolation:** time response interpolates smoothly, while absolute RSR
    remains weak and energy direction fails.
11. **Canonical degradation:** 12/27 versus frozen 21/27 at 140 s.
12. **Tight speed compliance:** it degrades to 63.0%, versus 77.8% at 60 s.
13. **Lagrange behavior:** tight episodes dominate deadline cost; shared
    multipliers spike early and settle near 1.5--3.3, with large seed variance.
14. **Energy-time trade-off:** not demonstrated.
15. **Credible formulation:** the input and benchmark formulation are sound,
    and controllability exists, but this fixed-budget SACLag baseline is not yet
    a credible robust Requirement-Conditioned controller.

## Decision

**Gate D** applies: the tight 20-s requirement is below 50% while the 60-s level
reaches 51.9%. Interpolation and time sensitivity are promising, but they cannot
compensate for low pooled RSR, two weak training seeds, canonical degradation,
and the absence of energy savings.

Most importantly:

> A single LongiControl policy can use an explicit deadline to adjust its travel
> time, and that response generalizes directionally to unseen intermediate
> deadlines. Under the frozen SACLag formulation and 300k budget, it cannot do so
> reliably while satisfying completion, deadline, and strict speed requirements,
> nor does it realize the intended energy benefit.

No next research method is implemented in this stage.
