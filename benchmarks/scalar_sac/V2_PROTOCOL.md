# Scalar Reward Study V2 protocol (frozen before evaluation)

## Purpose and provenance

Scalar Study V1 remains unchanged in `canonical.json`, `RESULTS.md`, and
`results-140s.json`. It established that superficially reasonable weights can
make inactivity preferable to completing the task. V2 is a separate,
analytically scaled scalar baseline; it does not reinterpret or overwrite V1.

The operational task and algorithm-independent evaluation are unchanged:
complete 1,000 m in at most 140 s, never exceed the applicable speed limit, and
then compare signed net energy among feasible episodes. Requirement Satisfaction
Rate (RSR), not return, is the primary outcome.

## Four-component V2 reward

V2 has four conceptual components:

```text
route achievement = w_progress * delta_position / 1000 m
                  + B_on_time on an on-time route completion

energy            = -w_energy * step_energy / 0.25 kWh

time              = -w_time * delta_time / 140 s

speed violation   = -w_speed_integral * speed_excess * delta_time / 1 m
                    -P_speed_event once at episode end if any excess occurred
```

The fixed event penalty makes the exact zero-excess requirement visible even
when the integrated excess is small. It is part of the same speed concept, not
an additional evaluation criterion. The completion bonus is deliberately
limited to completion at or before 140 s. Feasibility and physical metrics are
still computed independently from the reward.

## Episode-level scale analysis

The following representative behaviors were fixed before V2 training:

- standstill: 180 s and 0.0404 kWh;
- feasible efficient: 115 s, 0.23 kWh, no speed excess;
- feasible inefficient: 115 s, 0.30 kWh, no speed excess;
- fast violating: 90 s, 0.28 kWh, 5 m integrated excess and one event;
- slow completion: 160 s, 0.23 kWh, no speed excess.

Approximate undiscounted episode returns are:

| candidate | standstill | feasible efficient | feasible inefficient | fast violating | slow completion |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2-balanced | -0.402 | 2.335 | 2.195 | -4.721 | 0.254 |
| v2-energy-oriented | -0.483 | 2.875 | 2.595 | -4.281 | -0.206 |
| v2-strong-compliance | -0.402 | 2.335 | 2.195 | -10.721 | 0.254 |

All candidates satisfy the required sanity condition `feasible completion >
standstill`. They also order the representative efficient feasible drive above
the inefficient feasible drive and the violating drive. Even the late
completion remains above standstill. This does not assert an ordering for every
physically possible trajectory; it prevents the V1 scale inversion in the
representative operating range.

The candidate meanings are:

- `v2-balanced`: the smallest correction, with a +2 on-time bonus and a -2
  any-violation penalty;
- `v2-energy-oriented`: doubles energy emphasis but increases the completion
  bonus to +3 so plausible completion remains clearly preferred to inactivity;
- `v2-strong-compliance`: retains balanced energy scaling and increases both
  speed penalties.

No Cartesian hyperparameter expansion is performed.

## Predefined acceptance criterion

A reward configuration is a credible scalar baseline only if, at 300,000
online training steps on the exploratory evaluation tracks:

1. every training seed has RSR at least 8/9 (88.9%);
2. every training seed completes at least 8/9 tracks;
3. every training seed is speed compliant on at least 8/9 tracks;
4. the RSR range across training seeds is at most 1/9;
5. for each seed, mean energy on tracks jointly feasible with the fast compliant
   oracle is at most 1.25 times the oracle's mean on those tracks.

The 8/9 threshold allows one stochastic-track failure while demanding that the
result hold for all three independently trained policies. It is stricter than
the conservative oracle's 7/9 RSR but below the privileged fast oracle's 9/9.
The energy bound is intentionally loose because the oracle is not optimized for
energy. Feasibility is evaluated before efficiency.

## Budget study and track splits

Each of the three candidates is trained for all seeds 11/29/47. A single
continued run is evaluated at 100,000 and 300,000 online steps; validation
learning curves are recorded every 50,000 steps. The fixed SAC architecture and
optimizer settings are retained from V1. Replay capacity is increased to
400,000 solely so the longer trajectory is not clipped by the old 200,000-entry
capacity.

Track groups are disjoint:

- development/calibration: 2000–2008;
- validation and learning curves: 3000–3008;
- V1 exploratory evaluation and V1→V2 comparison: 1000–1008;
- reserved final paper test: 4000–4017.

V1 already exposed tracks 1000–1008, so they are no longer a pristine blind
test set. Seeds 4000–4017 must not be generated, evaluated, plotted, or inspected
until the complete paper protocol and all compared methods are frozen. This V2
stage does not run them.

The 1M-step budget is not automatic. It is justified only if validation RSR is
still materially improving near 300k and the acceptance boundary appears
reachable. A plateau far below acceptance indicates that longer training alone
is not a useful next experiment.
