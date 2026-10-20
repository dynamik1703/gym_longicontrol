# Source and Physics Audit

## Authoritative execution path

The reference uses the unchanged BMW i3-based production path:

- `src/gym_longicontrol/domain/dynamics.py::advance`
- `src/gym_longicontrol/domain/vehicle.py::VehicleModel`
- `src/gym_longicontrol/envs/longicontrol.py::LongiControlEnv.step`
- external `TaskSpecification`, `EpisodeMetrics` and `is_feasible`

All candidate and accepted actions are executed by the original Gymnasium
environment. The private computations choose actions; they do not replace the
authoritative rollout.

## State, action and integration

`VehicleState` contains position, velocity, acceleration, absolute jerk,
elapsed time and signed accumulated net energy. The action is clipped to
`[-1, 1]`. `VehicleModel.acceleration_from_action` interpolates between
velocity-dependent braking/coasting/traction envelopes built from the frozen
BMW i3 power constraints (`-50/+75 kW`) and acceleration bounds (`-3/+3 m/s²`).

Each 0.1 s step uses constant acceleration:

```text
v[t+1] = v[t] + a[t] dt
x[t+1] = x[t] + v[t] dt + 0.5 a[t] dt²
```

Velocity boundary crossings are clipped within the unchanged step. Power is
evaluated by the production BMW i3 MLP at `(v[t+1], a[t])`; signed energy is
`power_kw * dt / 3600`. Negative power is regenerative energy. The study did
not substitute generic vehicle equations.

## Speed and termination semantics

Track limits are right-continuous: a transition at `x` applies at exactly `x`.
After every completed integration step, the environment evaluates the limit at
`position[t+1]` and measures excess using `velocity[t+1]`. Equality is compliant;
any positive representable excess fails the canonical task. The controller
therefore checks both next position and next velocity.

Route completion is `position_m >= 1000`. The base environment never truncates;
the canonical `gym.wrappers.TimeLimit` truncates after 1,800 native steps.
Completion, deadline and speed compliance remain external metrics rather than
reward terms.

## Exact physical findings

Maximum braking from 100 to 60 km/h was reproduced directly:

| Quantity | Exact recorded value |
| --- | ---: |
| Steps | 60 |
| Time | 5.999999999999995 s |
| Distance | 135.0764556315701 m |
| Final speed | 16.603915137238552 m/s |

This verifies the earlier approximately 6.0 s / 135.08 m result. Other exact
maximum-braking paths are:

| Reduction | Time | Distance |
| --- | ---: | ---: |
| 60→20 km/h | 4.0 s | 45.05692925398084 m |
| 100→20 km/h | 10.0 s | 179.8249471715907 m |
| 80→40 km/h | 4.9 s | 83.55333320233326 m |
| 40→20 km/h | 1.9 s | 15.696111111111097 m |

The acceleration envelope is velocity dependent: maximum traction is 3 m/s²
through 16 m/s and then becomes power-limited; at 37 m/s it is about
0.8647 m/s². Maximum braking is -3 m/s² through 12 m/s and then becomes
power-limited; at 37 m/s it is about -1.4765 m/s².

At very low velocity, the historical interpolation approaches exactly zero
speed asymptotically because the braking acceleration at zero is zero. Reported
stopping profiles therefore use a declared practical standstill of `1e-6 m/s`:
20/40/60/80/100 km/h require respectively about
5.263/20.696/49.994/103.352/184.617 m.

The unconstrained, maximum-action, no-route-limit completion lower bound is
34.7 s. Per-track `T_min(x)` remains only an optimistic heuristic and is not
used as an exact minimum-time certificate.

## Preview-horizon consequence

A direct 100→20 km/h reduction needs about 179.8 m, beyond the 150 m sensor
horizon. The generator permits only 40 km/h adjacent changes, but two 100 m
spaced reductions (100→60→20) can expose the second transition too late for a
controller that continues at 100 km/h and reacts only when that second sign
becomes visible. Thus consecutive reductions can require an anticipation policy
beyond naive visible-sign braking.

The causal reference addresses this without hidden route access: its fixed
60 km/h cruise cap ensures that any newly visible legal reduction to the
20 km/h minimum is brakeable well within 150 m. This is conservative control,
not evidence that the partial-observation problem is absent in general.
