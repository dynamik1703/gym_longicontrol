# Source and Physics Audit

## Frozen lineage

- Repository: `dynamik1703/gym_longicontrol`
- Source commit: `1c63ec85673ad0f3bf4afb4b05aa4f182c353ecc`
- Source study: completed Hybrid RL Validation V2
- New branch: `research/slacktrader-v1`
- New directory only: `benchmarks/slacktrader_v1/`

No existing environment, previous benchmark, checkpoint, result, canonical
stream or Paper protection was modified.

## Authoritative production sources

| Source | SHA-256 at source commit |
| --- | --- |
| `src/gym_longicontrol/domain/dynamics.py` | `cb137989abe9954e719f445b72dd0aa39f25408835311ac9298eb16aa38afc08` |
| `src/gym_longicontrol/domain/vehicle.py` | `ce70255d711cb1479ab7a60f59d2c8d007e5767ac38b0c7bc3742bd2811f47b9` |
| BMW i3 power artifact | `88a63fddfabf0f61123b3eee90fc84dba363911697d8e5cc10eb1685393d866d` |
| frozen causal fallback | `6934594424a87231bfe9d9702ad2fe6ab0f9f962cf8aa6f7a730e154c425dac7` |
| frozen Hybrid shield source | `99f38f33d69267d9e204e99ad1e7116735fa48ce5367907c73c197edec4363ef` |
| frozen viability source | `3303459dccfe9a6d48118c83db41c59ed1c1396ac019d6a8e0f8f7ff60c61c6f` |
| corrected Hybrid SAC source | `feffe8168e72d4d8bd7c87cfb303e5e6ad0b3cc5ef92658cf8b31375e18c2520` |
| corrected SMDP source | `494839c92ab077584b72b4de7e53511402a40daee3d304e0dc34fd2ce250461f` |
| canonical training streams | `6ea37c48fa2f3cbbf43de8eac62cb96645485169e0a2f13f4bffcf7af6c0da05` |

The planner calls the same `advance`, `VehicleModel`, and BMW-i3 power model as
the environment. Candidate trajectories are not evaluated with a surrogate.

## Information boundary

Online code receives only the ordinary 8D observation, propagated controller
state/time, remembered visible transitions, exact known dynamics, task
constants, generator bounds and the scalar price. The controller API accepts no
environment, `Track`, seed, Oracle trajectory or future state. Full-route access
is confined to the separately labeled Development contextual reference.

The runner's stream fingerprint check may inspect the environment track solely
for provenance verification, as inherited from the canonical runner; the
controller and policy never receive that object or fingerprint.
