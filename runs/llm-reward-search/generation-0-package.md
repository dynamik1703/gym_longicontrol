# Fresh-session Reward Designer package

<!-- BEGIN TASK.md -->

# LongiControl reward-design task

Control a vehicle along a route.

Desired operational behavior, in priority order:

1. complete the route;
2. complete it within 140 seconds;
3. respect the speed limit throughout the route;
4. among successful behavior, prefer lower net energy consumption.

The training signal must be a deterministic scalar reward computed after each
simulator transition. Physical task success is measured separately from the
reward and must not be redefined by the reward designer.

Design the reward only from the fields documented in `ALLOWED_SIGNALS.md` and
the API in `REWARD_API.md`. Do not assume access to track identity, external
scores, saved files, network services, environment variables, optimal actions,
precomputed trajectories or other experiment artifacts.



<!-- BEGIN ALLOWED_SIGNALS.md -->

# Allowed reward signals

`RewardContext` is immutable. Every numeric value is finite and uses SI units
unless explicitly stated otherwise.

| Field | Unit | Meaning |
|---|---|---|
| `position_m` | m | Vehicle position after the current transition |
| `previous_position_m` | m | Vehicle position before the transition |
| `delta_position_m` | m | Current minus previous position |
| `velocity_m_s` | m/s | Velocity after the transition |
| `previous_velocity_m_s` | m/s | Velocity before the transition |
| `acceleration_m_s2` | m/s² | Acceleration after the transition |
| `previous_acceleration_m_s2` | m/s² | Acceleration before the transition |
| `action` | dimensionless | Applied continuous action in `[-1, 1]` |
| `speed_limit_m_s` | m/s | Current local speed limit |
| `future_speed_limit_1_m_s` | m/s | First visible future limit |
| `future_speed_limit_2_m_s` | m/s | Second visible future limit |
| `distance_to_future_limit_1_m` | m | Distance to the first visible limit change |
| `distance_to_future_limit_2_m` | m | Distance to the second visible limit change |
| `step_energy_kwh` | kWh | Signed net energy for the current transition |
| `cumulative_energy_kwh` | kWh | Signed net energy since episode reset |
| `elapsed_time_s` | s | Physical time since episode reset |
| `dt_s` | s | Simulator transition duration |
| `route_length_m` | m | Total route length |
| `time_budget_s` | s | Required maximum route-completion time |
| `completed` | bool | Whether the route ended by completion this step |
| `episode_ended` | bool | Whether completion or the fixed horizon ended the episode |

No other fields are available. In particular, candidates cannot access track
identity, experiment split identity, aggregate success scores, future outcomes,
hidden evaluator state, accumulated violation metrics, optimal trajectories,
optimal actions or any precomputed minimum-time quantity.



<!-- BEGIN REWARD_API.md -->

# Candidate reward API

Each candidate is one standalone UTF-8 Python file defining exactly one public
function:

```python
def compute_reward(ctx: RewardContext) -> RewardOutput: ...
```

The harness injects `RewardContext` and `RewardOutput`. A candidate may instead
import those two names from `benchmarks.llm_reward.reward_api`. `import math` is
the only other permitted import.

Return:

```python
RewardOutput(
    reward=<finite scalar>,
    components={<lower_snake_case_name>: <finite scalar>, ...},
)
```

The component dictionary is diagnostic only. Its values never affect physical
evaluation or candidate ranking directly. Component names must start with a
lowercase letter, contain only lowercase letters, digits and underscores, and
be at most 64 characters.

The function must be deterministic and free of side effects. It must not read
files, network state, process state or environment variables; call subprocesses;
use randomness; mutate global state; inspect track identities; or import other
project modules. The returned reward and every component must remain finite on
ordinary and terminal smoke contexts.

Store the short explicit design rationale separately from the source. Do not
include hidden reasoning.



## Generation instructions

Generate exactly 5 new candidate reward files.
Return each candidate as standalone Python source plus a short, visible design rationale. Do not provide hidden reasoning.
Do not access any information outside this package.
Every candidate will be screened with the same Stable-Baselines3 SAC learner for exactly 50,000 simulator transitions using one fixed seed.
Only aggregate Development feedback may be returned in a later generation.
Do not edit the task, API, algorithm, training budget or selection rule.
At least three of the five rewards must differ in functional form, not only in numeric coefficients.
No prior candidate or experiment feedback exists for this generation.
