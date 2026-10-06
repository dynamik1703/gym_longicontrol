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

Generate exactly 3 new candidate reward files.
Return each candidate as standalone Python source plus a short, visible design rationale. Do not provide hidden reasoning.
Do not access any information outside this package.
Every candidate will be screened with the same Stable-Baselines3 SAC learner for exactly 50,000 simulator transitions using one fixed seed.
Only aggregate Development feedback may be returned in a later generation.
Do not edit the task, API, algorithm, training budget or selection rule.

## Permitted parent material

The following parents were selected mechanically. You may change coefficients or functional forms, add/remove components, or combine ideas. Do not infer unavailable experiment information.

### Parent g1-c02

Visible rationale:
Makes progress valuable only while current and upcoming speed constraints are respected, and adds steep current-violation and preview tolls. This directly targets the universal speed failures while retaining deadline-responsive urgency and completion incentives.

```python
from benchmarks.llm_reward.reward_api import RewardContext, RewardOutput


def compute_reward(ctx: RewardContext) -> RewardOutput:
    route_scale = max(ctx.route_length_m, 1.0)
    remaining_distance = max(ctx.route_length_m - ctx.position_m, 0.0)
    remaining_time = max(ctx.time_budget_s - ctx.elapsed_time_s, ctx.dt_s, 0.1)
    required_speed = remaining_distance / remaining_time

    limit_scale = max(ctx.speed_limit_m_s, 1.0)
    current_excess = max(ctx.velocity_m_s - ctx.speed_limit_m_s, 0.0)
    current_ratio = current_excess / limit_scale

    preview_ceiling = max(ctx.speed_limit_m_s, 0.0)
    if ctx.future_speed_limit_1_m_s < ctx.speed_limit_m_s:
        preview_ceiling = min(
            preview_ceiling,
            max(ctx.future_speed_limit_1_m_s, 0.0)
            + max(ctx.distance_to_future_limit_1_m, 0.0) / 8.0,
        )
    if ctx.future_speed_limit_2_m_s < ctx.speed_limit_m_s:
        preview_ceiling = min(
            preview_ceiling,
            max(ctx.future_speed_limit_2_m_s, 0.0)
            + max(ctx.distance_to_future_limit_2_m, 0.0) / 12.0,
        )
    preview_excess = max(ctx.velocity_m_s - preview_ceiling, 0.0)
    preview_ratio = preview_excess / max(preview_ceiling, 1.0)

    compliance_gate = 1.0 / (
        1.0 + (8.0 * current_ratio) ** 4 + (3.0 * preview_ratio) ** 4
    )
    urgency = 1.0 + min(
        required_speed / (1.0 + max(ctx.speed_limit_m_s, 0.0)), 1.5
    )

    forward = max(ctx.delta_position_m, 0.0)
    backward = min(ctx.delta_position_m, 0.0)
    compliant_progress = (
        700.0 * forward / route_scale * compliance_gate * urgency
    )
    reverse_cost = 700.0 * backward / route_scale

    current_violation_toll = -ctx.dt_s * (
        30.0 * current_excess * current_excess
        + 4.0 * current_excess ** 4 / (1.0 + current_excess * current_excess)
    )
    preview_toll = -6.0 * ctx.dt_s * preview_excess * preview_excess
    energy_cost = -3.0 * ctx.step_energy_kwh

    progress_fraction = min(max(ctx.position_m / route_scale, 0.0), 1.0)
    terminal = 0.0
    if ctx.completed:
        terminal = 1200.0
        if ctx.elapsed_time_s <= ctx.time_budget_s:
            terminal += 240.0
    elif ctx.episode_ended:
        terminal = -1200.0 - 120.0 * max(1.0 - progress_fraction, 0.0)

    reward = (
        compliant_progress
        + reverse_cost
        + current_violation_toll
        + preview_toll
        + energy_cost
        + terminal
    )
    return RewardOutput(
        reward=reward,
        components={
            "compliant_progress": compliant_progress,
            "reverse_cost": reverse_cost,
            "current_violation_toll": current_violation_toll,
            "preview_toll": preview_toll,
            "energy_cost": energy_cost,
            "terminal": terminal,
        },
    )
```

Sanitized Development reflection:

```json
{
  "candidate_id": "g1-c02",
  "candidate_source_sha256": "714ed4124e98b2e12d1d89a72edb78d9ab1f04de8fd88e2e18bb74350417fd47",
  "development_metrics": {
    "completion_rate": 0.3333333333333333,
    "deadline_compliance_rate_among_completed": 1.0,
    "episode_count": 9,
    "episode_length_steps": {
      "count": 9,
      "max": 1800.0,
      "mean": 1613.7777777777778,
      "median": 1800.0,
      "min": 1207.0
    },
    "failure_mode_counts": {
      "feasible": 2,
      "incomplete+time": 2,
      "incomplete+time+speed": 4,
      "speed": 1
    },
    "feasible_energy_kwh": {
      "count": 2,
      "max": 0.3177860197003587,
      "mean": 0.27236955005465935,
      "median": 0.27236955005465935,
      "min": 0.22695308040896
    },
    "integrated_speed_violation_m": {
      "count": 9,
      "max": 1.1401606557643582,
      "mean": 0.2678285286016695,
      "median": 0.06063609639479033,
      "min": 0.0
    },
    "max_speed_violation_m_s": {
      "count": 9,
      "max": 2.3308995697226456,
      "mean": 0.5942576795379627,
      "median": 0.3695030006734008,
      "min": 0.0
    },
    "mean_integrated_speed_violation_m_among_relevant": 0.265025594720508,
    "normalized_route_progress_for_incomplete": {
      "count": 6,
      "max": 0.8184982902970378,
      "mean": 0.528153803836358,
      "median": 0.5162130335380467,
      "min": 0.29708806787512393
    },
    "requirement_satisfaction_rate": 0.2222222222222222,
    "speed_compliance_rate_among_relevant": 0.5,
    "travel_time_s": {
      "count": 9,
      "max": 179.99999999999406,
      "mean": 161.37777777777288,
      "median": 179.99999999999406,
      "min": 120.69999999999742
    }
  },
  "episode_reward_statistics": {
    "count": 33,
    "max": 2428.2447610676754,
    "mean": 293.53373343497253,
    "min": -4218.96481280867,
    "std": 1733.6789015567022
  },
  "generation": 1,
  "information_scope": "development-only-aggregate",
  "protocol_id": "llm-reward-search-v1",
  "ranking_fields": {
    "completion_rate": 0.3333333333333333,
    "deadline_compliance_among_completed": 1.0,
    "development_rsr": 0.2222222222222222,
    "mean_feasible_energy_kwh": 0.27236955005465935,
    "median_incomplete_route_progress": 0.5162130335380467,
    "speed_compliance_among_relevant": 0.5,
    "speed_violation_severity_among_relevant": 0.265025594720508
  },
  "reward_component_statistics": {
    "compliant_progress": {
      "count": 50000,
      "max": 2.338258588230673,
      "mean": 0.5473417639057226,
      "min": 0.0,
      "std": 0.4368961617534143
    },
    "current_violation_toll": {
      "count": 50000,
      "max": -0.0,
      "mean": -0.3406997593952075,
      "min": -104.12006534058897,
      "std": 3.1268937141777724
    },
    "energy_cost": {
      "count": 50000,
      "max": 0.002894928149846877,
      "mean": -0.000497708323026315,
      "min": -0.007226137511299481,
      "std": 0.0014480015718523036
    },
    "preview_toll": {
      "count": 50000,
      "max": -0.0,
      "mean": -0.06998539729405248,
      "min": -18.442493285114892,
      "std": 0.6040291880048629
    },
    "reverse_cost": {
      "count": 50000,
      "max": 0.0,
      "mean": 0.0,
      "min": 0.0,
      "std": 0.0
    },
    "terminal": {
      "count": 50000,
      "max": 1440.0,
      "mean": 0.06755648175779198,
      "min": -1316.5992415032551,
      "std": 34.08673539052822
    }
  },
  "schema_version": 1,
  "training_success_trajectory": {
    "completed_training_episode_count": 33,
    "development_checkpoints": [
      {
        "completion_rate": 0.7777777777777778,
        "cumulative_training_successes": 2,
        "requirement_satisfaction_rate": 0.0,
        "simulator_transitions": 10000
      },
      {
        "completion_rate": 0.7777777777777778,
        "cumulative_training_successes": 4,
        "requirement_satisfaction_rate": 0.1111111111111111,
        "simulator_transitions": 25000
      },
      {
        "completion_rate": 0.3333333333333333,
        "cumulative_training_successes": 7,
        "requirement_satisfaction_rate": 0.2222222222222222,
        "simulator_transitions": 50000
      }
    ],
    "first_success_training_step": 6593,
    "successful_training_episode_count": 7
  }
}
```

### Parent g1-c03

Visible rationale:
Tracks the deadline-implied speed only up to a buffered ceiling derived from current and visible future limits. A boundary guard, compliance-gated progress, and mild smoothness and energy costs aim for timely, anticipatory, efficient legal driving.

```python
from benchmarks.llm_reward.reward_api import RewardContext, RewardOutput


def compute_reward(ctx: RewardContext) -> RewardOutput:
    route_scale = max(ctx.route_length_m, 1.0)
    remaining_distance = max(ctx.route_length_m - ctx.position_m, 0.0)
    remaining_time = max(ctx.time_budget_s - ctx.elapsed_time_s, ctx.dt_s, 0.1)
    required_speed = remaining_distance / remaining_time

    buffered_ceiling = 0.95 * max(ctx.speed_limit_m_s, 0.0)
    if ctx.future_speed_limit_1_m_s < ctx.speed_limit_m_s:
        first_ceiling = (
            0.95 * max(ctx.future_speed_limit_1_m_s, 0.0)
            + max(ctx.distance_to_future_limit_1_m, 0.0) / 7.0
        )
        buffered_ceiling = min(buffered_ceiling, first_ceiling)
    if ctx.future_speed_limit_2_m_s < ctx.speed_limit_m_s:
        second_ceiling = (
            0.95 * max(ctx.future_speed_limit_2_m_s, 0.0)
            + max(ctx.distance_to_future_limit_2_m, 0.0) / 11.0
        )
        buffered_ceiling = min(buffered_ceiling, second_ceiling)

    target_speed = min(max(required_speed, 0.0), max(buffered_ceiling, 0.0))
    tracking_error = ctx.velocity_m_s - target_speed
    target_tracking = -0.8 * ctx.dt_s * tracking_error * tracking_error

    current_excess = max(ctx.velocity_m_s - ctx.speed_limit_m_s, 0.0)
    boundary_excess = max(ctx.velocity_m_s - buffered_ceiling, 0.0)
    speed_guard = -ctx.dt_s * (
        35.0 * current_excess * current_excess
        + 5.0 * boundary_excess * boundary_excess
    )

    forward = max(ctx.delta_position_m, 0.0)
    backward = min(ctx.delta_position_m, 0.0)
    compliance_gate = 1.0 / (1.0 + (4.0 * current_excess) ** 2)
    route_progress = 650.0 * forward / route_scale * compliance_gate
    reverse_cost = 650.0 * backward / route_scale

    acceleration_change = (
        ctx.acceleration_m_s2 - ctx.previous_acceleration_m_s2
    )
    smoothness_cost = -0.02 * ctx.dt_s * acceleration_change * acceleration_change
    energy_cost = -4.0 * ctx.step_energy_kwh

    progress_fraction = min(max(ctx.position_m / route_scale, 0.0), 1.0)
    terminal = 0.0
    if ctx.completed:
        terminal = 1200.0
        if ctx.elapsed_time_s <= ctx.time_budget_s:
            terminal += 240.0
    elif ctx.episode_ended:
        terminal = -1200.0 - 120.0 * max(1.0 - progress_fraction, 0.0)

    reward = (
        route_progress
        + reverse_cost
        + target_tracking
        + speed_guard
        + smoothness_cost
        + energy_cost
        + terminal
    )
    return RewardOutput(
        reward=reward,
        components={
            "route_progress": route_progress,
            "reverse_cost": reverse_cost,
            "target_tracking": target_tracking,
            "speed_guard": speed_guard,
            "smoothness_cost": smoothness_cost,
            "energy_cost": energy_cost,
            "terminal": terminal,
        },
    )
```

Sanitized Development reflection:

```json
{
  "candidate_id": "g1-c03",
  "candidate_source_sha256": "960b10469e4f1d43b6e159c76ad91a55ddae15330b610c4cb481d499151af21d",
  "development_metrics": {
    "completion_rate": 0.6666666666666666,
    "deadline_compliance_rate_among_completed": 0.16666666666666666,
    "episode_count": 9,
    "episode_length_steps": {
      "count": 9,
      "max": 1800.0,
      "mean": 1621.3333333333333,
      "median": 1639.0,
      "min": 1370.0
    },
    "failure_mode_counts": {
      "feasible": 1,
      "incomplete+time": 3,
      "time": 5
    },
    "feasible_energy_kwh": {
      "count": 1,
      "max": 0.221397190987718,
      "mean": 0.221397190987718,
      "median": 0.221397190987718,
      "min": 0.221397190987718
    },
    "integrated_speed_violation_m": {
      "count": 9,
      "max": 0.0,
      "mean": 0.0,
      "median": 0.0,
      "min": 0.0
    },
    "max_speed_violation_m_s": {
      "count": 9,
      "max": 0.0,
      "mean": 0.0,
      "median": 0.0,
      "min": 0.0
    },
    "mean_integrated_speed_violation_m_among_relevant": 0.0,
    "normalized_route_progress_for_incomplete": {
      "count": 3,
      "max": 0.8501746602957327,
      "mean": 0.6037003997208034,
      "median": 0.6702127313328996,
      "min": 0.2907138075337778
    },
    "requirement_satisfaction_rate": 0.1111111111111111,
    "speed_compliance_rate_among_relevant": 1.0,
    "travel_time_s": {
      "count": 9,
      "max": 179.99999999999406,
      "mean": 162.1333333333284,
      "median": 163.89999999999498,
      "min": 136.9999999999965
    }
  },
  "episode_reward_statistics": {
    "count": 35,
    "max": 1627.609017575509,
    "mean": -3537.3814101793214,
    "min": -25060.173666357994,
    "std": 6185.636148198877
  },
  "generation": 1,
  "information_scope": "development-only-aggregate",
  "protocol_id": "llm-reward-search-v1",
  "ranking_fields": {
    "completion_rate": 0.6666666666666666,
    "deadline_compliance_among_completed": 0.16666666666666666,
    "development_rsr": 0.1111111111111111,
    "mean_feasible_energy_kwh": 0.221397190987718,
    "median_incomplete_route_progress": 0.6702127313328996,
    "speed_compliance_among_relevant": 1.0,
    "speed_violation_severity_among_relevant": 0.0
  },
  "reward_component_statistics": {
    "energy_cost": {
      "count": 50000,
      "max": 0.003940711089313923,
      "mean": -0.000657619214008767,
      "min": -0.009645323405447211,
      "std": 0.001682669457277686
    },
    "reverse_cost": {
      "count": 50000,
      "max": 0.0,
      "mean": 0.0,
      "min": 0.0,
      "std": 0.0
    },
    "route_progress": {
      "count": 50000,
      "max": 1.1643842643227587,
      "mean": 0.38512143831164075,
      "min": 0.0,
      "std": 0.2700974824943214
    },
    "smoothness_cost": {
      "count": 50000,
      "max": -0.0,
      "mean": -0.00902773396303042,
      "min": -0.071757161003582,
      "std": 0.015212471260506856
    },
    "speed_guard": {
      "count": 50000,
      "max": -0.0,
      "mean": -0.5158937601852462,
      "min": -204.17490006025236,
      "std": 6.285698767645841
    },
    "target_tracking": {
      "count": 50000,
      "max": -1.965507917832473e-08,
      "mean": -2.7785662900198957,
      "min": -44.38622358900634,
      "std": 4.645496981166864
    },
    "terminal": {
      "count": 50000,
      "max": 1440.0,
      "mean": 0.43450071940722534,
      "min": -1311.7821834008787,
      "std": 35.57349919694124
    }
  },
  "schema_version": 1,
  "training_success_trajectory": {
    "completed_training_episode_count": 35,
    "development_checkpoints": [
      {
        "completion_rate": 0.1111111111111111,
        "cumulative_training_successes": 2,
        "requirement_satisfaction_rate": 0.1111111111111111,
        "simulator_transitions": 10000
      },
      {
        "completion_rate": 1.0,
        "cumulative_training_successes": 5,
        "requirement_satisfaction_rate": 0.3333333333333333,
        "simulator_transitions": 25000
      },
      {
        "completion_rate": 0.6666666666666666,
        "cumulative_training_successes": 12,
        "requirement_satisfaction_rate": 0.1111111111111111,
        "simulator_transitions": 50000
      }
    ],
    "first_success_training_step": 5720,
    "successful_training_episode_count": 12
  }
}
```
