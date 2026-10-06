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

### Parent g0-c02

Visible rationale:
Potential-difference shaping rewards gains in route completion while penalizing growth in schedule lag, with bounded overspeed shaping and terminal outcomes.

```python
from benchmarks.llm_reward.reward_api import RewardContext, RewardOutput


def compute_reward(ctx: RewardContext) -> RewardOutput:
    route_scale = max(ctx.route_length_m, 1.0)
    budget_scale = max(ctx.time_budget_s, 1.0)

    progress_now = min(max(ctx.position_m / route_scale, 0.0), 1.0)
    progress_before = min(max(ctx.previous_position_m / route_scale, 0.0), 1.0)
    time_now = max(ctx.elapsed_time_s, 0.0) / budget_scale
    time_before = max(ctx.elapsed_time_s - ctx.dt_s, 0.0) / budget_scale

    lag_now = max(time_now - progress_now, 0.0)
    lag_before = max(time_before - progress_before, 0.0)
    potential_now = 700.0 * progress_now - 180.0 * lag_now * lag_now
    potential_before = 700.0 * progress_before - 180.0 * lag_before * lag_before
    potential_change = potential_now - potential_before

    overspeed = max(ctx.velocity_m_s - ctx.speed_limit_m_s, 0.0)
    speed_fraction = overspeed / (
        1.0 + abs(ctx.velocity_m_s) + abs(ctx.speed_limit_m_s)
    )
    speed_barrier = -10.0 * ctx.dt_s * speed_fraction * speed_fraction
    energy_cost = -5.0 * ctx.step_energy_kwh

    terminal = 0.0
    if ctx.completed:
        terminal = 900.0
        if ctx.elapsed_time_s <= ctx.time_budget_s:
            terminal += 180.0
    elif ctx.episode_ended:
        remaining = max(1.0 - progress_now, 0.0)
        terminal = -900.0 - 100.0 * remaining

    reward = potential_change + speed_barrier + energy_cost + terminal
    return RewardOutput(
        reward=reward,
        components={
            "potential_change": potential_change,
            "speed_barrier": speed_barrier,
            "energy_cost": energy_cost,
            "terminal": terminal,
        },
    )
```

Sanitized Development reflection:

```json
{
  "candidate_id": "g0-c02",
  "candidate_source_sha256": "780a8f13f4416b60aa64e8ca01210eaabfa08c32af2764a2b081d16429bfeb03",
  "development_metrics": {
    "completion_rate": 1.0,
    "deadline_compliance_rate_among_completed": 1.0,
    "episode_count": 9,
    "episode_length_steps": {
      "count": 9,
      "max": 708.0,
      "mean": 566.4444444444445,
      "median": 581.0,
      "min": 467.0
    },
    "failure_mode_counts": {
      "speed": 9
    },
    "feasible_energy_kwh": {
      "count": 0,
      "max": null,
      "mean": null,
      "median": null,
      "min": null
    },
    "integrated_speed_violation_m": {
      "count": 9,
      "max": 477.34876524812313,
      "mean": 267.70644996462335,
      "median": 255.40738909068236,
      "min": 30.048091781762647
    },
    "max_speed_violation_m_s": {
      "count": 9,
      "max": 23.700059712859236,
      "mean": 17.529127905237843,
      "median": 18.043083473448156,
      "min": 8.12083795436445
    },
    "mean_integrated_speed_violation_m_among_relevant": 267.70644996462335,
    "normalized_route_progress_for_incomplete": {
      "count": 0,
      "max": null,
      "mean": null,
      "median": null,
      "min": null
    },
    "requirement_satisfaction_rate": 0.0,
    "speed_compliance_rate_among_relevant": 0.0,
    "travel_time_s": {
      "count": 9,
      "max": 70.80000000000025,
      "mean": 56.64444444444493,
      "median": 58.100000000000556,
      "min": 46.700000000000394
    }
  },
  "episode_reward_statistics": {
    "count": 85,
    "max": 1778.3950783433393,
    "mean": 1591.8223876856337,
    "min": -1282.2786059225764,
    "std": 629.2895387810222
  },
  "generation": 0,
  "information_scope": "development-only-aggregate",
  "protocol_id": "llm-reward-search-v1",
  "ranking_fields": {
    "completion_rate": 1.0,
    "deadline_compliance_among_completed": 1.0,
    "development_rsr": 0.0,
    "mean_feasible_energy_kwh": null,
    "median_incomplete_route_progress": 1.0,
    "speed_compliance_among_relevant": 0.0,
    "speed_violation_severity_among_relevant": 267.70644996462335
  },
  "reward_component_statistics": {
    "energy_cost": {
      "count": 50000,
      "max": 0.0050246816676548225,
      "mean": -0.0025612247539707486,
      "min": -0.012088082832432411,
      "std": 0.004983807753869001
    },
    "potential_change": {
      "count": 50000,
      "max": 2.5900000000001455,
      "mean": 1.135813379079333,
      "min": -0.3081158000948676,
      "std": 0.7214818074600435
    },
    "speed_barrier": {
      "count": 50000,
      "max": -0.0,
      "mean": -0.04895879025630754,
      "min": -0.5211955955851728,
      "std": 0.09158728853840759
    },
    "terminal": {
      "count": 50000,
      "max": 1080.0,
      "mean": 1.6313734630055836,
      "min": -998.7645695040278,
      "std": 44.238208429295305
    }
  },
  "schema_version": 1,
  "training_success_trajectory": {
    "completed_training_episode_count": 85,
    "development_checkpoints": [
      {
        "completion_rate": 0.4444444444444444,
        "cumulative_training_successes": 0,
        "requirement_satisfaction_rate": 0.0,
        "simulator_transitions": 10000
      },
      {
        "completion_rate": 1.0,
        "cumulative_training_successes": 0,
        "requirement_satisfaction_rate": 0.0,
        "simulator_transitions": 25000
      },
      {
        "completion_rate": 1.0,
        "cumulative_training_successes": 0,
        "requirement_satisfaction_rate": 0.0,
        "simulator_transitions": 50000
      }
    ],
    "first_success_training_step": null,
    "successful_training_episode_count": 0
  }
}
```

### Parent g0-c03

Visible rationale:
Multiplicative gating makes forward progress most valuable when it is speed-compliant and schedule-responsive, while retaining explicit reverse, energy, and terminal terms.

```python
from benchmarks.llm_reward.reward_api import RewardContext, RewardOutput


def compute_reward(ctx: RewardContext) -> RewardOutput:
    route_scale = max(ctx.route_length_m, 1.0)
    budget_scale = max(ctx.time_budget_s, 1.0)
    remaining_distance = max(ctx.route_length_m - ctx.position_m, 0.0)
    remaining_time = max(ctx.time_budget_s - ctx.elapsed_time_s, ctx.dt_s, 0.1)
    required_speed = remaining_distance / remaining_time

    overspeed = max(ctx.velocity_m_s - ctx.speed_limit_m_s, 0.0)
    overspeed_ratio = overspeed / max(ctx.speed_limit_m_s, 1.0)
    compliance_gate = 1.0 / (1.0 + overspeed_ratio ** 4)
    urgency = 1.0 + 2.0 * required_speed / (
        1.0 + required_speed + max(ctx.speed_limit_m_s, 0.0)
    )

    forward = max(ctx.delta_position_m, 0.0)
    backward = min(ctx.delta_position_m, 0.0)
    productive_progress = (
        220.0 * forward / route_scale * compliance_gate * urgency
    )
    reverse_cost = 220.0 * backward / route_scale
    violation_toll = -4.0 * ctx.dt_s * overspeed_ratio * overspeed_ratio
    energy_cost = -4.0 * ctx.step_energy_kwh

    terminal = 0.0
    if ctx.completed:
        terminal = 1100.0
        if ctx.elapsed_time_s <= ctx.time_budget_s:
            terminal += 220.0
    elif ctx.episode_ended:
        terminal = -1100.0

    reward = (
        productive_progress
        + reverse_cost
        + violation_toll
        + energy_cost
        + terminal
    )
    return RewardOutput(
        reward=reward,
        components={
            "productive_progress": productive_progress,
            "reverse_cost": reverse_cost,
            "violation_toll": violation_toll,
            "energy_cost": energy_cost,
            "terminal": terminal,
        },
    )
```

Sanitized Development reflection:

```json
{
  "candidate_id": "g0-c03",
  "candidate_source_sha256": "44446a4f75a0ad7dea840ccf1e6bf9a7c4fb811c544761f0b1d66eb4e6126dd5",
  "development_metrics": {
    "completion_rate": 1.0,
    "deadline_compliance_rate_among_completed": 1.0,
    "episode_count": 9,
    "episode_length_steps": {
      "count": 9,
      "max": 642.0,
      "mean": 458.22222222222223,
      "median": 443.0,
      "min": 381.0
    },
    "failure_mode_counts": {
      "speed": 9
    },
    "feasible_energy_kwh": {
      "count": 0,
      "max": null,
      "mean": null,
      "median": null,
      "min": null
    },
    "integrated_speed_violation_m": {
      "count": 9,
      "max": 656.5672022376858,
      "mean": 403.8312737888161,
      "median": 355.7326005167505,
      "min": 204.53456040700794
    },
    "max_speed_violation_m_s": {
      "count": 9,
      "max": 28.666666666666664,
      "mean": 23.456976373391235,
      "median": 23.804809439521655,
      "min": 17.434761924987136
    },
    "mean_integrated_speed_violation_m_among_relevant": 403.8312737888161,
    "normalized_route_progress_for_incomplete": {
      "count": 0,
      "max": null,
      "mean": null,
      "median": null,
      "min": null
    },
    "requirement_satisfaction_rate": 0.0,
    "speed_compliance_rate_among_relevant": 0.0,
    "travel_time_s": {
      "count": 9,
      "max": 64.20000000000063,
      "mean": 45.8222222222226,
      "median": 44.30000000000036,
      "min": 38.10000000000027
    }
  },
  "episode_reward_statistics": {
    "count": 74,
    "max": 1656.8720356248086,
    "mean": 1038.1202571544536,
    "min": -1099.140027159269,
    "std": 801.0470562175468
  },
  "generation": 0,
  "information_scope": "development-only-aggregate",
  "protocol_id": "llm-reward-search-v1",
  "ranking_fields": {
    "completion_rate": 1.0,
    "deadline_compliance_among_completed": 1.0,
    "development_rsr": 0.0,
    "mean_feasible_energy_kwh": null,
    "median_incomplete_route_progress": 1.0,
    "speed_compliance_among_relevant": 0.0,
    "speed_violation_severity_among_relevant": 403.8312737888161
  },
  "reward_component_statistics": {
    "energy_cost": {
      "count": 50000,
      "max": 0.004030481294313306,
      "mean": -0.0022873442377524087,
      "min": -0.009610955350040511,
      "std": 0.003437288523889149
    },
    "productive_progress": {
      "count": 50000,
      "max": 1.215389878170962,
      "mean": 0.31354747104124725,
      "min": 0.0,
      "std": 0.2831841212588981
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
      "max": 1320.0,
      "mean": 1.6016,
      "min": -1100.0,
      "std": 49.69614549882113
    },
    "violation_toll": {
      "count": 50000,
      "max": -0.0,
      "mean": -0.37605303118381955,
      "min": -12.814240000000002,
      "std": 1.4081162356185901
    }
  },
  "schema_version": 1,
  "training_success_trajectory": {
    "completed_training_episode_count": 74,
    "development_checkpoints": [
      {
        "completion_rate": 0.6666666666666666,
        "cumulative_training_successes": 0,
        "requirement_satisfaction_rate": 0.0,
        "simulator_transitions": 10000
      },
      {
        "completion_rate": 0.7777777777777778,
        "cumulative_training_successes": 0,
        "requirement_satisfaction_rate": 0.0,
        "simulator_transitions": 25000
      },
      {
        "completion_rate": 1.0,
        "cumulative_training_successes": 0,
        "requirement_satisfaction_rate": 0.0,
        "simulator_transitions": 50000
      }
    ],
    "first_success_training_step": null,
    "successful_training_episode_count": 0
  }
}
```
