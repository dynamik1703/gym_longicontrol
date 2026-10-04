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

