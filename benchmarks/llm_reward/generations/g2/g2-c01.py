from benchmarks.llm_reward.reward_api import RewardContext, RewardOutput


def compute_reward(ctx: RewardContext) -> RewardOutput:
    route_scale = max(ctx.route_length_m, 1.0)
    time_scale = max(ctx.time_budget_s, ctx.dt_s, 0.1)
    remaining_distance = max(ctx.route_length_m - ctx.position_m, 0.0)
    remaining_time = max(ctx.time_budget_s - ctx.elapsed_time_s, ctx.dt_s, 0.1)
    required_speed = remaining_distance / remaining_time

    current_limit = max(ctx.speed_limit_m_s, 0.0)
    safe_ceiling = 0.985 * current_limit
    first_limit = max(ctx.future_speed_limit_1_m_s, 0.0)
    second_limit = max(ctx.future_speed_limit_2_m_s, 0.0)
    if first_limit < current_limit:
        first_distance = max(ctx.distance_to_future_limit_1_m, 0.0)
        first_ceiling = (
            (0.98 * first_limit) ** 2 + 4.0 * first_distance
        ) ** 0.5
        safe_ceiling = min(safe_ceiling, first_ceiling)
    if second_limit < current_limit:
        second_distance = max(ctx.distance_to_future_limit_2_m, 0.0)
        second_ceiling = (
            (0.98 * second_limit) ** 2 + 3.0 * second_distance
        ) ** 0.5
        safe_ceiling = min(safe_ceiling, second_ceiling)

    target_speed = min(1.08 * required_speed, max(safe_ceiling, 0.0))
    pace_shortfall = max(target_speed - ctx.velocity_m_s, 0.0)
    pace_support = -0.22 * ctx.dt_s * pace_shortfall * pace_shortfall

    current_excess = max(ctx.velocity_m_s - current_limit, 0.0)
    preview_excess = max(ctx.velocity_m_s - safe_ceiling, 0.0)
    speed_guard = -ctx.dt_s * (
        45.0 * current_excess * current_excess
        + 6.0 * preview_excess * preview_excess
    )

    limit_scale = max(current_limit, 1.0)
    current_ratio = current_excess / limit_scale
    compliance_gate = 1.0 / (1.0 + (7.0 * current_ratio) ** 4)
    forward = max(ctx.delta_position_m, 0.0)
    backward = min(ctx.delta_position_m, 0.0)
    route_progress = 850.0 * forward / route_scale * compliance_gate
    reverse_cost = 850.0 * backward / route_scale

    progress_fraction = min(max(ctx.position_m / route_scale, 0.0), 1.0)
    time_fraction = max(ctx.elapsed_time_s / time_scale, 0.0)
    schedule_lag = max(time_fraction - progress_fraction, 0.0)
    schedule_cost = -1.2 * ctx.dt_s * min(schedule_lag, 1.5)
    energy_cost = -3.0 * ctx.step_energy_kwh

    terminal = 0.0
    if ctx.completed:
        terminal = 1500.0
        if ctx.elapsed_time_s <= ctx.time_budget_s:
            terminal += 450.0
    elif ctx.episode_ended:
        terminal = -1500.0 - 200.0 * max(1.0 - progress_fraction, 0.0)

    reward = (
        route_progress
        + reverse_cost
        + pace_support
        + speed_guard
        + schedule_cost
        + energy_cost
        + terminal
    )
    return RewardOutput(
        reward=reward,
        components={
            "route_progress": route_progress,
            "reverse_cost": reverse_cost,
            "pace_support": pace_support,
            "speed_guard": speed_guard,
            "schedule_cost": schedule_cost,
            "energy_cost": energy_cost,
            "terminal": terminal,
        },
    )
