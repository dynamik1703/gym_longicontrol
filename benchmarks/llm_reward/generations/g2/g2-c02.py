from benchmarks.llm_reward.reward_api import RewardContext, RewardOutput


def compute_reward(ctx: RewardContext) -> RewardOutput:
    route_scale = max(ctx.route_length_m, 1.0)
    time_scale = max(ctx.time_budget_s, ctx.dt_s, 0.1)
    current_limit = max(ctx.speed_limit_m_s, 0.0)

    preview_ceiling = 0.99 * current_limit
    first_limit = max(ctx.future_speed_limit_1_m_s, 0.0)
    second_limit = max(ctx.future_speed_limit_2_m_s, 0.0)
    if first_limit < current_limit:
        preview_ceiling = min(
            preview_ceiling,
            0.98 * first_limit
            + max(ctx.distance_to_future_limit_1_m, 0.0) / 7.5,
        )
    if second_limit < current_limit:
        preview_ceiling = min(
            preview_ceiling,
            0.98 * second_limit
            + max(ctx.distance_to_future_limit_2_m, 0.0) / 12.0,
        )

    current_excess = max(ctx.velocity_m_s - current_limit, 0.0)
    preview_excess = max(ctx.velocity_m_s - preview_ceiling, 0.0)
    limit_scale = max(current_limit, 1.0)
    current_ratio = current_excess / limit_scale
    preview_ratio = preview_excess / max(preview_ceiling, 1.0)
    compliance_gate = 1.0 / (
        1.0 + (7.0 * current_ratio) ** 4 + (2.5 * preview_ratio) ** 4
    )

    forward = max(ctx.delta_position_m, 0.0)
    backward = min(ctx.delta_position_m, 0.0)
    compliant_progress = 900.0 * forward / route_scale * compliance_gate
    reverse_cost = 900.0 * backward / route_scale

    progress_fraction = min(max(ctx.position_m / route_scale, 0.0), 1.0)
    previous_progress_fraction = min(
        max(ctx.previous_position_m / route_scale, 0.0), 1.0
    )
    time_fraction = min(max(ctx.elapsed_time_s / time_scale, 0.0), 2.0)
    previous_time_fraction = min(
        max((ctx.elapsed_time_s - ctx.dt_s) / time_scale, 0.0), 2.0
    )
    schedule_debt = max(time_fraction - progress_fraction, 0.0)
    previous_schedule_debt = max(
        previous_time_fraction - previous_progress_fraction, 0.0
    )
    raw_schedule_recovery = 600.0 * (
        previous_schedule_debt - schedule_debt
    )
    if raw_schedule_recovery > 0.0:
        schedule_recovery = raw_schedule_recovery * compliance_gate
    else:
        schedule_recovery = raw_schedule_recovery

    speed_guard = -ctx.dt_s * (
        48.0 * current_excess * current_excess
        + 7.0 * preview_excess * preview_excess
    )
    energy_cost = -2.5 * ctx.step_energy_kwh

    terminal = 0.0
    if ctx.completed:
        terminal = 1550.0
        if ctx.elapsed_time_s <= ctx.time_budget_s:
            terminal += 500.0
    elif ctx.episode_ended:
        terminal = -1550.0 - 220.0 * max(1.0 - progress_fraction, 0.0)

    reward = (
        compliant_progress
        + reverse_cost
        + schedule_recovery
        + speed_guard
        + energy_cost
        + terminal
    )
    return RewardOutput(
        reward=reward,
        components={
            "compliant_progress": compliant_progress,
            "reverse_cost": reverse_cost,
            "schedule_recovery": schedule_recovery,
            "speed_guard": speed_guard,
            "energy_cost": energy_cost,
            "terminal": terminal,
        },
    )
