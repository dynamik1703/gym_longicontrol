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
    potential_now = 700.0 * progress_now - 240.0 * lag_now * lag_now
    potential_before = 700.0 * progress_before - 240.0 * lag_before * lag_before
    potential_change = potential_now - potential_before

    current_excess = max(ctx.velocity_m_s - ctx.speed_limit_m_s, 0.0)
    margin_excess = max(
        ctx.velocity_m_s - 0.97 * max(ctx.speed_limit_m_s, 0.0), 0.0
    )
    current_speed_cost = -ctx.dt_s * (
        20.0 * current_excess * current_excess
        + 3.0 * margin_excess * margin_excess
    )

    preview_ceiling = max(ctx.speed_limit_m_s, 0.0)
    if ctx.future_speed_limit_1_m_s < preview_ceiling:
        preview_ceiling = min(
            preview_ceiling,
            max(ctx.future_speed_limit_1_m_s, 0.0)
            + max(ctx.distance_to_future_limit_1_m, 0.0) / 6.0,
        )
    if ctx.future_speed_limit_2_m_s < preview_ceiling:
        preview_ceiling = min(
            preview_ceiling,
            max(ctx.future_speed_limit_2_m_s, 0.0)
            + max(ctx.distance_to_future_limit_2_m, 0.0) / 10.0,
        )
    preview_excess = max(ctx.velocity_m_s - preview_ceiling, 0.0)
    preview_speed_cost = -4.0 * ctx.dt_s * preview_excess * preview_excess

    energy_cost = -3.0 * ctx.step_energy_kwh

    terminal = 0.0
    if ctx.completed:
        terminal = 1100.0
        if ctx.elapsed_time_s <= ctx.time_budget_s:
            terminal += 220.0
    elif ctx.episode_ended:
        terminal = -1100.0 - 100.0 * max(1.0 - progress_now, 0.0)

    reward = (
        potential_change
        + current_speed_cost
        + preview_speed_cost
        + energy_cost
        + terminal
    )
    return RewardOutput(
        reward=reward,
        components={
            "potential_change": potential_change,
            "current_speed_cost": current_speed_cost,
            "preview_speed_cost": preview_speed_cost,
            "energy_cost": energy_cost,
            "terminal": terminal,
        },
    )
