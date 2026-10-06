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
