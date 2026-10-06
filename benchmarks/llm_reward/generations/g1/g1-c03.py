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
