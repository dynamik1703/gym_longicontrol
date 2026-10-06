from benchmarks.llm_reward.reward_api import RewardContext, RewardOutput


def compute_reward(ctx: RewardContext) -> RewardOutput:
    route_scale = max(ctx.route_length_m, 1.0)
    time_scale = max(ctx.time_budget_s, ctx.dt_s, 0.1)
    current_limit = max(ctx.speed_limit_m_s, 0.0)
    speed_scale_sq = max(current_limit * current_limit, 1.0)

    current_excess = max(ctx.velocity_m_s - current_limit, 0.0)
    current_ratio = current_excess / max(current_limit, 1.0)
    compliance_gate = 1.0 / (1.0 + (8.0 * current_ratio) ** 4)

    kinetic_excess = 0.0
    first_limit = max(ctx.future_speed_limit_1_m_s, 0.0)
    if first_limit < current_limit:
        first_distance = max(ctx.distance_to_future_limit_1_m, 0.0)
        first_safe_speed_sq = (0.985 * first_limit) ** 2 + 4.4 * first_distance
        first_kinetic_excess = max(
            ctx.velocity_m_s * ctx.velocity_m_s - first_safe_speed_sq, 0.0
        ) / speed_scale_sq
        kinetic_excess = max(kinetic_excess, first_kinetic_excess)
    second_limit = max(ctx.future_speed_limit_2_m_s, 0.0)
    if second_limit < current_limit:
        second_distance = max(ctx.distance_to_future_limit_2_m, 0.0)
        second_safe_speed_sq = (0.985 * second_limit) ** 2 + 3.2 * second_distance
        second_kinetic_excess = max(
            ctx.velocity_m_s * ctx.velocity_m_s - second_safe_speed_sq, 0.0
        ) / speed_scale_sq
        kinetic_excess = max(kinetic_excess, second_kinetic_excess)

    current_speed_guard = -ctx.dt_s * (
        50.0 * current_excess * current_excess
        + 5.0 * current_excess ** 4 / (1.0 + current_excess * current_excess)
    )
    braking_guard = -14.0 * ctx.dt_s * kinetic_excess * kinetic_excess
    positive_acceleration = max(ctx.acceleration_m_s2, 0.0)
    unsafe_acceleration = (
        -0.25
        * ctx.dt_s
        * min(kinetic_excess, 2.0)
        * positive_acceleration
        * positive_acceleration
    )

    forward = max(ctx.delta_position_m, 0.0)
    backward = min(ctx.delta_position_m, 0.0)
    route_progress = 950.0 * forward / route_scale * compliance_gate
    reverse_cost = 950.0 * backward / route_scale

    progress_fraction = min(max(ctx.position_m / route_scale, 0.0), 1.0)
    time_fraction = max(ctx.elapsed_time_s / time_scale, 0.0)
    late_fraction = min(max(time_fraction - 1.0, 0.0), 1.0)
    time_cost = -0.24 * ctx.dt_s * (1.0 + 3.0 * late_fraction)

    if progress_fraction >= min(time_fraction, 1.0):
        energy_weight = 5.0
    else:
        energy_weight = 1.5
    energy_cost = -energy_weight * ctx.step_energy_kwh

    terminal = 0.0
    if ctx.completed:
        terminal = 1600.0
        if ctx.elapsed_time_s <= ctx.time_budget_s:
            terminal += 550.0
    elif ctx.episode_ended:
        terminal = -1600.0 - 250.0 * max(1.0 - progress_fraction, 0.0)

    reward = (
        route_progress
        + reverse_cost
        + current_speed_guard
        + braking_guard
        + unsafe_acceleration
        + time_cost
        + energy_cost
        + terminal
    )
    return RewardOutput(
        reward=reward,
        components={
            "route_progress": route_progress,
            "reverse_cost": reverse_cost,
            "current_speed_guard": current_speed_guard,
            "braking_guard": braking_guard,
            "unsafe_acceleration": unsafe_acceleration,
            "time_cost": time_cost,
            "energy_cost": energy_cost,
            "terminal": terminal,
        },
    )
