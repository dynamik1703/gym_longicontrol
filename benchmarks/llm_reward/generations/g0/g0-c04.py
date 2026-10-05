from benchmarks.llm_reward.reward_api import RewardContext, RewardOutput


def compute_reward(ctx: RewardContext) -> RewardOutput:
    route_scale = max(ctx.route_length_m, 1.0)
    budget_scale = max(ctx.time_budget_s, 1.0)

    progress = 160.0 * ctx.delta_position_m / route_scale
    expected_position = ctx.route_length_m * min(
        max(ctx.elapsed_time_s / budget_scale, 0.0), 1.0
    )
    lag_fraction = max(expected_position - ctx.position_m, 0.0) / route_scale
    if lag_fraction <= 0.02:
        schedule_cost = 0.0
    elif lag_fraction <= 0.10:
        schedule_cost = -0.5 * ctx.dt_s
    else:
        schedule_cost = -2.0 * ctx.dt_s

    overspeed = ctx.velocity_m_s - ctx.speed_limit_m_s
    if overspeed <= 0.0:
        speed_cost = 0.0
    elif overspeed <= 0.5:
        speed_cost = -ctx.dt_s * overspeed
    elif overspeed <= 2.0:
        speed_cost = -ctx.dt_s * (1.0 + 3.0 * (overspeed - 0.5))
    else:
        speed_cost = -ctx.dt_s * (8.0 + 8.0 * (overspeed - 2.0))

    energy_fraction = ctx.step_energy_kwh / (0.01 + abs(ctx.step_energy_kwh))
    bounded_energy = -0.05 * energy_fraction

    completion = 0.0
    on_time = 0.0
    failure = 0.0
    if ctx.completed:
        completion = 1200.0
        if ctx.elapsed_time_s <= ctx.time_budget_s:
            on_time = 240.0
    elif ctx.episode_ended:
        failure = -1200.0

    reward = (
        progress
        + schedule_cost
        + speed_cost
        + bounded_energy
        + completion
        + on_time
        + failure
    )
    return RewardOutput(
        reward=reward,
        components={
            "progress": progress,
            "schedule_cost": schedule_cost,
            "speed_cost": speed_cost,
            "bounded_energy": bounded_energy,
            "completion": completion,
            "on_time": on_time,
            "failure": failure,
        },
    )
