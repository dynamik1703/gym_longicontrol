from benchmarks.llm_reward.reward_api import RewardContext, RewardOutput


def compute_reward(ctx: RewardContext) -> RewardOutput:
    route_scale = max(ctx.route_length_m, 1.0)
    budget_scale = max(ctx.time_budget_s, 1.0)
    speed_scale = max(ctx.speed_limit_m_s, 1.0)

    progress = 100.0 * ctx.delta_position_m / route_scale
    time_cost = -ctx.dt_s / budget_scale
    overspeed = max(ctx.velocity_m_s - ctx.speed_limit_m_s, 0.0)
    speed_cost = -8.0 * ctx.dt_s * (overspeed / speed_scale) ** 2
    energy_cost = -6.0 * ctx.step_energy_kwh

    completion = 0.0
    timeliness = 0.0
    terminal_failure = 0.0
    if ctx.completed:
        completion = 1000.0
        if ctx.elapsed_time_s <= ctx.time_budget_s:
            timeliness = 200.0
        else:
            lateness = ctx.elapsed_time_s - ctx.time_budget_s
            timeliness = -200.0 * lateness / (budget_scale + lateness)
    elif ctx.episode_ended:
        terminal_failure = -1000.0

    reward = (
        progress
        + time_cost
        + speed_cost
        + energy_cost
        + completion
        + timeliness
        + terminal_failure
    )
    return RewardOutput(
        reward=reward,
        components={
            "progress": progress,
            "time_cost": time_cost,
            "speed_cost": speed_cost,
            "energy_cost": energy_cost,
            "completion": completion,
            "timeliness": timeliness,
            "terminal_failure": terminal_failure,
        },
    )
