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
