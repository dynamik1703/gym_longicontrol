from benchmarks.llm_reward.reward_api import RewardContext, RewardOutput


def compute_reward(ctx: RewardContext) -> RewardOutput:
    route_scale = max(ctx.route_length_m, 1.0)
    budget_scale = max(ctx.time_budget_s, 1.0)
    remaining_distance = max(ctx.route_length_m - ctx.position_m, 0.0)
    remaining_time = max(ctx.time_budget_s - ctx.elapsed_time_s, ctx.dt_s, 0.1)
    required_speed = remaining_distance / remaining_time

    overspeed = max(ctx.velocity_m_s - ctx.speed_limit_m_s, 0.0)
    overspeed_ratio = overspeed / max(ctx.speed_limit_m_s, 1.0)
    compliance_gate = 1.0 / (1.0 + overspeed_ratio ** 4)
    urgency = 1.0 + 2.0 * required_speed / (
        1.0 + required_speed + max(ctx.speed_limit_m_s, 0.0)
    )

    forward = max(ctx.delta_position_m, 0.0)
    backward = min(ctx.delta_position_m, 0.0)
    productive_progress = (
        220.0 * forward / route_scale * compliance_gate * urgency
    )
    reverse_cost = 220.0 * backward / route_scale
    violation_toll = -4.0 * ctx.dt_s * overspeed_ratio * overspeed_ratio
    energy_cost = -4.0 * ctx.step_energy_kwh

    terminal = 0.0
    if ctx.completed:
        terminal = 1100.0
        if ctx.elapsed_time_s <= ctx.time_budget_s:
            terminal += 220.0
    elif ctx.episode_ended:
        terminal = -1100.0

    reward = (
        productive_progress
        + reverse_cost
        + violation_toll
        + energy_cost
        + terminal
    )
    return RewardOutput(
        reward=reward,
        components={
            "productive_progress": productive_progress,
            "reverse_cost": reverse_cost,
            "violation_toll": violation_toll,
            "energy_cost": energy_cost,
            "terminal": terminal,
        },
    )
