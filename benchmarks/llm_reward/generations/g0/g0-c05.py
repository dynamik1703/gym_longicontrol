import math

from benchmarks.llm_reward.reward_api import RewardContext, RewardOutput


def compute_reward(ctx: RewardContext) -> RewardOutput:
    route_scale = max(ctx.route_length_m, 1.0)
    braking_rate = 1.5
    preview_cap_1 = math.sqrt(
        max(
            0.0,
            ctx.future_speed_limit_1_m_s ** 2
            + 2.0 * braking_rate * max(ctx.distance_to_future_limit_1_m, 0.0),
        )
    )
    preview_cap_2 = math.sqrt(
        max(
            0.0,
            ctx.future_speed_limit_2_m_s ** 2
            + 2.0 * braking_rate * max(ctx.distance_to_future_limit_2_m, 0.0),
        )
    )
    legal_envelope = max(
        min(ctx.speed_limit_m_s, preview_cap_1, preview_cap_2), 0.0
    )

    remaining_distance = max(ctx.route_length_m - ctx.position_m, 0.0)
    remaining_time = max(ctx.time_budget_s - ctx.elapsed_time_s, ctx.dt_s, 0.1)
    schedule_speed = remaining_distance / remaining_time
    cruise_floor = 0.65 * max(ctx.speed_limit_m_s, 0.0)
    target_speed = min(legal_envelope, max(schedule_speed, cruise_floor))

    tracking_error = ctx.velocity_m_s - target_speed
    error_scale = max(target_speed, 1.0)
    normalized_error = tracking_error / error_scale
    absolute_error = abs(normalized_error)
    if absolute_error <= 0.2:
        tracking_cost = -2.0 * ctx.dt_s * 0.5 * normalized_error ** 2
    else:
        tracking_cost = -2.0 * ctx.dt_s * (0.2 * absolute_error - 0.02)

    progress = 180.0 * ctx.delta_position_m / route_scale
    current_overspeed = max(ctx.velocity_m_s - ctx.speed_limit_m_s, 0.0)
    legal_cost = -12.0 * ctx.dt_s * (
        current_overspeed / max(ctx.speed_limit_m_s, 1.0)
    ) ** 2
    energy_cost = -5.0 * ctx.step_energy_kwh

    terminal = 0.0
    if ctx.completed:
        terminal = 1050.0
        if ctx.elapsed_time_s <= ctx.time_budget_s:
            terminal += 210.0
    elif ctx.episode_ended:
        terminal = -1050.0

    reward = progress + tracking_cost + legal_cost + energy_cost + terminal
    return RewardOutput(
        reward=reward,
        components={
            "progress": progress,
            "tracking_cost": tracking_cost,
            "legal_cost": legal_cost,
            "energy_cost": energy_cost,
            "terminal": terminal,
        },
    )
