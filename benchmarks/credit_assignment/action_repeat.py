"""Benchmark-only action repeat with explicit physical-step accounting."""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real
from typing import Any

import gymnasium as gym


def _positive_integer(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _acceleration_sign(value: float) -> int:
    return 1 if value > 1e-9 else -1 if value < -1e-9 else 0


class ActionRepeat(gym.Wrapper):
    """Hold one action while the wrapped environment advances at its native dt.

    The wrapper sums scalar rewards, returns the final observation and propagates
    a natural termination or truncation immediately.  ``max_simulator_steps`` is
    used only to stop a training run at an exact physical interaction budget.
    Episode metrics remain owned by the wrapped environment and are therefore
    updated once per native simulator transition.
    """

    def __init__(
        self,
        env: gym.Env,
        repeat: int,
        *,
        max_simulator_steps: int | None = None,
        capture_trace: bool = False,
    ):
        super().__init__(env)
        self.repeat = _positive_integer("repeat", repeat)
        self.max_simulator_steps = (
            None
            if max_simulator_steps is None
            else _positive_integer("max_simulator_steps", max_simulator_steps)
        )
        self.capture_trace = bool(capture_trace)
        self.simulator_steps = 0
        self.agent_decisions = 0
        self.episode_simulator_steps = 0
        self.episode_agent_decisions = 0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.episode_simulator_steps = 0
        self.episode_agent_decisions = 0
        return observation, info

    def step(self, action):
        if (
            self.max_simulator_steps is not None
            and self.simulator_steps >= self.max_simulator_steps
        ):
            raise RuntimeError("The simulator interaction budget is exhausted")

        accumulated_reward = 0.0
        reward_components: dict[str, float] = {}
        historical_reward = 0.0
        trace: list[dict[str, float]] = []
        energy_sum = 0.0
        traction_energy_sum = 0.0
        regenerative_energy_sum = 0.0
        absolute_jerk_sum = 0.0
        max_abs_jerk = 0.0
        absolute_acceleration_sum = 0.0
        nonzero_acceleration_signs: list[int] = []
        observation = None
        final_info: dict[str, Any] = {}
        terminated = truncated = False
        executed = 0

        for _ in range(self.repeat):
            if (
                self.max_simulator_steps is not None
                and self.simulator_steps >= self.max_simulator_steps
            ):
                break
            observation, reward, terminated, truncated, info = self.env.step(action)
            if isinstance(reward, bool) or not isinstance(reward, Real):
                raise TypeError("ActionRepeat requires a scalar reward")
            reward_value = float(reward)
            if not isfinite(reward_value):
                raise ValueError("ActionRepeat requires finite rewards")
            accumulated_reward += reward_value
            executed += 1
            self.simulator_steps += 1
            self.episode_simulator_steps += 1
            final_info = dict(info)

            step_energy = float(info.get("step_energy_kwh", 0.0))
            jerk = abs(float(info.get("jerk_m_s3", 0.0)))
            acceleration = float(info.get("acceleration_m_s2", 0.0))
            energy_sum += step_energy
            traction_energy_sum += max(step_energy, 0.0)
            regenerative_energy_sum += max(-step_energy, 0.0)
            absolute_jerk_sum += jerk
            max_abs_jerk = max(max_abs_jerk, jerk)
            absolute_acceleration_sum += abs(acceleration)
            sign = _acceleration_sign(acceleration)
            if sign:
                nonzero_acceleration_signs.append(sign)

            components = info.get("benchmark_reward_components")
            if isinstance(components, dict):
                for name, value in components.items():
                    reward_components[name] = reward_components.get(name, 0.0) + float(
                        value
                    )
            if "historical_reward" in info:
                historical_reward += float(info["historical_reward"])
            if self.capture_trace:
                trace.append(
                    {
                        "time_s": float(info["elapsed_time_s"]),
                        "position_m": float(info["position_m"]),
                        "velocity_m_s": float(info["velocity_m_s"]),
                        "speed_limit_m_s": float(info["speed_limit_m_s"]),
                        "acceleration_m_s2": acceleration,
                        "jerk_m_s3": float(info["jerk_m_s3"]),
                        "action": float(action[0]),
                        "cumulative_energy_kwh": float(
                            info["episode_metrics"]["energy_kwh"]
                        ),
                    }
                )
            if terminated or truncated:
                break

        if observation is None or executed == 0:
            raise RuntimeError("ActionRepeat did not execute a simulator transition")

        self.agent_decisions += 1
        self.episode_agent_decisions += 1
        budget_reached = bool(
            self.max_simulator_steps is not None
            and self.simulator_steps == self.max_simulator_steps
        )
        if budget_reached and not (terminated or truncated):
            truncated = True
            final_info["simulator_budget_truncated"] = True

        sign_changes = sum(
            left != right
            for left, right in zip(
                nonzero_acceleration_signs, nonzero_acceleration_signs[1:]
            )
        )
        final_info.update(
            {
                "action_repeat": self.repeat,
                "simulator_steps_this_decision": executed,
                "simulator_steps_total": self.simulator_steps,
                "episode_simulator_steps": self.episode_simulator_steps,
                "agent_decisions_total": self.agent_decisions,
                "episode_agent_decisions": self.episode_agent_decisions,
                "last_simulator_step_energy_kwh": float(
                    final_info.get("step_energy_kwh", 0.0)
                ),
                "step_energy_kwh": energy_sum,
                "action_repeat_diagnostics": {
                    "traction_energy_kwh": traction_energy_sum,
                    "regenerative_energy_kwh": regenerative_energy_sum,
                    "absolute_jerk_sum": absolute_jerk_sum,
                    "max_abs_jerk_m_s3": max_abs_jerk,
                    "absolute_acceleration_sum": absolute_acceleration_sum,
                    "first_nonzero_acceleration_sign": (
                        nonzero_acceleration_signs[0]
                        if nonzero_acceleration_signs
                        else 0
                    ),
                    "last_nonzero_acceleration_sign": (
                        nonzero_acceleration_signs[-1]
                        if nonzero_acceleration_signs
                        else 0
                    ),
                    "acceleration_sign_change_count": sign_changes,
                },
            }
        )
        if reward_components:
            final_info["benchmark_reward_components"] = reward_components
        if "historical_reward" in final_info:
            final_info["historical_reward"] = historical_reward
        if self.capture_trace:
            final_info["action_repeat_trace"] = trace
        return (
            observation,
            accumulated_reward,
            bool(terminated),
            bool(truncated),
            final_info,
        )
