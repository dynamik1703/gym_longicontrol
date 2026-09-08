"""Thin Gymnasium lifecycle around the simulation components."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.observation import observation
from gym_longicontrol.domain.reward import reward_components
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.track import FixedTrackGenerator, StochasticTrackGenerator
from gym_longicontrol.domain.vehicle import VehicleModel


class LongiControlEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        car_id="BMW_electric_i3_2014",
        reward_weights=(1, 0.5, 1, 1),
        energy_factor=1.0,
        render_mode=None,
        *,
        config=None,
        track_generator=None,
        vehicle=None,
    ):
        self.config = config if config is not None else SimulationConfig()
        self.vehicle = vehicle if vehicle is not None else VehicleModel(car_id)
        self.track_generator = (
            track_generator
            if track_generator is not None
            else FixedTrackGenerator(track_length_m=self.config.track_length_m)
        )
        self.reward_weights = np.array(reward_weights, dtype=np.float64, copy=True)
        if (
            self.reward_weights.shape != (4,)
            or not np.isfinite(self.reward_weights).all()
        ):
            raise ValueError("reward_weights must contain four finite values")
        self.reward_weights.setflags(write=False)
        if not np.isfinite(energy_factor) or not 0 <= energy_factor <= 5:
            raise ValueError("energy_factor must be in [0, 5]")
        self.energy_factor = float(energy_factor)
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render_mode {render_mode!r}")
        self.render_mode = render_mode
        self.metadata = {
            **type(self).metadata,
            "render_fps": max(1, round(1 / self.config.dt_s)),
        }
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float64)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(8,), dtype=np.float64)
        self.state = VehicleState()
        self.track = None
        self._renderer = None
        self._terminated = False

    def _observation(self):
        sensor = self.track.sense(self.state.position_m, self.config.sensor_range_m)
        return observation(
            self.state, sensor, self.vehicle.specs, self.config, self.energy_factor
        )

    def _info(self, power_kw=0.0, step_energy_kwh=0.0, components=None):
        state = self.state
        limit = self.track.sense(
            state.position_m, self.config.sensor_range_m
        ).current_limit_m_s
        return {
            "position_m": state.position_m,
            "velocity_m_s": state.velocity_m_s,
            "velocity_km_h": state.velocity_m_s * 3.6,
            "acceleration_m_s2": state.acceleration_m_s2,
            "jerk_m_s3": state.jerk_m_s3,
            "elapsed_time_s": state.elapsed_time_s,
            "total_energy_kwh": state.total_energy_kwh,
            "power_kw": power_kw,
            "step_energy_kwh": step_energy_kwh,
            "speed_limit_m_s": limit,
            "speed_limit_km_h": limit * 3.6,
            "reward_components": (
                dict(components)
                if components is not None
                else dict.fromkeys(("forward", "energy", "jerk", "shock"), 0.0)
            ),
            # Transitional aliases retain the original units.
            "position": state.position_m,
            "velocity": state.velocity_m_s * 3.6,
            "acceleration": state.acceleration_m_s2,
            "jerk": state.jerk_m_s3,
            "time": state.elapsed_time_s,
            "energy": state.total_energy_kwh,
            "speed_limit": limit * 3.6,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError("No reset options are currently supported")
        self.track = self.track_generator(self.np_random)
        self.state = VehicleState()
        self._terminated = False
        if self._renderer is not None:
            self._renderer.reset()
        if self.render_mode == "human":
            self.render()
        return self._observation(), self._info()

    def step(self, action):
        if self.track is None or self._terminated:
            raise gym.error.ResetNeeded("Call reset() before stepping a new episode")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (1,) or not np.isfinite(action).all():
            raise ValueError("action must be a finite one-element array")
        # Preserve the original clipping policy.
        action = float(np.clip(action[0], -1, 1))
        self.state, power, energy = advance(
            self.state, action, self.vehicle, self.config
        )
        sensor = self.track.sense(self.state.position_m, self.config.sensor_range_m)
        components = reward_components(
            self.state, sensor, power, self.vehicle.specs, self.config
        )
        reward = float(np.dot(self.reward_weights, list(components.values())))
        self._terminated = self.state.position_m >= self.config.track_length_m
        if self.render_mode == "human":
            self.render()
        return (
            self._observation(),
            reward,
            bool(self._terminated),
            False,
            self._info(power, energy, components),
        )

    def render(self):
        if self.render_mode is None:
            return None
        if self.track is None:
            raise gym.error.ResetNeeded("Call reset() before rendering")
        if self._renderer is None:
            from gym_longicontrol.rendering import Renderer

            self._renderer = Renderer(self.render_mode, self.metadata["render_fps"])
        return self._renderer.render(self.state, self.track, self.config)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


class DeterministicTrack(LongiControlEnv):
    def __init__(
        self,
        car_id="BMW_electric_i3_2014",
        speed_limit_positions=(0.0, 0.25, 0.5, 0.75),
        speed_limits=(50, 80, 40, 50),
        reward_weights=(1, 0.5, 1, 1),
        energy_factor=1.0,
        render_mode=None,
        *,
        config=None,
        vehicle=None,
    ):
        config = config if config is not None else SimulationConfig()
        generator = FixedTrackGenerator(
            speed_limit_positions, speed_limits, config.track_length_m
        )
        super().__init__(
            car_id,
            reward_weights,
            energy_factor,
            render_mode,
            config=config,
            track_generator=generator,
            vehicle=vehicle,
        )


class StochasticTrack(LongiControlEnv):
    def __init__(
        self,
        car_id="BMW_electric_i3_2014",
        reward_weights=(1, 0.5, 1, 1),
        energy_factor=1.0,
        render_mode=None,
        *,
        config=None,
        vehicle=None,
    ):
        config = config if config is not None else SimulationConfig()
        super().__init__(
            car_id,
            reward_weights,
            energy_factor,
            render_mode,
            config=config,
            vehicle=vehicle,
            track_generator=StochasticTrackGenerator(config.track_length_m),
        )
