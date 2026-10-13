"""Scalar and vectorized exact LongiControl physics rollouts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import ceil

import numpy as np

from benchmarks.model_based_rl.adapter import model_step
from benchmarks.model_based_rl.physics_model import PhysicsDynamicsModel
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.vehicle import VehicleModel

from .adapter import ModelState, TrackContext


@dataclass
class BatchedPhysicsState:
    position_m: np.ndarray
    velocity_m_s: np.ndarray
    acceleration_m_s2: np.ndarray
    jerk_m_s3: np.ndarray
    elapsed_time_s: np.ndarray
    total_energy_kwh: np.ndarray
    maximum_speed_excess_m_s: np.ndarray
    speed_violation_count: np.ndarray
    violation_active: np.ndarray
    integrated_speed_violation_m: np.ndarray
    episode_step: np.ndarray

    def __len__(self) -> int:
        return int(self.position_m.shape[0])

    def take(self, indices: np.ndarray) -> BatchedPhysicsState:
        return BatchedPhysicsState(
            **{
                name: np.asarray(getattr(self, name))[indices].copy()
                for name in self.__dataclass_fields__
            }
        )

    def to_model_states(self) -> list[ModelState]:
        states = []
        for index in range(len(self)):
            states.append(
                ModelState(
                    vehicle=VehicleState(
                        position_m=float(self.position_m[index]),
                        velocity_m_s=float(self.velocity_m_s[index]),
                        acceleration_m_s2=float(self.acceleration_m_s2[index]),
                        jerk_m_s3=float(self.jerk_m_s3[index]),
                        elapsed_time_s=float(self.elapsed_time_s[index]),
                        total_energy_kwh=float(self.total_energy_kwh[index]),
                    ),
                    maximum_speed_excess_m_s=float(
                        self.maximum_speed_excess_m_s[index]
                    ),
                    speed_violation_count=int(self.speed_violation_count[index]),
                    violation_active=bool(self.violation_active[index]),
                    integrated_speed_violation_m=float(
                        self.integrated_speed_violation_m[index]
                    ),
                    episode_step=int(self.episode_step[index]),
                )
            )
        return states


@dataclass(frozen=True)
class PhysicsStepBatch:
    state: BatchedPhysicsState
    observation: np.ndarray
    next_observation: np.ndarray
    objective: np.ndarray
    costs: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    signed_step_energy_kwh: np.ndarray


def pack_states(states: Sequence[ModelState]) -> BatchedPhysicsState:
    if not states:
        raise ValueError("At least one physics state is required")
    return BatchedPhysicsState(
        position_m=np.asarray([s.vehicle.position_m for s in states], dtype=np.float64),
        velocity_m_s=np.asarray(
            [s.vehicle.velocity_m_s for s in states], dtype=np.float64
        ),
        acceleration_m_s2=np.asarray(
            [s.vehicle.acceleration_m_s2 for s in states], dtype=np.float64
        ),
        jerk_m_s3=np.asarray([s.vehicle.jerk_m_s3 for s in states], dtype=np.float64),
        elapsed_time_s=np.asarray(
            [s.vehicle.elapsed_time_s for s in states], dtype=np.float64
        ),
        total_energy_kwh=np.asarray(
            [s.vehicle.total_energy_kwh for s in states], dtype=np.float64
        ),
        maximum_speed_excess_m_s=np.asarray(
            [s.maximum_speed_excess_m_s for s in states], dtype=np.float64
        ),
        speed_violation_count=np.asarray(
            [s.speed_violation_count for s in states], dtype=np.int64
        ),
        violation_active=np.asarray(
            [s.violation_active for s in states], dtype=bool
        ),
        integrated_speed_violation_m=np.asarray(
            [s.integrated_speed_violation_m for s in states], dtype=np.float64
        ),
        episode_step=np.asarray([s.episode_step for s in states], dtype=np.int64),
    )


def _context_groups(
    contexts: Sequence[TrackContext],
) -> list[tuple[np.ndarray, TrackContext]]:
    groups: dict[int, list[int]] = {}
    exemplars: dict[int, TrackContext] = {}
    for index, context in enumerate(contexts):
        key = id(context.track)
        groups.setdefault(key, []).append(index)
        exemplars[key] = context
    return [
        (np.asarray(indices, dtype=np.int64), exemplars[key])
        for key, indices in groups.items()
    ]


def _track_query(
    positions_m: np.ndarray, contexts: Sequence[TrackContext]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size = len(contexts)
    current = np.empty(size, dtype=np.float64)
    future = np.empty((size, 2), dtype=np.float64)
    distances = np.empty((size, 2), dtype=np.float64)
    remaining = np.empty(size, dtype=np.float64)
    for rows, context in _context_groups(contexts):
        track = context.track
        position = positions_m[rows]
        indices = np.searchsorted(track.positions_m, position, side="right") - 1
        indices = np.clip(indices, 0, len(track.positions_m) - 1)
        current[rows] = track.limits_m_s[indices]
        visible = current[rows].copy()
        for offset in (1, 2):
            target = indices + offset
            valid = target < len(track.positions_m)
            ahead = np.full(rows.shape, context.sensor_range_m, dtype=np.float64)
            ahead[valid] = track.positions_m[target[valid]] - position[valid]
            seen = valid & (ahead <= context.sensor_range_m)
            visible[seen] = track.limits_m_s[target[seen]]
            future[rows, offset - 1] = visible
            distances[rows, offset - 1] = np.where(
                seen, ahead, context.sensor_range_m
            )

        ends = np.r_[track.positions_m[1:], context.track_length_m]
        durations = (ends - track.positions_m) / track.limits_m_s
        suffix_after = np.r_[np.cumsum(durations[:0:-1])[::-1], 0.0]
        clipped = np.minimum(position, context.track_length_m)
        segment_remaining = (
            ends[indices] - np.maximum(clipped, track.positions_m[indices])
        ) / track.limits_m_s[indices]
        remaining[rows] = np.maximum(0.0, segment_remaining + suffix_after[indices])
    return current, future, distances, remaining


def observations(
    state: BatchedPhysicsState,
    contexts: Sequence[TrackContext],
    vehicle: VehicleModel,
    config: SimulationConfig,
) -> np.ndarray:
    if len(state) != len(contexts):
        raise ValueError("State/context batch sizes differ")
    current, future, distances, _remaining = _track_query(
        state.position_m, contexts
    )
    energy_factor = np.asarray(
        [context.energy_factor for context in contexts], dtype=np.float64
    )
    raw = np.column_stack(
        (
            state.velocity_m_s,
            state.acceleration_m_s2,
            current,
            future,
            distances,
            energy_factor,
        )
    )
    low = np.asarray(
        [0.0, vehicle.specs.acceleration_limits[0], 0, 0, 0, 0, 0, 0],
        dtype=np.float64,
    )
    high = np.asarray(
        [
            vehicle.specs.velocity_limits[1],
            vehicle.specs.acceleration_limits[1],
            vehicle.specs.velocity_limits[1],
            vehicle.specs.velocity_limits[1],
            vehicle.specs.velocity_limits[1],
            config.sensor_range_m,
            config.sensor_range_m,
            5.0,
        ],
        dtype=np.float64,
    )
    return (raw - low) / (high - low)


def batched_step(
    state: BatchedPhysicsState,
    actions: np.ndarray,
    *,
    contexts: Sequence[TrackContext],
    vehicle: VehicleModel,
    config: SimulationConfig,
    energy_scale_kwh: float = 0.25,
    deadline_s: float = 140.0,
    max_episode_steps: int = 1800,
) -> PhysicsStepBatch:
    """Advance a batch with equations identical to ``domain.dynamics.advance``."""

    size = len(state)
    actions = np.asarray(actions, dtype=np.float64).reshape(-1)
    if actions.shape != (size,) or not np.isfinite(actions).all():
        raise ValueError("Actions must be one finite scalar per state")
    actions = np.clip(actions, -1.0, 1.0)
    if len(contexts) != size:
        raise ValueError("State/context batch sizes differ")
    observation = observations(state, contexts, vehicle, config)
    velocity = state.velocity_m_s
    minimum = np.interp(
        velocity,
        vehicle.velocity_grid,
        vehicle.negative_acceleration,
        left=0.0,
        right=-1e-6,
    )
    coast = np.interp(
        velocity,
        vehicle.velocity_grid,
        vehicle.coasting_acceleration,
        left=0.0,
        right=-1e-6,
    )
    maximum = np.interp(
        velocity,
        vehicle.velocity_grid,
        vehicle.positive_acceleration,
        left=0.6258544444444445,
        right=-1e-6,
    )
    limit = np.where(actions > 0.0, maximum, minimum)
    acceleration = (limit - coast) * np.abs(actions) + coast
    low, high = vehicle.specs.velocity_limits
    acceleration = np.minimum(
        np.maximum(acceleration, (low - velocity) / config.dt_s),
        (high - velocity) / config.dt_s,
    )
    next_velocity = velocity + acceleration * config.dt_s
    signed_energy = (
        vehicle.power_model.predict(np.column_stack((next_velocity, acceleration)))
        * config.dt_s
        / 3600.0
    )
    next_position = (
        state.position_m
        + velocity * config.dt_s
        + 0.5 * acceleration * config.dt_s**2
    )
    next_elapsed = state.elapsed_time_s + config.dt_s
    current_limit, _future, _distances, remaining = _track_query(
        next_position, contexts
    )
    excess = np.maximum(0.0, next_velocity - current_limit)
    violating = excess > 0.0
    next_state = BatchedPhysicsState(
        position_m=next_position,
        velocity_m_s=next_velocity,
        acceleration_m_s2=acceleration,
        jerk_m_s3=np.abs(acceleration - state.acceleration_m_s2) / config.dt_s,
        elapsed_time_s=next_elapsed,
        total_energy_kwh=state.total_energy_kwh + signed_energy,
        maximum_speed_excess_m_s=np.maximum(
            state.maximum_speed_excess_m_s, excess
        ),
        speed_violation_count=(
            state.speed_violation_count
            + (violating & ~state.violation_active).astype(np.int64)
        ),
        violation_active=violating,
        integrated_speed_violation_m=(
            state.integrated_speed_violation_m + excess * config.dt_s
        ),
        episode_step=state.episode_step + 1,
    )
    track_lengths = np.asarray(
        [context.track_length_m for context in contexts], dtype=np.float64
    )
    terminated = next_position >= track_lengths
    truncated = (next_state.episode_step >= max_episode_steps) & ~terminated
    deficit = np.maximum(0.0, -(deadline_s - next_elapsed - remaining))
    costs = np.column_stack(
        (
            excess * config.dt_s,
            config.dt_s * deficit / deadline_s,
        )
    )
    return PhysicsStepBatch(
        state=next_state,
        observation=observation,
        next_observation=observations(next_state, contexts, vehicle, config),
        objective=-signed_energy / energy_scale_kwh,
        costs=costs,
        terminated=terminated,
        truncated=truncated,
        signed_step_energy_kwh=signed_energy,
    )


def scalar_step(
    state: ModelState,
    action: float,
    *,
    context: TrackContext,
    vehicle: VehicleModel,
    config: SimulationConfig,
    source_real_transition_id: int = 0,
    rng: np.random.Generator | None = None,
):
    """Reference projection through the already-validated scalar adapter."""

    return model_step(
        PhysicsDynamicsModel(vehicle, config),
        state,
        action,
        context=context,
        vehicle=vehicle,
        config=config,
        energy_scale_kwh=0.25,
        deadline_s=140.0,
        max_episode_steps=1800,
        source_real_transition_id=source_real_transition_id,
        rng=np.random.default_rng(0) if rng is None else rng,
    )


@dataclass(frozen=True)
class BrakingHorizon:
    start_speed_km_h: float
    target_speed_km_h: float
    steps: int
    time_s: float
    distance_m: float
    final_speed_m_s: float


def braking_horizon(
    start_speed_km_h: float,
    target_speed_km_h: float,
    *,
    vehicle: VehicleModel | None = None,
    config: SimulationConfig | None = None,
) -> BrakingHorizon:
    """Measure exact maximal-braking time; no tracks or policy results are opened."""

    vehicle = VehicleModel() if vehicle is None else vehicle
    config = SimulationConfig() if config is None else config
    state = VehicleState(velocity_m_s=start_speed_km_h / 3.6)
    target = target_speed_km_h / 3.6
    steps = 0
    while state.velocity_m_s > target:
        state, _power, _energy = advance(state, -1.0, vehicle, config)
        steps += 1
        if steps > ceil(60.0 / config.dt_s):
            raise RuntimeError("Maximal braking failed to reach the target")
    return BrakingHorizon(
        start_speed_km_h=float(start_speed_km_h),
        target_speed_km_h=float(target_speed_km_h),
        steps=steps,
        time_s=steps * config.dt_s,
        distance_m=state.position_m,
        final_speed_m_s=state.velocity_m_s,
    )


def scalar_rollout(
    state: ModelState,
    actions: Iterable[float],
    *,
    context: TrackContext,
    vehicle: VehicleModel,
    config: SimulationConfig,
) -> list:
    transitions = []
    for transition_id, action in enumerate(actions):
        transition = scalar_step(
            state,
            float(action),
            context=context,
            vehicle=vehicle,
            config=config,
            source_real_transition_id=transition_id,
        )
        transitions.append(transition)
        state = transition.next_state
        if transition.terminated or transition.truncated:
            break
    return transitions
