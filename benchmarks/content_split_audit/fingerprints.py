"""Canonical route/task fingerprints and the future split-protection gate."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from importlib.resources import files
from math import isfinite
from numbers import Integral, Real
from typing import Any, Literal

import numpy as np

from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.track import Track
from gym_longicontrol.domain.vehicle import VehicleSpecs

FINGERPRINT_SCHEMA = "longicontrol-content-fingerprint-v1"
_REQUIRED_TASK_SEMANTICS = frozenset(
    {
        "observation",
        "dynamics",
        "objective",
        "costs",
        "feasibility",
        "termination",
        "requirements",
    }
)


def _canonical(value: Any) -> Any:
    """Convert supported values to a deterministic, JSON-safe value tree."""

    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, np.ndarray):
        return [_canonical(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        if not isfinite(result):
            raise ValueError("Canonical fingerprints reject non-finite floats")
        return {"__float_hex__": result.hex()}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Canonical fingerprint mapping keys must be strings")
        if "__float_hex__" in value:
            raise ValueError("__float_hex__ is reserved by canonical serialization")
        return {key: _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical(item) for item in value]
    raise TypeError(f"Unsupported canonical fingerprint value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Return the one canonical serialization accepted by this audit."""

    return json.dumps(
        _canonical(value),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def content_sha256(value: Any) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def route_payload(
    track: Track,
    *,
    config: SimulationConfig = SimulationConfig(),
) -> dict[str, Any]:
    """Describe realized infrastructure, without task/reward requirements."""

    positions = [float(value) for value in track.positions_m]
    limits = [float(value) for value in track.limits_m_s]
    ends = positions[1:] + [float(config.track_length_m)]
    segments = [
        {
            "index": index,
            "start_m": start,
            "end_m": end,
            "length_m": end - start,
            "speed_limit_m_s": limit,
        }
        for index, (start, end, limit) in enumerate(zip(positions, ends, limits))
    ]
    return {
        "schema": FINGERPRINT_SCHEMA,
        "kind": "route",
        "route_length_m": float(config.track_length_m),
        "cumulative_segment_positions_m": positions,
        "speed_limit_transition_positions_m": positions[1:],
        "speed_limits_m_s": limits,
        "segments": segments,
        "infrastructure_observation": {
            "sensor_range_m": float(config.sensor_range_m),
            "lookahead_transitions": 2,
            "speed_limit_semantics": "right-continuous-piecewise-constant",
            "unseen_transition_distance_clamp_m": float(config.sensor_range_m),
        },
    }


def default_environment_parameters(*, energy_factor: float = 1.0) -> dict[str, Any]:
    """Frozen physical parameters shared by the audited stochastic tasks."""

    power_resource = files("gym_longicontrol").joinpath(
        "assets/vehicle/BMW_electric_i3_2014.npz"
    )
    return {
        "simulation": asdict(SimulationConfig()),
        "initial_vehicle_state": asdict(VehicleState()),
        "vehicle": {
            "car_id": "BMW_electric_i3_2014",
            "specifications": asdict(VehicleSpecs()),
            "power_model_artifact_sha256": sha256(
                power_resource.read_bytes()
            ).hexdigest(),
            "air_density_kg_m3": 1.2,
            "rolling_resistance_coefficient": 0.015,
            "gravity_m_s2": 9.81,
            "action_to_acceleration_interpolation_velocity_grid_m_s": list(
                range(0, 38)
            ),
        },
        "energy_factor": float(energy_factor),
        "integration": {
            "name": "constant-acceleration-explicit-step-v1",
            "velocity_boundary_crossing_clipped_within_step": True,
            "energy_conversion": "power_kw * dt_s / 3600",
            "jerk": "abs(current_acceleration-previous_acceleration)/dt_s",
        },
        "track_generator": {
            "name": "StochasticTrackGenerator",
            "candidate_transition_positions_m": list(range(100, 1000, 100)),
            "transition_count_draw": "clip(uniform_integer_[0,9],minimum=1)",
            "speed_grid_km_h": list(range(20, 101, 10)),
            "maximum_adjacent_change_km_h": 40,
            "equal_adjacent_limits_forbidden": True,
            "initial_choice_reference_km_h": 40,
            "initial_excluded_limit_m_s": 20.0 / 3.6,
        },
    }


def full_task_payload(
    track: Track,
    task_semantics: Mapping[str, Any],
    *,
    config: SimulationConfig = SimulationConfig(),
    environment_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    missing = _REQUIRED_TASK_SEMANTICS - set(task_semantics)
    if missing:
        raise ValueError(f"Missing full-task semantic fields: {sorted(missing)}")
    route = route_payload(track, config=config)
    parameters = (
        default_environment_parameters()
        if environment_parameters is None
        else dict(environment_parameters)
    )
    parameters["simulation"] = asdict(config)
    return {
        "schema": FINGERPRINT_SCHEMA,
        "kind": "full-task",
        "route_sha256": content_sha256(route),
        "route": route,
        "environment_parameters": parameters,
        "task_semantics": dict(task_semantics),
    }


@dataclass(frozen=True)
class TaskFingerprint:
    route_sha256: str
    full_task_sha256: str
    numeric_seed: int | None = None


def fingerprint_track(
    track: Track,
    task_semantics: Mapping[str, Any],
    *,
    numeric_seed: int | None = None,
    config: SimulationConfig = SimulationConfig(),
    environment_parameters: Mapping[str, Any] | None = None,
) -> TaskFingerprint:
    route = route_payload(track, config=config)
    full = full_task_payload(
        track,
        task_semantics,
        config=config,
        environment_parameters=environment_parameters,
    )
    return TaskFingerprint(
        route_sha256=content_sha256(route),
        full_task_sha256=content_sha256(full),
        numeric_seed=numeric_seed,
    )


class ReservedTaskError(ValueError):
    """Raised before replay insertion when a task violates a split policy."""


def assert_task_not_reserved(
    task: TaskFingerprint,
    reserved_numeric_seeds: Sequence[int] | set[int] | frozenset[int],
    reserved_route_hashes: Sequence[str] | set[str] | frozenset[str],
    reserved_full_task_hashes: Sequence[str] | set[str] | frozenset[str],
    *,
    policy: Literal["S", "F", "R"] = "R",
) -> None:
    """Enforce Seed-only (S), Full-task-clean (F), or Route-clean (R)."""

    if policy not in {"S", "F", "R"}:
        raise ValueError("policy must be one of 'S', 'F', or 'R'")
    if task.numeric_seed is not None and task.numeric_seed in reserved_numeric_seeds:
        raise ReservedTaskError(f"numeric seed {task.numeric_seed} is reserved")
    if policy in {"F", "R"} and task.full_task_sha256 in reserved_full_task_hashes:
        raise ReservedTaskError("full-task content is reserved")
    if policy == "R" and task.route_sha256 in reserved_route_hashes:
        raise ReservedTaskError("route content is reserved")
