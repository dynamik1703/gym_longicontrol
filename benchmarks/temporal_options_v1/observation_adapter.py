"""Causal 70-dimensional state and 18-action interface for later learning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from benchmarks.causal_mpc_v2.planner import PROFILES

from .option_state import OptionState, TerminationReason

PHASES = (
    "inactive",
    "regulated_feedback",
    "pulse_traction",
    "traction",
    "coast",
    "partial_traction",
    "brake",
    "fallback",
)
TERMINATIONS = tuple(reason.value for reason in TerminationReason)
PROFILE_IDS = tuple(profile.profile_id for profile in PROFILES)
ACTION_IDS = ("CONTINUE", *(f"START({profile_id})" for profile_id in PROFILE_IDS))
BASE_DIMENSION = 16
OPTION_OBSERVATION_DIMENSION = (
    BASE_DIMENSION
    + 1
    + len(PROFILE_IDS)
    + len(PHASES)
    + 3
    + len(TERMINATIONS)
    + len(ACTION_IDS)
)


@dataclass(frozen=True)
class OptionObservation:
    vector: np.ndarray
    valid_action_mask: np.ndarray


def valid_action_mask(active_option: OptionState | None) -> np.ndarray:
    mask = np.ones(len(ACTION_IDS), dtype=np.float64)
    mask[0] = float(active_option is not None)
    return mask


def augment_observation(
    base_strategy_observation: np.ndarray,
    *,
    active_option: OptionState | None,
    previous_termination: str,
) -> OptionObservation:
    base = np.asarray(base_strategy_observation, dtype=np.float64)
    if base.shape != (BASE_DIMENSION,) or not np.isfinite(base).all():
        raise ValueError("base strategy observation must be a finite 16-vector")
    profile = np.zeros(len(PROFILE_IDS), dtype=np.float64)
    phases = np.zeros(len(PHASES), dtype=np.float64)
    active = float(active_option is not None)
    progress = elapsed = remaining = 0.0
    if active_option is None:
        phases[PHASES.index("inactive")] = 1.0
    else:
        profile[PROFILE_IDS.index(active_option.profile_id)] = 1.0
        phases[PHASES.index(active_option.phase)] = 1.0
        progress = active_option.progress
        elapsed = min(1.0, active_option.native_steps_elapsed / 400.0)
        remaining = active_option.remaining_distance_m / 100.0
    termination = np.zeros(len(TERMINATIONS), dtype=np.float64)
    termination[TERMINATIONS.index(str(previous_termination))] = 1.0
    mask = valid_action_mask(active_option)
    vector = np.concatenate(
        (
            base,
            np.array([active]),
            profile,
            phases,
            np.array([progress, elapsed, remaining]),
            termination,
            mask,
        )
    )
    if vector.shape != (OPTION_OBSERVATION_DIMENSION,):
        raise RuntimeError("augmented option observation contract changed")
    return OptionObservation(vector=vector, valid_action_mask=mask)


def action_from_index(index: int) -> tuple[str, str | None]:
    value = int(index)
    if not 0 <= value < len(ACTION_IDS):
        raise ValueError("high-level action index is outside [0, 17]")
    return ("CONTINUE", None) if value == 0 else ("START", PROFILE_IDS[value - 1])


def discount_for_duration(gamma_native: float, native_steps: int) -> float:
    gamma = float(gamma_native)
    steps = int(native_steps)
    if not 0.0 <= gamma <= 1.0 or steps < 0:
        raise ValueError("gamma must be in [0,1] and duration nonnegative")
    return gamma**steps
