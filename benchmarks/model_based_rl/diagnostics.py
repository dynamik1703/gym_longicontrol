"""Safety-sensitive learned-model and exact-physics diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .model_state import ProjectedTransition


@dataclass(frozen=True)
class ErrorSummary:
    count: int
    position_mae_m: float
    position_rmse_m: float
    velocity_mae_m_s: float
    velocity_rmse_m_s: float
    acceleration_mae_m_s2: float
    acceleration_rmse_m_s2: float
    signed_energy_mae_kwh: float
    signed_energy_rmse_kwh: float
    false_safe_count: int
    false_safe_rate: float
    false_unsafe_count: int
    false_unsafe_rate: float
    speed_excess_mae_m_s: float
    deadline_category_error_count: int
    deadline_category_error_rate: float
    termination_accuracy: float
    maximum_ensemble_variance: float | None

    def as_dict(self) -> dict:
        return asdict(self)


def _mae_rmse(errors: np.ndarray) -> tuple[float, float]:
    return float(np.abs(errors).mean()), float(np.sqrt(np.square(errors).mean()))


def summarize_predictions(
    predicted: list[ProjectedTransition], actual: list[ProjectedTransition]
) -> ErrorSummary:
    if not predicted or len(predicted) != len(actual):
        raise ValueError(
            "Prediction and reference sequences must be non-empty and match"
        )
    predicted_values = np.asarray(
        [
            [
                row.next_state.vehicle.position_m,
                row.next_state.vehicle.velocity_m_s,
                row.next_state.vehicle.acceleration_m_s2,
                row.next_state.vehicle.total_energy_kwh
                - row.state.vehicle.total_energy_kwh,
            ]
            for row in predicted
        ]
    )
    actual_values = np.asarray(
        [
            [
                row.next_state.vehicle.position_m,
                row.next_state.vehicle.velocity_m_s,
                row.next_state.vehicle.acceleration_m_s2,
                row.next_state.vehicle.total_energy_kwh
                - row.state.vehicle.total_energy_kwh,
            ]
            for row in actual
        ]
    )
    errors = predicted_values - actual_values
    position = _mae_rmse(errors[:, 0])
    velocity = _mae_rmse(errors[:, 1])
    acceleration = _mae_rmse(errors[:, 2])
    energy = _mae_rmse(errors[:, 3])
    predicted_excess = np.asarray([row.costs[0] / 0.1 for row in predicted])
    actual_excess = np.asarray([row.costs[0] / 0.1 for row in actual])
    actual_violation = actual_excess > 0
    predicted_violation = predicted_excess > 0
    false_safe = actual_violation & ~predicted_violation
    false_unsafe = ~actual_violation & predicted_violation
    actual_deadline = np.asarray([row.costs[1] > 0 for row in actual])
    predicted_deadline = np.asarray([row.costs[1] > 0 for row in predicted])
    variance_values = [
        max(row.predictive_variance)
        for row in predicted
        if row.predictive_variance is not None
    ]
    return ErrorSummary(
        count=len(predicted),
        position_mae_m=position[0],
        position_rmse_m=position[1],
        velocity_mae_m_s=velocity[0],
        velocity_rmse_m_s=velocity[1],
        acceleration_mae_m_s2=acceleration[0],
        acceleration_rmse_m_s2=acceleration[1],
        signed_energy_mae_kwh=energy[0],
        signed_energy_rmse_kwh=energy[1],
        false_safe_count=int(false_safe.sum()),
        false_safe_rate=float(false_safe.mean()),
        false_unsafe_count=int(false_unsafe.sum()),
        false_unsafe_rate=float(false_unsafe.mean()),
        speed_excess_mae_m_s=float(np.abs(predicted_excess - actual_excess).mean()),
        deadline_category_error_count=int(
            (actual_deadline != predicted_deadline).sum()
        ),
        deadline_category_error_rate=float(
            (actual_deadline != predicted_deadline).mean()
        ),
        termination_accuracy=float(
            np.mean(
                [
                    (left.terminated, left.truncated)
                    == (right.terminated, right.truncated)
                    for left, right in zip(predicted, actual)
                ]
            )
        ),
        maximum_ensemble_variance=(
            None if not variance_values else float(max(variance_values))
        ),
    )


def open_loop_horizon_errors(
    predicted_positions: np.ndarray,
    actual_positions: np.ndarray,
    *,
    horizons: tuple[int, ...] = (1, 5, 10, 25, 50),
) -> dict[str, float | None]:
    """Descriptive diagnostic only; never changes the frozen H=1 rollout."""

    if predicted_positions.shape != actual_positions.shape:
        raise ValueError("Open-loop arrays must have identical shapes")
    return {
        str(horizon): (
            None
            if len(predicted_positions) < horizon
            else float(
                np.sqrt(
                    np.mean(
                        (
                            predicted_positions[horizon - 1]
                            - actual_positions[horizon - 1]
                        )
                        ** 2
                    )
                )
            )
        )
        for horizon in horizons
    }
