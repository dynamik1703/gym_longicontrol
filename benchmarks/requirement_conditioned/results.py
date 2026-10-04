"""Episode-level persistence for requirement-conditioned evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean, median
from typing import Any


@dataclass(frozen=True)
class RequirementEpisodeEvaluation:
    track_seed: int
    requirement_kind: str
    requirement_margin_s: float
    t_min_start_s: float
    max_time_s: float
    completed: bool
    deadline_met: bool
    speed_compliant: bool
    feasible: bool
    travel_time_s: float
    energy_kwh: float
    speed_violation_count: int
    max_speed_violation_m_s: float
    integrated_speed_violation_m: float
    deadline_deficit_integral_s: float
    maximum_deadline_deficit_s: float
    step_count: int
    final_position_m: float
    traction_energy_kwh: float
    regenerative_energy_kwh: float
    mean_abs_jerk_m_s3: float
    max_abs_jerk_m_s3: float
    mean_abs_acceleration_m_s2: float
    mean_abs_action: float
    action_total_variation: float
    mean_abs_action_change: float
    acceleration_sign_change_count: int
    action_profile: tuple[float, ...]


def summarize_episodes(episodes) -> dict[str, Any]:
    values = tuple(episodes)
    if not values:
        raise ValueError("At least one episode is required")
    count = len(values)
    feasible_energy = [item.energy_kwh for item in values if item.feasible]
    return {
        "episode_count": count,
        "requirement_satisfaction_rate": sum(item.feasible for item in values) / count,
        "completion_rate": sum(item.completed for item in values) / count,
        "deadline_compliance_rate": sum(item.deadline_met for item in values) / count,
        "speed_compliance_rate": sum(item.speed_compliant for item in values) / count,
        "mean_travel_time_s": fmean(item.travel_time_s for item in values),
        "median_travel_time_s": float(median(item.travel_time_s for item in values)),
        "mean_feasible_energy_kwh": (
            fmean(feasible_energy) if feasible_energy else None
        ),
        "median_feasible_energy_kwh": (
            float(median(feasible_energy)) if feasible_energy else None
        ),
        "mean_max_speed_violation_m_s": fmean(
            item.max_speed_violation_m_s for item in values
        ),
        "mean_integrated_speed_violation_m": fmean(
            item.integrated_speed_violation_m for item in values
        ),
    }


@dataclass(frozen=True)
class RequirementRunResult:
    benchmark_name: str
    configuration_sha256: str
    evaluation_split_id: str
    training_seed: int
    simulator_step_target: int
    simulator_steps: int
    gradient_updates: int
    training_wall_time_s: float
    episodes: tuple[RequirementEpisodeEvaluation, ...]
    summaries_by_margin: dict[str, dict[str, Any]]
    overall_summary: dict[str, Any]
    checkpoint_multipliers: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, **asdict(self)}


def make_result(**kwargs) -> RequirementRunResult:
    episodes = tuple(kwargs.pop("episodes"))
    margins = sorted({item.requirement_margin_s for item in episodes})
    summaries = {
        f"{margin:g}": summarize_episodes(
            item for item in episodes if item.requirement_margin_s == margin
        )
        for margin in margins
    }
    return RequirementRunResult(
        episodes=episodes,
        summaries_by_margin=summaries,
        overall_summary=summarize_episodes(episodes),
        **kwargs,
    )


def save_result(path: str | Path, result: RequirementRunResult) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_result(path: str | Path) -> RequirementRunResult:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.pop("schema_version", None) != 1:
        raise ValueError("Unsupported requirement result schema")
    raw["episodes"] = tuple(
        RequirementEpisodeEvaluation(
            **{**item, "action_profile": tuple(item["action_profile"])}
        )
        for item in raw["episodes"]
    )
    raw["checkpoint_multipliers"] = tuple(raw["checkpoint_multipliers"])
    result = RequirementRunResult(**raw)
    expected = make_result(
        benchmark_name=result.benchmark_name,
        configuration_sha256=result.configuration_sha256,
        evaluation_split_id=result.evaluation_split_id,
        training_seed=result.training_seed,
        simulator_step_target=result.simulator_step_target,
        simulator_steps=result.simulator_steps,
        gradient_updates=result.gradient_updates,
        training_wall_time_s=result.training_wall_time_s,
        episodes=result.episodes,
        checkpoint_multipliers=result.checkpoint_multipliers,
    )
    if (
        expected.summaries_by_margin != result.summaries_by_margin
        or expected.overall_summary != result.overall_summary
    ):
        raise ValueError("Stored summaries do not match episodes")
    return result
