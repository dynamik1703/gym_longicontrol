"""Create sanitized Development-only reward-reflection artifacts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from statistics import fmean, median
from typing import Any

from .history import (
    DEFAULT_HISTORY_PATH,
    candidate_by_id,
    load_history,
    save_history,
)
from .protocol import DEFAULT_PROTOCOL_PATH, load_protocol

FORBIDDEN_FEEDBACK_TERMS = (
    "validation",
    "paper_final",
    "paper-final",
    "constrained_v2",
    "requirement_conditioned",
    "binary_reward",
    "deadline_slack",
    "deadline-slack",
)


def _read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write(path: str | Path, text: str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _distribution(values) -> dict[str, float | int | None]:
    numeric = [float(value) for value in values]
    if not numeric:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(numeric),
        "mean": fmean(numeric),
        "median": float(median(numeric)),
        "min": min(numeric),
        "max": max(numeric),
    }


def assert_development_only(screening: dict[str, Any], protocol) -> None:
    if screening.get("evaluation_scope") != "development-only":
        raise ValueError("Reflection source is not marked Development-only")
    expected = set(protocol.screening.development_tracks)
    for checkpoint in screening["checkpoints"]:
        if checkpoint["evaluation_split_id"] != protocol.screening.development_split_id:
            raise ValueError("Reflection source includes a non-Development split")
        seeds = {item["evaluation_seed"] for item in checkpoint["episodes"]}
        if seeds != expected:
            raise ValueError(
                "Reflection source does not contain exact Development tracks"
            )
    forbidden = (
        set(protocol.track_policy.historical_exploratory)
        | set(protocol.track_policy.validation)
        | set(protocol.track_policy.paper_final_test_reserved)
    )
    if any(
        item["evaluation_seed"] in forbidden
        for checkpoint in screening["checkpoints"]
        for item in checkpoint["episodes"]
    ):
        raise ValueError("Forbidden track leaked into screening")


def build_feedback(screening: dict[str, Any], protocol) -> dict[str, Any]:
    assert_development_only(screening, protocol)
    final = screening["checkpoints"][-1]
    episodes = final["episodes"]
    route_length = float(screening["route_length_m"])
    if route_length <= 0:
        raise ValueError("Screening route length must be positive")
    incomplete_progress = [
        min(max(item["final_position_m"] / route_length, 0.0), 1.0)
        for item in episodes
        if not item["completed"]
    ]
    relevant = [
        item
        for item in episodes
        if item["completed"]
        or item["final_position_m"] / route_length
        >= protocol.ranking.substantial_progress_threshold
    ]
    completed = [item for item in episodes if item["completed"]]
    feasible = [item for item in episodes if item["feasible"]]
    deadline_compliance = (
        sum(item["travel_time_s"] <= protocol.task.max_time_s for item in completed)
        / len(completed)
        if completed
        else 0.0
    )
    speed_compliance = (
        sum(
            item["max_speed_violation_m_s"] <= protocol.task.max_speed_violation_m_s
            for item in relevant
        )
        / len(relevant)
        if relevant
        else 0.0
    )
    speed_severity = (
        fmean(item["integrated_speed_violation_m"] for item in relevant)
        if relevant
        else None
    )
    training = screening["training_outcomes"]
    return {
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "candidate_id": screening["candidate_id"],
        "candidate_source_sha256": screening["candidate_source_sha256"],
        "generation": screening["generation"],
        "information_scope": "development-only-aggregate",
        "reward_component_statistics": screening["component_statistics"],
        "episode_reward_statistics": screening["episode_reward_statistics"],
        "training_success_trajectory": {
            "successful_training_episode_count": sum(
                item["success"] for item in training
            ),
            "completed_training_episode_count": len(training),
            "first_success_training_step": next(
                (item["training_step"] for item in training if item["success"]), None
            ),
            "development_checkpoints": [
                {
                    "simulator_transitions": item["simulator_transitions"],
                    "requirement_satisfaction_rate": item["summary"][
                        "requirement_satisfaction_rate"
                    ],
                    "completion_rate": item["summary"]["completion_rate"],
                    "cumulative_training_successes": item[
                        "cumulative_training_successes"
                    ],
                }
                for item in screening["checkpoints"]
            ],
        },
        "development_metrics": {
            "episode_count": len(episodes),
            "requirement_satisfaction_rate": sum(item["feasible"] for item in episodes)
            / len(episodes),
            "completion_rate": len(completed) / len(episodes),
            "deadline_compliance_rate_among_completed": deadline_compliance,
            "speed_compliance_rate_among_relevant": speed_compliance,
            "mean_integrated_speed_violation_m_among_relevant": speed_severity,
            "failure_mode_counts": dict(
                sorted(
                    Counter(
                        "feasible"
                        if item["feasible"]
                        else "+".join(
                            name
                            for name, failed in (
                                ("incomplete", not item["completed"]),
                                (
                                    "time",
                                    item["travel_time_s"] > protocol.task.max_time_s,
                                ),
                                (
                                    "speed",
                                    item["max_speed_violation_m_s"]
                                    > protocol.task.max_speed_violation_m_s,
                                ),
                            )
                            if failed
                        )
                        for item in episodes
                    ).items()
                )
            ),
            "travel_time_s": _distribution(item["travel_time_s"] for item in episodes),
            "max_speed_violation_m_s": _distribution(
                item["max_speed_violation_m_s"] for item in episodes
            ),
            "integrated_speed_violation_m": _distribution(
                item["integrated_speed_violation_m"] for item in episodes
            ),
            "feasible_energy_kwh": _distribution(
                item["energy_kwh"] for item in feasible
            ),
            "episode_length_steps": _distribution(
                item["step_count"] for item in episodes
            ),
            "normalized_route_progress_for_incomplete": _distribution(
                incomplete_progress
            ),
        },
        "ranking_fields": {
            "development_rsr": sum(item["feasible"] for item in episodes)
            / len(episodes),
            "completion_rate": len(completed) / len(episodes),
            "median_incomplete_route_progress": (
                float(median(incomplete_progress)) if incomplete_progress else 1.0
            ),
            "deadline_compliance_among_completed": deadline_compliance,
            "speed_compliance_among_relevant": speed_compliance,
            "speed_violation_severity_among_relevant": speed_severity,
            "mean_feasible_energy_kwh": (
                fmean(item["energy_kwh"] for item in feasible) if feasible else None
            ),
        },
    }


def validate_feedback(feedback: dict[str, Any]) -> None:
    serialized = json.dumps(feedback, sort_keys=True).lower()
    for term in FORBIDDEN_FEEDBACK_TERMS:
        if term in serialized:
            raise ValueError(f"Forbidden information in reflection: {term}")
    if "evaluation_seed" in serialized or "source_path" in serialized:
        raise ValueError("Per-track identity or source path leaked into reflection")
    if feedback.get("information_scope") != "development-only-aggregate":
        raise ValueError("Reflection is not explicitly Development-only")


def _markdown(feedback: dict[str, Any]) -> str:
    metrics = feedback["development_metrics"]
    ranking = feedback["ranking_fields"]
    components = feedback["reward_component_statistics"]
    lines = [
        f"# Reward reflection: {feedback['candidate_id']}",
        "",
        "Scope: aggregated Development information only.",
        "",
        "## Task outcomes",
        "",
        f"- RSR: {metrics['requirement_satisfaction_rate']:.6f}",
        f"- Completion rate: {metrics['completion_rate']:.6f}",
        "- Deadline compliance among completed: "
        f"{metrics['deadline_compliance_rate_among_completed']:.6f}",
        "- Speed compliance among relevant trajectories: "
        f"{metrics['speed_compliance_rate_among_relevant']:.6f}",
        "- Failure modes: "
        f"{json.dumps(metrics['failure_mode_counts'], sort_keys=True)}",
        "",
        "## Reward components",
        "",
    ]
    for name, summary in sorted(components.items()):
        lines.append(
            f"- `{name}`: mean={summary['mean']:.6g}, std={summary['std']:.6g}, "
            f"min={summary['min']:.6g}, max={summary['max']:.6g}"
        )
    lines.extend(
        [
            "",
            "## Deterministic ranking fields",
            "",
            "```json",
            json.dumps(ranking, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def generate_feedback(
    candidate_id: str,
    *,
    protocol,
    history_path: str | Path,
    output_root: str | Path,
) -> tuple[Path, Path]:
    history = load_history(history_path, protocol)
    if history["status"] != "OPEN":
        raise RuntimeError("Feedback generation requires an OPEN search")
    candidate = candidate_by_id(history, candidate_id)
    if candidate["status"] != "screened" or candidate["screening"] is None:
        raise RuntimeError("Candidate must be screened before reflection")
    screening = _read(candidate["screening"]["result_path"])
    feedback = build_feedback(screening, protocol)
    validate_feedback(feedback)
    root = Path(output_root)
    json_path = _write(
        root / f"{candidate_id}.json",
        json.dumps(feedback, indent=2, sort_keys=True) + "\n",
    )
    markdown_path = _write(root / f"{candidate_id}.md", _markdown(feedback))
    updated = deepcopy(history)
    record = candidate_by_id(updated, candidate_id)
    record["reflection"] = {
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
    }
    record["status"] = "reflected"
    save_history(history_path, updated, protocol)
    return json_path, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_id")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    paths = generate_feedback(
        args.candidate_id,
        protocol=load_protocol(args.protocol),
        history_path=args.history,
        output_root=args.output_root,
    )
    print("\n".join(map(str, paths)))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
