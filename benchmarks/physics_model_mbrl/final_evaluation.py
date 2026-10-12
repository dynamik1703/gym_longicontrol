"""One-shot external Validation for three frozen Physics policies."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import fmean, median
from typing import Any

from benchmarks.constrained_rl.adapter import (
    deterministic_policy_adapter,
    multiplier_values,
)
from benchmarks.constrained_rl_v2.adapter import build_agent
from benchmarks.constrained_rl_v2.costs import DenseDeadlineTaskWrapper
from benchmarks.model_based_rl.checkpointing import file_sha256
from benchmarks.model_based_rl.execution import atomic_json, utc_now
from benchmarks.model_based_rl.recovery import (
    restore_policy_state,
    verify_recovery_checkpoint,
)
from benchmarks.scalar_sac.evaluation import (
    EpisodeEvaluation,
    EvaluationSummary,
    evaluate_policy,
)
from benchmarks.scalar_sac.experiment import _base_environment

from .config import TRAINING_SEEDS, load_configuration
from .execution import (
    assert_validation_authorized,
    load_or_create_manifest,
    validation_ready,
)


def _evaluate_seed(
    *,
    seed: int,
    checkpoint: Path,
    expected_sha256: str,
    tracks: tuple[int, ...],
    device: str,
    threads: int,
) -> dict[str, Any]:
    configuration = load_configuration()
    parent = configuration.parent
    if file_sha256(checkpoint) != expected_sha256:
        raise RuntimeError(f"Final checkpoint hash differs for Physics seed {seed}")
    payload, metadata = verify_recovery_checkpoint(
        checkpoint,
        configuration_sha256=parent.configuration_sha256,
    )
    if payload["model_condition"] != "physics" or payload["training_seed"] != seed:
        raise RuntimeError("Validation checkpoint identity differs")
    construction = DenseDeadlineTaskWrapper(
        _base_environment(parent.v2),
        task=parent.v2.task,
        energy_scale_kwh=parent.v2.objective.energy_scale_kwh,
        deadline_normalization_s=parent.v2.deadline_cost.normalization_s,
        max_simulator_steps=parent.v2.simulator_step_budget,
    )
    environment = _base_environment(parent.v2)
    try:
        agent, _logger = build_agent(
            parent.v2,
            construction,
            training_seed=seed,
            device=device,
            threads=threads,
        )
        restore_policy_state(agent.policy, payload)
        agent.policy.eval()
        episodes, summary = evaluate_policy(
            deterministic_policy_adapter(agent.policy),
            environment,
            task=parent.v2.task,
            evaluation_seeds=tracks,
        )
        return {
            "schema_version": 1,
            "training_seed": seed,
            "evaluation_tracks": list(tracks),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": expected_sha256,
            "checkpoint_metadata": metadata,
            "multipliers": {
                "lambda_speed": multiplier_values(agent.policy)[0],
                "lambda_deadline": multiplier_values(agent.policy)[1],
            },
            "episodes": [asdict(item) for item in episodes],
            "summary": asdict(summary),
        }
    finally:
        environment.close()
        construction.close()


def _exclusive_failure_mode(episode: dict[str, Any], max_time_s: float) -> str:
    if episode["feasible"]:
        return "feasible"
    if not episode["completed"]:
        return "incomplete"
    deadline = float(episode["travel_time_s"]) > max_time_s
    speed = float(episode["max_speed_violation_m_s"]) > 0.0
    if deadline and speed:
        return "deadline_and_speed"
    if deadline:
        return "deadline_only"
    if speed:
        return "speed_only"
    return "other"


def _seconds_between(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()


def _development_curves(output_root: Path) -> dict[str, list[dict[str, Any]]]:
    curves = {}
    configuration = load_configuration()
    for seed in TRAINING_SEEDS:
        values = []
        for step in configuration.raw["real_transition_checkpoints"]:
            path = (
                output_root
                / f"physics-seed-{seed}"
                / f"step-{step:06d}"
                / "development.json"
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            values.append({"real_transitions": step, "summary": payload["summary"]})
        curves[str(seed)] = values
    return curves


def _training_diagnostics(
    output_root: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    from benchmarks.model_based_rl.checkpointing import load_checkpoint

    result = {}
    for seed in TRAINING_SEEDS:
        run_id = f"physics-seed-{seed}"
        run = manifest["runs"][run_id]
        checkpoint = output_root / run_id / "step-300000" / "checkpoint.pt"
        payload = load_checkpoint(
            checkpoint, expected_sha256=run["final_checkpoint_sha256"]
        )
        diagnostics = payload["diagnostics"]
        td = [
            row
            for row in diagnostics
            if "td_error_real_mean" in row and "td_error_model_mean" in row
        ]
        snapshots = [
            row
            for row in diagnostics
            if row.get("record_type") == "physics_only_checkpoint_diagnostics"
        ]
        attempt_seconds = [
            _seconds_between(item.get("started_at"), item.get("ended_at"))
            for item in run["attempts"]
        ]
        result[str(seed)] = {
            "counters": payload["counters"],
            "synthetic_sampled": int(payload["synthetic_replay"].sampled_total),
            "pid_real_episode_updates": int(payload["pid_real_episode_updates"]),
            "source_td_error": {
                "sample_count": len(td),
                "mean_real": fmean(row["td_error_real_mean"] for row in td),
                "mean_model": fmean(row["td_error_model_mean"] for row in td),
                "final_real": td[-1]["td_error_real_mean"],
                "final_model": td[-1]["td_error_model_mean"],
            },
            "checkpoint_diagnostics": snapshots,
            "attempt_wall_clock_seconds": attempt_seconds,
            "total_recorded_wall_clock_seconds": sum(
                value for value in attempt_seconds if value is not None
            ),
        }
    return result


def _aggregate(
    evaluations: list[dict[str, Any]], output_root: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    configuration = load_configuration()
    episodes = [episode for item in evaluations for episode in item["episodes"]]
    task = configuration.parent.v2.task
    summary = EvaluationSummary.from_episodes(
        tuple(EpisodeEvaluation(**episode) for episode in episodes),
        task,
    )
    counts = {
        "requirement_satisfaction": sum(item["feasible"] for item in episodes),
        "completion": sum(item["completed"] for item in episodes),
        "deadline_compliance": sum(
            float(item["travel_time_s"]) <= task.max_time_s for item in episodes
        ),
        "speed_compliance": sum(
            float(item["max_speed_violation_m_s"]) <= task.max_speed_violation_m_s
            for item in episodes
        ),
    }
    modes = {
        name: 0
        for name in (
            "feasible",
            "incomplete",
            "deadline_only",
            "speed_only",
            "deadline_and_speed",
            "other",
        )
    }
    for episode in episodes:
        modes[_exclusive_failure_mode(episode, task.max_time_s)] += 1
    feasible_energy = [item["energy_kwh"] for item in episodes if item["feasible"]]
    gate = configuration.raw["interpretation_gates"]["strong_improvement"]
    strong = (
        counts["requirement_satisfaction"]
        >= gate["minimum_requirement_satisfaction_count"]
        and counts["speed_compliance"] >= gate["minimum_speed_compliance_count"]
        and counts["completion"] == gate["required_completion_count"]
        and counts["deadline_compliance"] == gate["required_deadline_compliance_count"]
    )
    if strong:
        classification = "STRONG_IMPROVEMENT"
    elif (
        counts["requirement_satisfaction"]
        <= configuration.raw["interpretation_gates"][
            "no_material_improvement_maximum_rsr_count"
        ]
    ):
        classification = "NO_MATERIAL_IMPROVEMENT"
    else:
        classification = "INCONCLUSIVE_BETWEEN_PREREGISTERED_GATES"
    reference = configuration.raw["historical_reference"]
    return {
        "schema_version": 1,
        "study": configuration.raw["name"],
        "configuration_sha256": configuration.configuration_sha256,
        "evaluated_at_utc": utc_now(),
        "validation_episode_count": len(episodes),
        "per_seed": {
            str(item["training_seed"]): {
                "requirement_satisfaction_count": sum(
                    episode["feasible"] for episode in item["episodes"]
                ),
                "episode_count": len(item["episodes"]),
                "summary": item["summary"],
                "multipliers": item["multipliers"],
            }
            for item in evaluations
        },
        "pooled_counts": counts,
        "pooled_summary": asdict(summary),
        "exclusive_failure_modes": modes,
        "feasible_only_energy_kwh": {
            "count": len(feasible_energy),
            "mean": fmean(feasible_energy) if feasible_energy else None,
            "median": float(median(feasible_energy)) if feasible_energy else None,
        },
        "interpretation": {
            "classification": classification,
            "strong_gate_passed": strong,
            "historical_reference": reference,
            "deltas_from_frozen_v2": {
                name: counts[name] - reference[name][0]
                for name in (
                    "requirement_satisfaction",
                    "completion",
                    "deadline_compliance",
                    "speed_compliance",
                )
            },
            "degradation_context": configuration.raw["interpretation_gates"][
                "degradation"
            ],
        },
        "evaluations": evaluations,
        "development_curves": _development_curves(output_root),
        "training_diagnostics": _training_diagnostics(output_root, manifest),
        "physics_parity": json.loads(
            Path("benchmarks/physics_model_mbrl/parity_verification.json").read_text(
                encoding="utf-8"
            )
        ),
        "paper_test": {"state": "SEALED", "episode_count": 0},
    }


def _write_report(path: Path, result: dict[str, Any]) -> None:
    counts = result["pooled_counts"]
    energy = result["feasible_only_energy_kwh"]
    lines = [
        "# Physics-Dynamics MBRL results",
        "",
        "The evaluated controller is a learned SACLag actor trained with exact ",
        "one-step physics-model augmentation. It is not MPC and performs no model ",
        "query at evaluation.",
        "",
        "## Validation",
        "",
        *(
            f"- Seed {seed}: {values['requirement_satisfaction_count']}/"
            f"{values['episode_count']} RSR"
            for seed, values in result["per_seed"].items()
        ),
        f"- Pooled RSR: {counts['requirement_satisfaction']}/27",
        f"- Completion: {counts['completion']}/27",
        f"- Deadline compliance: {counts['deadline_compliance']}/27",
        f"- Speed compliance: {counts['speed_compliance']}/27",
        "- Preregistered interpretation: "
        f"`{result['interpretation']['classification']}`",
        "",
        "## Energy and failures",
        "",
        f"Feasible-only energy uses {energy['count']} real Validation episodes; mean = "
        f"{energy['mean']}, median = {energy['median']} kWh.",
        "",
        "Exclusive failure modes: "
        f"`{json.dumps(result['exclusive_failure_modes'], sort_keys=True)}`.",
        "",
        "## Context",
        "",
        "Frozen Constrained V2 achieved 21/27 RSR. This is a close historical ",
        "comparison, "
        "not a perfect one-factor causal control, because the Physics condition mixes "
        "one-step synthetic targets with the frozen two-step real-data path.",
        "",
        "Development curves, full compact episode metrics, source-wise TD diagnostics, "
        "learner losses, multipliers, costs, accounting, parity and wall-clock are ",
        "stored "
        "in `results.json`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(
    output_root: str | Path, *, device: str = "cpu", threads: int = 4
) -> Path:
    root = Path(output_root)
    manifest_path, manifest = load_or_create_manifest(root)
    validation = manifest["validation"]
    if validation["state"] == "SEALED":
        assert_validation_authorized(manifest)
        validation.update(
            {
                "state": "RUNNING",
                "opened_at": utc_now(),
                "tracks": load_configuration().raw["track_splits"]["validation"],
                "episodes_planned": 27,
            }
        )
        atomic_json(manifest_path, manifest)
    elif validation["state"] != "RUNNING":
        raise RuntimeError("Validation has already completed and cannot be reopened")
    elif not validation_ready(manifest):
        raise RuntimeError(
            "Interrupted Validation cannot resume without frozen policies"
        )
    configuration = load_configuration()
    tracks = tuple(configuration.raw["track_splits"]["validation"])
    result_directory = root / "validation"
    evaluations = []
    for seed in TRAINING_SEEDS:
        partial = result_directory / f"physics-seed-{seed}.json"
        if partial.exists():
            value = json.loads(partial.read_text(encoding="utf-8"))
            if value["evaluation_tracks"] != list(tracks):
                raise RuntimeError("Partial Validation track set differs")
        else:
            run = manifest["runs"][f"physics-seed-{seed}"]
            value = _evaluate_seed(
                seed=seed,
                checkpoint=root
                / f"physics-seed-{seed}"
                / "step-300000"
                / "checkpoint.pt",
                expected_sha256=run["final_checkpoint_sha256"],
                tracks=tracks,
                device=device,
                threads=threads,
            )
            atomic_json(partial, value)
        evaluations.append(value)
    result = _aggregate(evaluations, root, manifest)
    destination = Path("benchmarks/physics_model_mbrl/results.json")
    atomic_json(destination, result)
    _write_report(Path("benchmarks/physics_model_mbrl/RESULTS.md"), result)
    manifest["validation"].update(
        {
            "state": "COMPLETED",
            "completed_at": utc_now(),
            "episode_count": 27,
            "results_path": str(destination),
        }
    )
    atomic_json(manifest_path, manifest)
    return destination
