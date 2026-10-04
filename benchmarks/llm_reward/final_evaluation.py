"""Winner-only final training and Validation, gated behind an immutable freeze."""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from benchmarks.scalar_sac.evaluation import evaluate_policy
from benchmarks.scalar_sac.experiment import _base_environment
from benchmarks.scalar_sb3.experiment import _model, _policy_adapter

from .candidate_validation import load_reward_function, source_sha256
from .history import DEFAULT_HISTORY_PATH, load_history, save_history
from .protocol import DEFAULT_PROTOCOL_PATH, load_protocol, protocol_sha256
from .reward_api import CandidateRewardWrapper


def _write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def validate_final_evaluation_gate(
    history: dict[str, Any], protocol, final_reward_path: str | Path
) -> dict[str, Any]:
    if history["status"] != "CLOSED" or history["winner"] is None:
        raise RuntimeError("Final evaluation is forbidden until the search is CLOSED")
    winner = history["winner"]
    path = Path(final_reward_path)
    if Path(winner["final_reward_path"]).resolve() != path.resolve():
        raise ValueError("Only the mechanically frozen winner may be evaluated")
    source = path.read_text(encoding="utf-8")
    if source_sha256(source) != winner["source_sha256"]:
        raise ValueError("Frozen winner source hash mismatch")
    metadata = json.loads(Path(winner["metadata_path"]).read_text(encoding="utf-8"))
    if (
        metadata["source_sha256"] != winner["source_sha256"]
        or metadata["protocol_sha256"] != protocol_sha256(protocol)
        or metadata["search_status"] != "CLOSED"
    ):
        raise ValueError("Frozen winner metadata mismatch")
    return winner


def _training_environment(protocol, path: Path, winner):
    function, report = load_reward_function(path)
    if report.source_sha256 != winner["source_sha256"]:
        raise ValueError("Validated reward hash differs from frozen winner")
    return CandidateRewardWrapper(
        _base_environment(protocol),
        compute_reward=function,
        task=protocol.task,
        candidate_id=winner["candidate_id"],
        source_sha256=winner["source_sha256"],
    )


def _failure_mode(episode, task) -> str:
    failures = []
    if not episode.completed:
        failures.append("incomplete")
    if episode.travel_time_s > task.max_time_s:
        failures.append("time")
    if episode.max_speed_violation_m_s > task.max_speed_violation_m_s:
        failures.append("speed")
    return "+".join(failures) if failures else "feasible"


def _train_and_evaluate_seed(protocol, winner, reward_path, seed, output_root, device):
    import stable_baselines3
    from stable_baselines3.common.logger import Logger

    import gym_longicontrol

    output = Path(output_root) / f"training-seed-{seed}"
    if output.exists():
        raise FileExistsError(f"Final result already exists: {output}")
    output.mkdir(parents=True)
    environment = _training_environment(protocol, reward_path, winner)
    started = time.perf_counter()
    try:
        model = _model(protocol, "sac", environment, seed, device)
        model.set_logger(Logger(folder=None, output_formats=[]))
        model.learn(
            total_timesteps=protocol.final_evaluation.simulator_transitions_per_seed,
            progress_bar=False,
        )
        training_time = time.perf_counter() - started
        model.save(output / "model.zip")
        evaluation = _base_environment(protocol)
        try:
            episodes, summary = evaluate_policy(
                _policy_adapter(model),
                evaluation,
                task=protocol.task,
                evaluation_seeds=protocol.final_evaluation.validation_tracks,
            )
        finally:
            evaluation.close()
    finally:
        environment.close()
    payload = {
        "schema_version": 1,
        "protocol_id": protocol.protocol_id,
        "protocol_sha256": protocol_sha256(protocol),
        "candidate_id": winner["candidate_id"],
        "candidate_source_sha256": winner["source_sha256"],
        "algorithm": "Stable-Baselines3 SAC",
        "gym_longicontrol_version": gym_longicontrol.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "environment_id": protocol.environment_id,
        "max_episode_steps": protocol.max_episode_steps,
        "task": asdict(protocol.task),
        "sac_configuration": asdict(protocol.sac),
        "training_seed": seed,
        "simulator_transitions": (
            protocol.final_evaluation.simulator_transitions_per_seed
        ),
        "gradient_updates": int(getattr(model, "_n_updates", 0)),
        "training_wall_time_s": training_time,
        "evaluation_split_id": protocol.final_evaluation.evaluation_split_id,
        "episodes": [
            {**asdict(item), "failure_mode": _failure_mode(item, protocol.task)}
            for item in episodes
        ],
        "summary": asdict(summary),
        "failure_mode_counts": {
            mode: sum(_failure_mode(item, protocol.task) == mode for item in episodes)
            for mode in sorted(
                {_failure_mode(item, protocol.task) for item in episodes}
            )
        },
    }
    return _write_json(output / "validation-result.json", payload)


def run_final_evaluation(
    *,
    protocol,
    history_path: str | Path,
    final_reward_path: str | Path,
    output_root: str | Path,
    device: str = "auto",
) -> Path:
    history = load_history(history_path, protocol)
    winner = validate_final_evaluation_gate(history, protocol, final_reward_path)
    root = Path(output_root)
    if root.exists():
        raise FileExistsError(f"Final-evaluation root already exists: {root}")
    root.mkdir(parents=True)
    result_paths = [
        _train_and_evaluate_seed(
            protocol,
            winner,
            Path(final_reward_path),
            seed,
            root,
            device,
        )
        for seed in protocol.final_evaluation.training_seeds
    ]
    manifest = _write_json(
        root / "final-evaluation-manifest.json",
        {
            "schema_version": 1,
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol_sha256(protocol),
            "candidate_id": winner["candidate_id"],
            "candidate_source_sha256": winner["source_sha256"],
            "search_status": "CLOSED",
            "result_paths": [str(path) for path in result_paths],
            "training_seed_count": len(protocol.final_evaluation.training_seeds),
            "total_final_training_transitions": (
                len(protocol.final_evaluation.training_seeds)
                * protocol.final_evaluation.simulator_transitions_per_seed
            ),
        },
    )
    updated = deepcopy(history)
    if updated["engineering_effort"]["final_training_transitions"] != 0:
        raise RuntimeError("Final training was already recorded")
    updated["engineering_effort"]["final_training_transitions"] = (
        len(protocol.final_evaluation.training_seeds)
        * protocol.final_evaluation.simulator_transitions_per_seed
    )
    save_history(history_path, updated, protocol)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument(
        "--final-reward", type=Path, default=Path(__file__).with_name("FINAL_REWARD.py")
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda", "mps"), default="auto"
    )
    args = parser.parse_args()
    print(
        run_final_evaluation(
            protocol=load_protocol(args.protocol),
            history_path=args.history,
            final_reward_path=args.final_reward,
            output_root=args.output_root,
            device=args.device,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
