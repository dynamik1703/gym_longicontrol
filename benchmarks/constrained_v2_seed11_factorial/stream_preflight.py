"""Materialize the two frozen seed-11 streams and enforce split protection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.constrained_rl_v2.config import load_configuration
from benchmarks.constrained_v2_repro_audit.harness import (
    _build_current_policy,
    _module_sha256,
    _track_row,
    current_track_fingerprints,
    historical_track_fingerprints,
)
from benchmarks.scalar_sac.experiment import _base_environment

DEFAULT_AUDIT = Path("benchmarks/constrained_v2_repro_audit/results.json")
DEFAULT_OUTPUT = Path("benchmarks/constrained_v2_seed11_factorial/results.json")
DEFAULT_ARCHIVED_CURRENT = Path(
    "/Users/romanliessner/gitlab/gym_longicontrol_multistep_mbrl/runs/"
    "multistep-physics-mbrl-v1/matched_real_only-seed-11/step-300000/"
    "checkpoint.pt"
)


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_state_sha256(value: Any) -> str:
    """Hash nested learner state without relying on pickle/container bytes."""

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if item is None:
            digest.update(b"none;")
        elif isinstance(item, bool):
            digest.update(f"bool:{int(item)};".encode())
        elif isinstance(item, int):
            digest.update(f"int:{item};".encode())
        elif isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("Learner state contains a non-finite float")
            digest.update(f"float:{item.hex()};".encode())
        elif isinstance(item, str):
            digest.update(b"str:")
            digest.update(item.encode())
            digest.update(b";")
        elif isinstance(item, bytes):
            digest.update(b"bytes:")
            digest.update(item)
            digest.update(b";")
        elif isinstance(item, dict):
            digest.update(b"dict{")
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(f"{type(item).__name__}[".encode())
            for child in item:
                visit(child)
            digest.update(b"]")
        elif hasattr(item, "detach"):
            array = item.detach().cpu().contiguous().numpy()
            visit(array)
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"array:")
            digest.update(str(array.dtype).encode())
            digest.update(json.dumps(list(array.shape)).encode())
            digest.update(array.tobytes())
            digest.update(b";")
        else:
            raise TypeError(f"Unsupported learner-state value: {type(item)!r}")

    visit(value)
    return digest.hexdigest()


def initial_learner_hashes() -> dict[str, Any]:
    conditions = (
        "A_histTrack_histCollector_seed11",
        "B_histTrack_currentCollector_seed11",
        "C_currentTrack_histCollector_seed11",
        "D_currentTrack_currentCollector_seed11",
    )
    by_condition: dict[str, Any] = {}
    for condition in conditions:
        environment, policy = _build_current_policy(11)
        try:
            actor = _module_sha256(policy.actor)
            critics = [_module_sha256(item) for item in policy.critics]
            target_critics = [_module_sha256(item) for item in policy.critics_old]
            actor_optimizer = stable_state_sha256(policy.actor_optim.state_dict())
            critic_optimizer = stable_state_sha256(policy.critics_optim.state_dict())
            alpha_optimizer = stable_state_sha256(
                policy._alpha_optim.state_dict()  # noqa: SLF001
            )
            alpha_state = {
                "effective_alpha": float(policy._alpha.detach().cpu()),  # noqa: SLF001
                "log_alpha_sha256": stable_state_sha256(
                    policy._log_alpha  # noqa: SLF001
                ),
                "target_entropy": float(policy._target_entropy),  # noqa: SLF001
                "optimizer_sha256": alpha_optimizer,
            }
            pid_states = [item.state_dict() for item in policy.lag_optims]
            row = {
                "actor_sha256": actor,
                "reward_critic_sha256": critics[0],
                "speed_cost_critic_sha256": critics[1],
                "deadline_cost_critic_sha256": critics[2],
                "target_reward_critic_sha256": target_critics[0],
                "target_speed_cost_critic_sha256": target_critics[1],
                "target_deadline_cost_critic_sha256": target_critics[2],
                "actor_optimizer_sha256": actor_optimizer,
                "critic_optimizer_sha256": critic_optimizer,
                "alpha": alpha_state,
                "pid_state_sha256": stable_state_sha256(pid_states),
                "pid_multipliers": [
                    float(item.get_lag()) for item in policy.lag_optims
                ],
            }
            row["combined_learner_sha256"] = stable_state_sha256(row)
            by_condition[condition] = row
        finally:
            environment.close()
    values = list(by_condition.values())
    return {
        "by_condition": by_condition,
        "all_identical": all(item == values[0] for item in values[1:]),
        "audit_expected_actor_sha256": (
            "72e9207137ecd2f8e1c6506915f1348eb59f47133a3b181cbf09098888e928c5"
        ),
        "actor_matches_audit": all(
            item["actor_sha256"]
            == "72e9207137ecd2f8e1c6506915f1348eb59f47133a3b181cbf09098888e928c5"
            for item in values
        ),
    }


def study_track_row(raw: dict[str, Any], *, task: dict[str, float]) -> dict[str, Any]:
    content = {
        "task": task,
        "positions_m": raw["positions_m"],
        "limits_m_s": raw["limits_m_s"],
        "energy_factor": raw["energy_factor"],
        "initial_observation": raw["initial_observation"],
    }
    row = {
        "episode_index": int(raw["episode_index"]),
        "audit_sha256": raw["sha256"],
        "canonical_sha256": canonical_sha256(content),
        "content": content,
    }
    if "explicit_reset_seed" in raw:
        row["explicit_reset_seed"] = int(raw["explicit_reset_seed"])
    return row


def reserved_rows() -> list[dict[str, Any]]:
    configuration = load_configuration()
    split_seeds = {
        "development": tuple(range(2000, 2009)),
        "validation": tuple(range(3000, 3009)),
        "paper": tuple(range(4000, 4018)),
    }
    task = {
        "max_time_s": configuration.task.max_time_s,
        "max_speed_violation_m_s": configuration.task.max_speed_violation_m_s,
    }
    environment = _base_environment(configuration)
    rows: list[dict[str, Any]] = []
    try:
        for split, seeds in split_seeds.items():
            for seed in seeds:
                observation, _ = environment.reset(seed=seed)
                row = study_track_row(
                    _track_row(environment, observation, seed), task=task
                )
                row.update({"split": split, "track_seed": seed})
                rows.append(row)
    finally:
        environment.close()
    return rows


def archived_current_seeds(path: str | Path) -> dict[str, Any]:
    import torch

    source = Path(path)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    recorder = payload["environment_state"]
    wrapper = recorder.env.env
    seeds = [int(value) for value in wrapper.seeds_used]
    return {
        "checkpoint_path": str(source),
        "checkpoint_sha256": file_sha256(source),
        "completed_episodes": int(recorder.completed_episodes),
        "reset_seed_count": len(seeds),
        "reset_seeds": seeds,
    }


def build_report(
    *,
    count: int = 500,
    audit_path: str | Path = DEFAULT_AUDIT,
    archived_current_path: str | Path = DEFAULT_ARCHIVED_CURRENT,
) -> dict[str, Any]:
    if count < 500:
        raise ValueError("The frozen preflight requires at least 500 definitions")
    configuration = load_configuration()
    task = {
        "max_time_s": configuration.task.max_time_s,
        "max_speed_violation_m_s": configuration.task.max_speed_violation_m_s,
    }
    historical_raw = historical_track_fingerprints(11, count=count)
    current_raw = current_track_fingerprints(11, count=count)
    historical = [study_track_row(row, task=task) for row in historical_raw]
    current = [study_track_row(row, task=task) for row in current_raw]
    reserved = reserved_rows()
    reserved_by_hash = {row["canonical_sha256"]: row for row in reserved}

    def collisions(stream: list[dict[str, Any]]) -> list[dict[str, Any]]:
        found = []
        for row in stream:
            reserved_row = reserved_by_hash.get(row["canonical_sha256"])
            if reserved_row is not None:
                found.append(
                    {
                        "episode_index": row["episode_index"],
                        "stream_explicit_reset_seed": row.get("explicit_reset_seed"),
                        "reserved_split": reserved_row["split"],
                        "reserved_track_seed": reserved_row["track_seed"],
                        "canonical_sha256": row["canonical_sha256"],
                        "audit_sha256": row["audit_sha256"],
                    }
                )
        return found

    audit_source = Path(audit_path)
    audit = json.loads(audit_source.read_text(encoding="utf-8"))
    audit_seed11 = audit["rng_and_tracks"]["by_seed"]["11"]
    historical_first_100_match = historical_raw[:100] == audit_seed11[
        "historical_first_100_used_track_fingerprints"
    ]
    current_first_100_match = current_raw[:100] == audit_seed11[
        "current_first_100_used_track_fingerprints"
    ]

    archive = archived_current_seeds(archived_current_path)
    generated_current_seeds = [row["explicit_reset_seed"] for row in current_raw]
    archived_prefix = archive["reset_seeds"]
    archived_current_prefix_match = (
        generated_current_seeds[: len(archived_prefix)] == archived_prefix
    )
    reserved_seed_ids = {
        row["track_seed"] for row in reserved
    }
    current_reserved_seed_id_overlap = sorted(
        set(generated_current_seeds) & reserved_seed_ids
    )

    historical_collisions = collisions(historical)
    current_collisions = collisions(current)
    exact_stream_checks_pass = (
        historical_first_100_match
        and current_first_100_match
        and archived_current_prefix_match
        and historical[0]["canonical_sha256"] != current[0]["canonical_sha256"]
    )
    split_content_protection_pass = not (
        historical_collisions or current_collisions
    )
    main_training_authorized = (
        exact_stream_checks_pass and split_content_protection_pass
    )
    minimum_episode_count = (300_000 + 1_800 - 1) // 1_800
    unavoidable = [
        item
        for item in (*historical_collisions, *current_collisions)
        if item["episode_index"] < minimum_episode_count
    ]
    return {
        "schema_version": 1,
        "study": "constrained-v2-seed11-track-collector-factorial",
        "status": "EXECUTION_GATE_FAILED" if not main_training_authorized else "READY",
        "main_training_authorized": main_training_authorized,
        "reason": (
            "Exact frozen streams collide by canonical task/track content with "
            "Development track 2006 before the minimum possible episode count."
            if not split_content_protection_pass
            else None
        ),
        "simulator_transitions_consumed": 0,
        "audit_source": {
            "path": str(audit_source),
            "sha256": file_sha256(audit_source),
            "historical_first_100_match": historical_first_100_match,
            "current_first_100_match": current_first_100_match,
            "current_archive_match_recorded_by_audit": bool(
                audit_seed11["current_generated_seeds_match_checkpoint"]
            ),
        },
        "archived_current_source": {
            **archive,
            "generated_reset_seed_prefix_match": archived_current_prefix_match,
        },
        "stream_definition": {
            "historical": (
                "DenseDeadlineTaskWrapper initial_seed=11; continuous base-env "
                "np_random; used reset draws 0,2,4,... from FastCollector's "
                "terminal reset plus final reset"
            ),
            "current": (
                "default_rng(SeedSequence([11, 0x4D564531])); explicit integer "
                "per reset; reserved numeric split IDs rejected"
            ),
        },
        "checks": {
            "exact_stream_checks_pass": exact_stream_checks_pass,
            "streams_differ_at_episode_zero": (
                historical[0]["canonical_sha256"]
                != current[0]["canonical_sha256"]
            ),
            "current_reserved_numeric_seed_overlap": current_reserved_seed_id_overlap,
            "reserved_numeric_seed_protection_pass": (
                not current_reserved_seed_id_overlap
            ),
            "reserved_content_fingerprint_protection_pass": (
                split_content_protection_pass
            ),
            "minimum_episodes_at_300k_with_1800_step_cap": minimum_episode_count,
            "collisions_before_minimum_episode_count": unavoidable,
        },
        "initial_learner": initial_learner_hashes(),
        "collisions": {
            "historical": historical_collisions,
            "current": current_collisions,
        },
        "historical_first_500": historical,
        "current_first_500": current,
        "reserved_tracks": reserved,
        "execution": {
            "conditions_started": 0,
            "real_transitions": 0,
            "gradient_updates": 0,
            "development_episodes": 0,
            "validation_episodes": 0,
            "paper_episodes": 0,
            "preflight_runtime_benchmark_run": False,
            "preflight_runtime_benchmark_reason": (
                "Main-execution eligibility had already failed at the split-content "
                "protection gate; no simulator transitions were consumed."
            ),
        },
    }


def atomic_json(path: str | Path, payload: Any) -> Path:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument(
        "--archived-current", type=Path, default=DEFAULT_ARCHIVED_CURRENT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    report = build_report(
        count=args.count,
        audit_path=args.audit,
        archived_current_path=args.archived_current,
    )
    atomic_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "reason")}, indent=2))
    return 2 if not report["main_training_authorized"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
