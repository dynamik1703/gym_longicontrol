"""Generate the static content-split audit without taking simulator steps."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from math import comb
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.constrained_rl_v2.costs import optimistic_remaining_time_s
from benchmarks.requirement_conditioned.requirements import BalancedMarginSampler
from gym_longicontrol.domain.state import SimulationConfig
from gym_longicontrol.domain.track import Track

from .fingerprints import fingerprint_track
from .streams import (
    ALL_NUMERICALLY_RESERVED,
    DEVELOPMENT_SEEDS,
    PAPER_SEEDS,
    VALIDATION_SEEDS,
    collisions,
    continuous_tracks,
    duplicate_statistics,
    explicit_seed_stream,
    fingerprint_rows,
    track_from_seed,
)

HERE = Path(__file__).resolve().parent
CRITICAL_EPISODES = 500
TRAINING_SEEDS = (11, 29, 47)


def v2_semantics() -> dict[str, Any]:
    return {
        "observation": {
            "name": "longicontrol-core-eight-dimensional-v1",
            "fields": [
                "velocity",
                "acceleration",
                "current_speed_limit",
                "future_speed_limit_1",
                "future_speed_limit_2",
                "distance_to_transition_1",
                "distance_to_transition_2",
                "energy_factor",
            ],
            "normalization": (
                "affine to [0,1] using vehicle acceleration/velocity bounds, "
                "sensor_range_m, and energy_factor range [0,5]"
            ),
        },
        "dynamics": {
            "name": "longicontrol-native-dynamics-v1",
            "action_range": [-1.0, 1.0],
            "time_step_s": 0.1,
        },
        "objective": {
            "name": "negative-net-energy",
            "energy_scale_kwh": 0.25,
            "base_environment_reward_ignored": True,
        },
        "costs": {
            "names": ["speed_integral_m", "deadline_deficit_integral_s"],
            "deadline_form": "optimistic-speed-limit-deadline-deficit-integral",
            "deadline_normalization_s": 140.0,
            "terminal_failure_cost": False,
        },
        "feasibility": {
            "cost_limits": [0.0, 0.0],
            "maximum_speed_violation_m_s": 0.0,
            "maximum_time_s": 140.0,
        },
        "termination": {
            "physical": "position_m >= route_length_m",
            "outer_time_limit_steps": 1800,
            "training_budget_may_truncate": True,
        },
        "requirements": {
            "max_time_s": 140.0,
            "max_speed_violation_m_s": 0.0,
        },
    }


def scalar_semantics() -> dict[str, Any]:
    value = v2_semantics()
    value["objective"] = {
        "name": "scalar-benchmark-reward-v2",
        "source": "benchmarks/scalar_sac/v2_reward.py",
    }
    value["costs"] = {"exposed_to_learner": False}
    return value


def her_semantics() -> dict[str, Any]:
    value = v2_semantics()
    value["observation"] = {
        "name": "goal-conditioned-dict-observation-v1",
        "keys": ["observation", "achieved_goal", "desired_goal"],
    }
    value["objective"] = {
        "name": "canonical-sparse-goal-reward",
        "her_relabeling": True,
    }
    value["costs"] = {"exposed_to_learner": False}
    return value


def crl_semantics() -> dict[str, Any]:
    value = v2_semantics()
    value["observation"] = {
        "name": "requirement-aware-projected-goal-crl-v1",
        "policy_state_dim": 12,
        "projected_goal_dim": 3,
        "canonical_command": [1.0, 1.0, 1.0],
    }
    value["objective"] = {
        "name": "projected-future-contrastive-objective",
        "historical_environment_reward_recorded": True,
        "historical_environment_reward": {
            "component_order": ["forward", "energy", "jerk", "shock"],
            "weights": [1.0, 0.5, 1.0, 1.0],
        },
        "energy_role": "evaluation-only",
    }
    value["costs"] = {"exposed_to_learner": False}
    return value


def binary_semantics() -> dict[str, Any]:
    value = v2_semantics()
    value["objective"] = {
        "name": "binary-canonical-success",
        "success_reward": 1.0,
        "otherwise_reward": 0.0,
    }
    value["costs"] = {"exposed_to_learner": False}
    return value


def llm_semantics() -> dict[str, Any]:
    value = v2_semantics()
    value["objective"] = {
        "name": "llm-reward-search-final-winner",
        "candidate_id": "g1-c02",
        "source_sha256": (
            "714ed4124e98b2e12d1d89a72edb78d9ab1f04de8fd88e2e18bb74350417fd47"
        ),
        "source": "benchmarks/llm_reward/FINAL_REWARD.py",
    }
    value["costs"] = {"exposed_to_learner": False}
    return value


def requirement_semantics(track: Track, margin_s: float) -> dict[str, Any]:
    value = v2_semantics()
    t_min = optimistic_remaining_time_s(
        track, position_m=0.0, track_length_m=SimulationConfig().track_length_m
    )
    deadline = t_min + margin_s
    value["observation"] = {
        "name": "requirement-conditioned-ten-dimensional-v1",
        "base_features": 8,
        "additional_fields": ["max_time_s / 180", "elapsed_time_s / 180"],
    }
    value["costs"] = {
        "names": ["speed_integral_m", "deadline_deficit_integral_s"],
        "deadline_form": "optimistic-speed-limit-deadline-deficit-integral",
        "deadline_normalization": "per-realized-deadline",
    }
    value["feasibility"] = {
        "cost_limits": [0.0, 0.0],
        "maximum_speed_violation_m_s": 0.0,
        "maximum_time_s": deadline,
    }
    value["requirements"] = {
        "margin_s": margin_s,
        "optimistic_minimum_time_s": t_min,
        "max_time_s": deadline,
        "max_speed_violation_m_s": 0.0,
    }
    return value


def _reference_rows(
    seeds: Sequence[int], semantics: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return fingerprint_rows(
        [track_from_seed(seed) for seed in seeds],
        semantics,
        numeric_seeds=list(seeds),
    )


def _critical_streams() -> dict[str, list[dict[str, Any]]]:
    historical_draws = list(range(0, 2 * CRITICAL_EPISODES, 2))
    historical = fingerprint_rows(
        continuous_tracks(11, historical_draws),
        v2_semantics(),
        numeric_seeds=[11, *([None] * (CRITICAL_EPISODES - 1))],
        draw_indices=historical_draws,
    )
    current_seeds = explicit_seed_stream(
        11, domain_tag=0x4D564531, count=CRITICAL_EPISODES
    )
    current = fingerprint_rows(
        [track_from_seed(seed) for seed in current_seeds],
        v2_semantics(),
        numeric_seeds=current_seeds,
        draw_indices=list(range(CRITICAL_EPISODES)),
    )
    return {"T_HIST": historical, "T_CURRENT": current}


def _probability_report(
    rows: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    *,
    split: str,
    stream: str,
) -> dict[str, Any]:
    found = collisions(rows, reference, split=split, stream=stream)
    route_episode_count = len(
        {row["training_episode_index"] for row in found if row["route_sha256_match"]}
    )
    full_episode_count = len(
        {
            row["training_episode_index"]
            for row in found
            if row["full_task_sha256_match"]
        }
    )

    def metric(count: int) -> dict[str, Any]:
        rate = count / len(rows)
        return {
            "observed_matching_episodes": count,
            "empirical_probability": rate,
            "expected_collision_count": {
                str(size): rate * size for size in (100, 200, 300, 500)
            },
        }

    return {
        "route": metric(route_episode_count),
        "full_task": metric(full_episode_count),
    }


def _fixed_continuous_rows(
    seed: int,
    count: int,
    semantics: Mapping[str, Any],
    *,
    stride: int = 1,
    offset: int = 0,
) -> list[dict[str, Any]]:
    draws = [offset + stride * index for index in range(count)]
    return fingerprint_rows(
        continuous_tracks(seed, draws),
        semantics,
        numeric_seeds=[seed, *([None] * (count - 1))],
        draw_indices=draws,
    )


def _fixed_explicit_rows(
    seed: int,
    count: int,
    semantics: Mapping[str, Any],
    *,
    domain_tag: int,
    upper_bound: int = 2**31 - 1,
) -> list[dict[str, Any]]:
    reset_seeds = explicit_seed_stream(
        seed, domain_tag=domain_tag, count=count, upper_bound=upper_bound
    )
    return fingerprint_rows(
        [track_from_seed(reset_seed) for reset_seed in reset_seeds],
        semantics,
        numeric_seeds=reset_seeds,
        draw_indices=list(range(count)),
    )


def _requirement_rows(seed: int, count: int) -> list[dict[str, Any]]:
    draws = [1 + 3 * index for index in range(count)]
    tracks = continuous_tracks(seed, draws)
    sampler = BalancedMarginSampler((20.0, 40.0, 60.0), seed=90_000 + seed)
    semantics = [requirement_semantics(track, sampler.sample()) for track in tracks]
    return fingerprint_rows(
        tracks,
        semantics,
        numeric_seeds=[None] * count,
        draw_indices=draws,
    )


def _requirement_reference_rows(seeds: Sequence[int]) -> list[dict[str, Any]]:
    rows = []
    for seed in seeds:
        track = track_from_seed(seed)
        for margin in (20.0, 40.0, 60.0):
            fingerprint = fingerprint_track(
                track, requirement_semantics(track, margin), numeric_seed=seed
            )
            rows.append(
                {
                    "episode_index": len(rows),
                    "reset_seed": seed,
                    "rng_draw_index": None,
                    "requirement_margin_s": margin,
                    "route_sha256": fingerprint.route_sha256,
                    "full_task_sha256": fingerprint.full_task_sha256,
                }
            )
    return rows


def _integrity(route_count: int, full_count: int, reconstruction: str) -> str:
    if reconstruction != "EXACT":
        return "UNKNOWN"
    if full_count:
        return "FULL_TASK_OVERLAP"
    if route_count:
        return "ROUTE_OVERLAP_ONLY"
    return "CLEAN"


def _audit_study_runs(
    name: str,
    reconstruction: str,
    streams: Sequence[tuple[str, int, list[dict[str, Any]]]],
    development: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
    *,
    exact_scope: str,
    limitation: str | None = None,
) -> dict[str, Any]:
    run_results = []
    totals = Counter()
    for run_id, seed, rows in streams:
        dev = collisions(
            rows, development, split="Development", stream=f"{name}/{run_id}"
        )
        val = collisions(
            rows, validation, split="Validation", stream=f"{name}/{run_id}"
        )
        counts = {
            "development_route_collision_count": sum(
                bool(row["route_sha256_match"]) for row in dev
            ),
            "development_full_task_collision_count": sum(
                bool(row["full_task_sha256_match"]) for row in dev
            ),
            "validation_route_collision_count": sum(
                bool(row["route_sha256_match"]) for row in val
            ),
            "validation_full_task_collision_count": sum(
                bool(row["full_task_sha256_match"]) for row in val
            ),
        }
        totals.update(counts)
        run_results.append(
            {
                "run_id": run_id,
                "training_seed": seed,
                "training_tasks_with_transitions": len(rows),
                **counts,
                "collisions": [*dev, *val],
            }
        )
    return {
        "study": name,
        "reconstruction": reconstruction,
        "scope": exact_scope,
        "limitation": limitation,
        "runs": run_results,
        "totals": dict(totals),
        "development_integrity": _integrity(
            totals["development_route_collision_count"],
            totals["development_full_task_collision_count"],
            reconstruction,
        ),
        "validation_integrity": _integrity(
            totals["validation_route_collision_count"],
            totals["validation_full_task_collision_count"],
            reconstruction,
        ),
    }


def _study_audit() -> list[dict[str, Any]]:
    descriptors = {
        "v2": v2_semantics(),
        "scalar": scalar_semantics(),
        "her": her_semantics(),
        "crl": crl_semantics(),
        "binary": binary_semantics(),
        "llm": llm_semantics(),
    }
    references = {
        key: {
            "development": _reference_rows(DEVELOPMENT_SEEDS, descriptor),
            "validation": _reference_rows(VALIDATION_SEEDS, descriptor),
        }
        for key, descriptor in descriptors.items()
    }
    requirement_references = {
        "development": _requirement_reference_rows(DEVELOPMENT_SEEDS),
        "validation": _requirement_reference_rows(VALIDATION_SEEDS),
    }
    studies = []
    historical_counts = {11: 249, 29: 240, 47: 245}
    studies.append(
        _audit_study_runs(
            "historical Constrained V2",
            "PARTIAL",
            [
                (
                    f"seed-{seed}",
                    seed,
                    _fixed_continuous_rows(seed, count, descriptors["v2"], stride=2),
                )
                for seed, count in historical_counts.items()
            ],
            references["v2"]["development"],
            references["v2"]["validation"],
            exact_scope="source-exact current-runtime reconstruction",
            limitation=(
                "The run-producing commit plus NumPy, Torch, and Gymnasium versions "
                "were not recorded; these are not recovered historical task bytes."
            ),
        )
    )
    current_counts = {11: 223, 29: 231, 47: 241}
    studies.append(
        _audit_study_runs(
            "current MVE matched Real-Only",
            "EXACT",
            [
                (
                    f"matched_real_only-seed-{seed}",
                    seed,
                    _fixed_explicit_rows(
                        seed,
                        count,
                        descriptors["v2"],
                        domain_tag=0x4D564531,
                    ),
                )
                for seed, count in current_counts.items()
            ],
            references["v2"]["development"],
            references["v2"]["validation"],
            exact_scope="all persisted reset seeds used by the completed policies",
        )
    )
    h1_counts = {11: 250, 29: 234, 47: 239}
    studies.append(
        _audit_study_runs(
            "H=1 Physics MBRL",
            "EXACT",
            [
                (
                    f"physics-seed-{seed}",
                    seed,
                    _fixed_explicit_rows(
                        seed,
                        count,
                        descriptors["v2"],
                        domain_tag=0x4D42524C,
                    ),
                )
                for seed, count in h1_counts.items()
            ],
            references["v2"]["development"],
            references["v2"]["validation"],
            exact_scope="all completed real episodes reported by final heartbeats",
        )
    )
    requirement_counts = {11: 250, 29: 259, 47: 271}
    studies.append(
        _audit_study_runs(
            "Requirement-Conditioned RL",
            "PARTIAL",
            [
                (f"seed-{seed}", seed, _requirement_rows(seed, count))
                for seed, count in requirement_counts.items()
            ],
            requirement_references["development"],
            requirement_references["validation"],
            exact_scope="current-runtime reconstruction of used draws 1,4,7,...",
            limitation=(
                "Run artifacts record FSRL/Tianshou and episode counts but not the "
                "NumPy/Gymnasium versions or realized tracks."
            ),
        )
    )
    scalar_counts = {11: 185, 29: 170, 47: 178}
    studies.append(
        _audit_study_runs(
            "matched scalar SAC",
            "PARTIAL",
            [
                (
                    f"sac-seed-{seed}",
                    seed,
                    _fixed_continuous_rows(seed, count, descriptors["scalar"]),
                )
                for seed, count in scalar_counts.items()
            ],
            references["scalar"]["development"],
            references["scalar"]["validation"],
            exact_scope="current-runtime reconstruction through final active episode",
            limitation=(
                "SB3 version and episode count are persisted, but NumPy/Gymnasium "
                "versions and realized track identities are not."
            ),
        )
    )
    her_counts = {11: 170, 29: 170, 47: 167}
    studies.append(
        _audit_study_runs(
            "HER",
            "EXACT",
            [
                (
                    f"sac-her-seed-{seed}",
                    seed,
                    _fixed_continuous_rows(seed, count, descriptors["her"]),
                )
                for seed, count in her_counts.items()
            ],
            references["her"]["development"],
            references["her"]["validation"],
            exact_scope=(
                "pinned dependency/source execution and recorded completed plus "
                "partial episode counts"
            ),
        )
    )
    crl_counts = {
        (4, 11): 301,
        (4, 29): 278,
        (4, 47): 297,
        (16, 11): 291,
        (16, 29): 281,
        (16, 47): 262,
    }
    studies.append(
        _audit_study_runs(
            "CRL",
            "EXACT",
            [
                (
                    f"depth-{depth}-seed-{seed}",
                    seed,
                    _fixed_explicit_rows(
                        seed,
                        count,
                        descriptors["crl"],
                        domain_tag=0x43524C31,
                        upper_bound=np.iinfo(np.uint32).max,
                    ),
                )
                for (depth, seed), count in crl_counts.items()
            ],
            references["crl"]["development"],
            references["crl"]["validation"],
            exact_scope="all manifest-recorded training-track seeds used per policy",
        )
    )
    binary_counts = {11: 207, 29: 179, 47: 218}
    studies.append(
        _audit_study_runs(
            "Binary Success",
            "PARTIAL",
            [
                (
                    f"seed-{seed}",
                    seed,
                    _fixed_continuous_rows(seed, count, descriptors["binary"]),
                )
                for seed, count in binary_counts.items()
            ],
            references["binary"]["development"],
            references["binary"]["validation"],
            exact_scope="current-runtime reconstruction of completed episodes",
            limitation=(
                "Completed episode counts and SB3 are recorded, but the exact "
                "NumPy/Gymnasium runtime and realized tracks are not."
            ),
        )
    )
    llm_counts = {11: 260, 29: 167, 47: 167}
    studies.append(
        _audit_study_runs(
            "LLM Reward Search final policies",
            "EXACT",
            [
                (
                    f"seed-{seed}",
                    seed,
                    _fixed_continuous_rows(seed, count, descriptors["llm"]),
                )
                for seed, count in llm_counts.items()
            ],
            references["llm"]["development"],
            references["llm"]["validation"],
            exact_scope=(
                "frozen final reward/source, recorded dependency versions, and "
                "saved SB3 episode counters through the active final episode"
            ),
        )
    )
    return studies


def _generator_space() -> dict[str, Any]:
    speed_paths_by_transition_count = {}
    contributions = {}
    states = {speed: 1 for speed in range(30, 81, 10)}
    total = 0
    for transition_count in range(1, 10):
        next_states: dict[int, int] = {}
        for previous, paths in states.items():
            for speed in range(max(20, previous - 40), min(100, previous + 40) + 1, 10):
                if speed != previous:
                    next_states[speed] = next_states.get(speed, 0) + paths
        states = next_states
        speed_paths = sum(states.values())
        contribution = comb(9, transition_count) * speed_paths
        speed_paths_by_transition_count[str(transition_count)] = speed_paths
        contributions[str(transition_count)] = contribution
        total += contribution
    return {
        "nominal_seed_namespaces": {
            "continuous_initial_seed": (
                "unbounded Python integer accepted by SeedSequence"
            ),
            "explicit_mve_and_mbrl": 2**31 - 1,
            "explicit_crl": int(np.iinfo(np.uint32).max),
        },
        "exact_route_content_count": total,
        "v2_full_task_content_count": total,
        "speed_sequence_count_by_transition_count": speed_paths_by_transition_count,
        "route_count_contribution_by_transition_count": contributions,
        "transition_count_probability": {
            "1": 0.2,
            **{str(value): 0.1 for value in range(2, 10)},
        },
        "mechanism": (
            "Nine 100 m transition sites, a clipped discrete transition-count draw, "
            "and a 20..100 km/h grid with adjacent changes limited to 40 km/h."
        ),
        "interpretation": (
            "The support is finite and highly non-uniform: 20% of tasks use one "
            "transition but that stratum contains only 351 routes."
        ),
    }


def _clean_candidates(
    critical: Mapping[str, Sequence[Mapping[str, Any]]],
    development: Sequence[Mapping[str, Any]],
    validation: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    forbidden_routes = {
        row["route_sha256"]
        for rows in (*critical.values(), development, validation)
        for row in rows
    }
    forbidden_full = {
        row["full_task_sha256"]
        for rows in (*critical.values(), development, validation)
        for row in rows
    }
    training_numeric = {
        int(row["reset_seed"])
        for rows in critical.values()
        for row in rows
        if row["reset_seed"] is not None
    }
    candidates = []
    seed = 9_000_000
    while len(candidates) < 18:
        if seed in ALL_NUMERICALLY_RESERVED or seed in training_numeric:
            seed += 1
            continue
        fingerprint = fingerprint_track(
            track_from_seed(seed), v2_semantics(), numeric_seed=seed
        )
        if (
            fingerprint.route_sha256 not in forbidden_routes
            and fingerprint.full_task_sha256 not in forbidden_full
            and fingerprint.route_sha256
            not in {row["route_sha256"] for row in candidates}
        ):
            candidates.append(
                {
                    "numeric_seed": seed,
                    "route_sha256": fingerprint.route_sha256,
                    "full_task_sha256": fingerprint.full_task_sha256,
                    "evaluated": False,
                }
            )
        seed += 1
    return candidates


def generate() -> dict[str, Any]:
    # Paper content is intentionally absent from every construction below. This is the
    # only permitted paper check: its numeric IDs are members of the rejection set.
    if not set(PAPER_SEEDS) <= ALL_NUMERICALLY_RESERVED:
        raise AssertionError("Paper numeric seeds are not fully excluded")
    critical = _critical_streams()
    development = _reference_rows(DEVELOPMENT_SEEDS, v2_semantics())
    validation = _reference_rows(VALIDATION_SEEDS, v2_semantics())
    critical_results = {}
    all_collisions = []
    for name, rows in critical.items():
        development_collisions = collisions(
            rows, development, split="Development", stream=name
        )
        validation_collisions = collisions(
            rows, validation, split="Validation", stream=name
        )
        all_collisions.extend(development_collisions)
        all_collisions.extend(validation_collisions)
        critical_results[name] = {
            "stream_definition": (
                {
                    "initial_seed": 11,
                    "used_rng_draw_indices": "0,2,4,...,998",
                    "reset_seed_after_episode_zero": None,
                }
                if name == "T_HIST"
                else {
                    "seed_sequence_entropy": [11, 0x4D564531],
                    "explicit_reset_seeds": True,
                    "excluded_numeric_seeds_include_paper": True,
                }
            ),
            "episodes": rows,
            "duplicate_statistics": duplicate_statistics(rows),
            "development_collisions": development_collisions,
            "validation_collisions": validation_collisions,
            "empirical_probabilities": {
                "development": _probability_report(
                    rows,
                    development,
                    split="Development",
                    stream=name,
                ),
                "validation": _probability_report(
                    rows, validation, split="Validation", stream=name
                ),
            },
        }
    observed_known = {
        (row["training_stream"], row["training_episode_index"], row["reference_seed"])
        for row in all_collisions
        if row["split"] == "Development"
    }
    expected_known = {
        ("T_HIST", 158, 2006),
        ("T_CURRENT", 163, 2006),
        ("T_CURRENT", 242, 2006),
    }
    if not expected_known <= observed_known:
        raise AssertionError("Known Development-2006 collision regression failed")
    return {
        "schema_version": 1,
        "audit_type": "static-content-only",
        "source_commit": "c728da04e9028a0cbb75d715050c688f19bae1e1",
        "fingerprint_schema": "longicontrol-content-fingerprint-v1",
        "longicontrol_simulator_transitions": 0,
        "verification_test_simulator_transitions": {
            "total": 476,
            "scope": "pre-existing environment/dynamics unit tests only",
            "breakdown": {
                "gymnasium_contract_checks_and_explicit_steps": 210,
                "seed_replay": 200,
                "energy_and_context_semantics": 60,
                "finish_and_time_limit": 3,
                "direct_dynamics_unit_checks": 2,
                "headless_registration_subprocess": 1,
            },
            "development_validation_or_paper_tracks": 0,
        },
        "total_simulator_transitions_including_verification": 476,
        "policy_training_transitions": 0,
        "policy_evaluation_episodes": 0,
        "development_instantiated_statically": list(DEVELOPMENT_SEEDS),
        "validation_instantiated_statically": list(VALIDATION_SEEDS),
        "paper": {
            "numeric_seeds": list(PAPER_SEEDS),
            "numeric_exclusion_verified": True,
            "instantiated": False,
            "fingerprinted": False,
            "content_inspected": False,
        },
        "critical_streams": critical_results,
        "known_2006_regression_verified": True,
        "generator_space": _generator_space(),
        "completed_studies": _study_audit(),
        "recommended_policy": "R",
        "recommended_repair": "Option 1 — preserve streams and use a new clean set",
        "proposed_diagnostic_set": {
            "namespace": "sequential integers beginning at 9,000,000",
            "status": "DESIGNED_NOT_USED",
            "clean_against": [
                "T_HIST first 500 tasks",
                "T_CURRENT first 500 tasks",
                "Development 2000-2008",
                "Validation 3000-3008",
            ],
            "candidates": _clean_candidates(critical, development, validation),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    args = parser.parse_args(argv)
    payload = generate()
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
