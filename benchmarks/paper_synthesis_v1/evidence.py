"""Reconcile immutable study records into paper-synthesis registries."""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import statistics
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]
NOT_RECORDED = "NOT_RECORDED"


def read_json(relative: str) -> dict[str, Any]:
    return json.loads((REPOSITORY_ROOT / relative).read_text(encoding="utf-8"))


def method(
    method_id: str,
    name: str,
    version: str,
    comparison_family: str,
    algorithm_family: str,
    branch: str,
    source_commit: str,
    *,
    seeds: list[int] | str = NOT_RECORDED,
    transitions: int | str = NOT_RECORDED,
    updates: int | str = NOT_RECORDED,
    synthetic: int | str = 0,
    counterfactual_steps: int | str = 0,
    panel: str = NOT_RECORDED,
    split_hash: str = NOT_RECORDED,
    routes: int | str = NOT_RECORDED,
    pairs: int | str = NOT_RECORDED,
    rsr: tuple[int, int] | None = None,
    completion: int | str = NOT_RECORDED,
    deadline: int | str = NOT_RECORDED,
    speed: int | str = NOT_RECORDED,
    energy: float | str | None = NOT_RECORDED,
    travel_time: float | str = NOT_RECORDED,
    observation: str = "8D causal core observation",
    action: str = "continuous native action in [-1,1] at 10 Hz",
    physics: str = "environment dynamics only",
    hidden_actor: bool | str = False,
    privileged_critic: bool | str = False,
    objective: str = NOT_RECORDED,
    safety: str = "none",
    config_hash: str = NOT_RECORDED,
    stream_hash: str = NOT_RECORDED,
    classification: str = NOT_RECORDED,
    timing: str = "PROSPECTIVE",
    basis: str = "CONTEXTUAL_ONLY",
    limitations: str = "",
    incidents: str = "none recorded",
    checkpoint: str = NOT_RECORDED,
    evidence_paths: list[str] | None = None,
) -> dict[str, Any]:
    numerator, denominator = rsr if rsr is not None else (NOT_RECORDED,) * 2
    return {
        "method_id": method_id,
        "method_name": name,
        "version": version,
        "comparison_family": comparison_family,
        "algorithm_family": algorithm_family,
        "branch": branch,
        "source_commit": source_commit,
        "frozen_source_config_hash": config_hash,
        "observation_contract": observation,
        "action_semantics": action,
        "physics_information": physics,
        "actor_uses_hidden_information": hidden_actor,
        "critic_uses_privileged_training_information": privileged_critic,
        "reward_cost_formulation": objective,
        "safety_viability_mechanism": safety,
        "training_seeds": seeds,
        "training_task_stream_hash": stream_hash,
        "real_native_training_transitions": transitions,
        "gradient_updates": updates,
        "model_synthetic_transitions": synthetic,
        "counterfactual_simulator_steps": counterfactual_steps,
        "evaluation_panel": panel,
        "evaluation_split_hash": split_hash,
        "distinct_evaluation_routes": routes,
        "completed_controller_task_pairs": pairs,
        "rsr_numerator": numerator,
        "rsr_denominator": denominator,
        "completion_numerator": completion,
        "deadline_compliance_numerator": deadline,
        "speed_compliance_numerator": speed,
        "feasible_only_signed_energy_mean_kwh": energy,
        "travel_time_mean_s": travel_time,
        "scientific_classification": classification,
        "evaluation_timing": timing,
        "comparison_basis": basis,
        "limitations": limitations,
        "integrity_incidents": incidents,
        "checkpoint_availability": checkpoint,
        "authoritative_evidence_paths": evidence_paths or [],
    }


def method_rows() -> list[dict[str, Any]]:
    task_branch = "research/task-specification-benchmarks"
    task_commit = "56d0bc94341ffcfda9cee29596489977be8b82ea"
    rows = [
        method(
            "scalar_custom_grid",
            "Historical custom Scalar SAC reward grid",
            "140-second factorial",
            "A_STANDARD_REWARD_RL",
            "custom SAC",
            task_branch,
            task_commit,
            seeds=[11, 29, 47],
            transitions=2_640_000,
            panel="historical holdout 1000-1008",
            routes=9,
            pairs=216,
            rsr=(5, 216),
            objective="8 scalar reward settings",
            config_hash="8b2e631873810b9a7b054b12e94060697b606eb34db98b9949b0a39937035d87",
            classification="ALMOST_ALL_REWARD_SETTINGS_FAIL",
            limitations="24 policies share nine evaluation routes; sparse feasible energy",
            evidence_paths=["benchmarks/scalar_sac/RESULTS.md", "benchmarks/scalar_sac/results-140s.json"],
        ),
        method(
            "sb3_sac_v2b",
            "Scalar SB3 SAC",
            "V2-B Balanced",
            "A_STANDARD_REWARD_RL",
            "SB3 SAC",
            task_branch,
            task_commit,
            seeds=[11, 29, 47],
            transitions=900_000,
            updates=899_697,
            panel="Validation 3000-3008",
            routes=9,
            pairs=27,
            rsr=(5, 27),
            completion=5,
            deadline=5,
            speed=27,
            objective="fixed shaped scalar reward",
            classification="WEAK_SEED_DEPENDENT_SCALAR_BASELINE",
            basis="PAIRED_SAME_TASKS",
            limitations="one seed contributes four of five successes",
            evidence_paths=["benchmarks/scalar_sb3/RESULTS.md", "benchmarks/scalar_sb3/results.json"],
        ),
        method(
            "sb3_ppo_v2b",
            "Scalar SB3 PPO",
            "V2-B Balanced",
            "A_STANDARD_REWARD_RL",
            "SB3 PPO",
            task_branch,
            task_commit,
            seeds=[11, 29, 47],
            transitions=900_000,
            updates=4380,
            panel="Validation 3000-3008",
            routes=9,
            pairs=27,
            rsr=(0, 27),
            completion=0,
            deadline=0,
            speed=27,
            objective="fixed shaped scalar reward",
            classification="ZERO_FULL_BRAKING",
            basis="PAIRED_SAME_TASKS",
            evidence_paths=["benchmarks/scalar_sb3/RESULTS.md", "benchmarks/scalar_sb3/results.json"],
        ),
        method(
            "scalar_action_repeat_5",
            "Scalar SAC action-repeat",
            "credit condition C",
            "A_STANDARD_REWARD_RL",
            "SB3 SAC",
            task_branch,
            task_commit,
            seeds=[11, 29, 47],
            transitions=900_000,
            updates="179825 total agent-decision updates",
            panel="Validation 3000-3008",
            routes=9,
            pairs=27,
            rsr=(9, 27),
            completion=19,
            deadline=17,
            speed=15,
            action="continuous action held for up to five native steps",
            objective="V2-B scalar reward; gamma 0.99",
            config_hash="79f269ac6b0eb9fc2a49fc30267c57332f6f1a267d01452f3c14bb887a800481",
            classification="NO_MATERIAL_ROBUSTNESS_IMPROVEMENT",
            basis="PAIRED_SAME_TASKS",
            limitations="decision frequency and update count both change",
            evidence_paths=["benchmarks/credit_assignment/RESULTS.md", "benchmarks/credit_assignment/results.json"],
        ),
        method(
            "binary_success_sac",
            "Binary Success Reward SAC",
            "V1",
            "A_STANDARD_REWARD_RL",
            "SB3 SAC",
            task_branch,
            task_commit,
            seeds=[11, 29, 47],
            transitions=900_000,
            panel="Validation 3000-3008",
            routes=9,
            pairs=27,
            rsr=(0, 27),
            completion=0,
            deadline=0,
            speed=27,
            objective="terminal 1 iff exact task feasibility, otherwise 0",
            classification="GATE_D_NO_ESCAPE_FROM_ZERO",
            basis="PAIRED_SAME_TASKS",
            evidence_paths=["benchmarks/binary_reward/RESULTS.md", "benchmarks/binary_reward/results.json"],
        ),
        method(
            "llm_reward_sac",
            "Frozen LLM reward winner",
            "g1-c02",
            "A_STANDARD_REWARD_RL",
            "SB3 SAC",
            "research/llm-reward-search",
            "46b48d129a1bad4d989a5e676d7a538ed7e2eaf6",
            seeds=[11, 29, 47],
            transitions=900_000,
            updates=899_700,
            panel="Validation 3000-3008",
            routes=9,
            pairs=27,
            rsr=(1, 27),
            completion=8,
            deadline=8,
            speed=20,
            energy=0.197414,
            objective="immutable LLM-generated scalar reward",
            config_hash="7f1da03f98c8d1bd7f3607710ca62bab82f19f86670cd8b27d5458d83f28ca4e",
            classification="SPARSE_NONROBUST_SUCCESS",
            basis="PAIRED_SAME_TASKS",
            limitations="one reward search; screening used seed 11; one feasible episode",
            evidence_paths=["benchmarks/llm_reward/FINAL_RESULTS.md", "benchmarks/llm_reward/final-results.json"],
        ),
        method(
            "constrained_v1",
            "Explicit constrained RL",
            "V1",
            "B_EXPLICIT_CONSTRAINTS",
            "FSRL SAC-Lagrangian",
            task_branch,
            task_commit,
            seeds=[11, 29, 47],
            transitions=900_000,
            updates=90_000,
            panel="Validation 3000-3008",
            routes=9,
            pairs=27,
            rsr=(0, 27),
            completion=0,
            deadline=0,
            speed=27,
            objective="negative signed energy; speed integral and sparse task-failure costs",
            classification="CONSTRAINTS_SATISFIED_BY_STANDSTILL",
            basis="PAIRED_SAME_TASKS",
            evidence_paths=["benchmarks/constrained_rl/RESULTS.md", "benchmarks/constrained_rl/results.json"],
        ),
        method(
            "constrained_v2",
            "Dense-deadline constrained RL",
            "V2",
            "B_EXPLICIT_CONSTRAINTS",
            "FSRL SAC-Lagrangian",
            task_branch,
            task_commit,
            seeds=[11, 29, 47],
            transitions=900_000,
            updates=90_000,
            panel="Validation 3000-3008",
            routes=9,
            pairs=27,
            rsr=(21, 27),
            completion=27,
            deadline=27,
            speed=21,
            energy=0.18961,
            objective="negative signed energy; speed and dense deadline-deficit costs",
            classification="LEARNABLE_NOT_SEED_ROBUST",
            basis="PAIRED_SAME_TASKS",
            checkpoint="LOCAL_PRESENT; SHA-256 newly audited, not recorded in original result",
            limitations="speed compliance misses preregistered robustness criterion",
            evidence_paths=["benchmarks/constrained_rl_v2/RESULTS.md", "benchmarks/constrained_rl_v2/results.json"],
        ),
        method(
            "sameinfo_scalar",
            "Same-Information Scalar SAC",
            "V1",
            "B_EXPLICIT_CONSTRAINTS",
            "FSRL-compatible scalar SAC",
            "research/same-information-scalar-vs-constrained",
            "78a50b7ad866beb72a93f06cb1c5d6bcaa25cc12",
            seeds=[11, 29, 47],
            transitions=900_000,
            updates=90_000,
            panel="SAMEINFO_VALIDATION_V1",
            split_hash="d8a658e96b0090f8e0cdcdfdec3b291329045dc3dac9541e0603f34d916bf10a",
            routes=18,
            pairs=54,
            rsr=(0, 54),
            completion=0,
            deadline=0,
            speed=53,
            objective="same raw energy/speed/deadline information as fixed scalarization",
            config_hash="96c587c9b14c0238257be60983bc9db7951fdabe2cebd867d5b9b13627e8c1f2",
            classification="CONSTRAINT_AVOIDING_CRAWL",
            basis="PAIRED_SAME_STREAM",
            checkpoint="LOCAL_PRESENT_HASH_MATCHES_RECORDED",
            evidence_paths=["benchmarks/same_information_scalar_vs_constrained/RESULTS.md", "benchmarks/same_information_scalar_vs_constrained/results.json"],
        ),
        method(
            "sameinfo_constrained",
            "Matched Constrained RL reference",
            "V2 checkpoints on SAMEINFO_VALIDATION_V1",
            "B_EXPLICIT_CONSTRAINTS",
            "FSRL SAC-Lagrangian",
            "research/same-information-scalar-vs-constrained",
            "78a50b7ad866beb72a93f06cb1c5d6bcaa25cc12",
            seeds=[11, 29, 47],
            transitions=900_000,
            updates=90_000,
            panel="SAMEINFO_VALIDATION_V1",
            split_hash="d8a658e96b0090f8e0cdcdfdec3b291329045dc3dac9541e0603f34d916bf10a",
            routes=18,
            pairs=54,
            rsr=(23, 54),
            completion=44,
            deadline=26,
            speed=46,
            energy=0.190465,
            objective="negative energy with explicit speed/deadline costs",
            classification="CONSTRAINED_ROBUST_ADVANTAGE",
            basis="PAIRED_SAME_STREAM",
            checkpoint="REFERENCE_HASHES_RECORDED; executable copies not present in this worktree",
            limitations="algorithmic differences extend beyond scalar vs multipliers",
            evidence_paths=["benchmarks/same_information_scalar_vs_constrained/RESULTS.md", "benchmarks/same_information_scalar_vs_constrained/results.json"],
        ),
        method(
            "requirement_conditioned",
            "Requirement-Conditioned SAC",
            "V1",
            "C_FUTURE_GOAL_CONDITIONED",
            "SB3 SAC",
            task_branch,
            task_commit,
            seeds=[11, 29, 47],
            transitions=900_000,
            panel="five-requirement Validation matrix on 3000-3008",
            routes=9,
            pairs=135,
            rsr=(48, 135),
            objective="scalar reward conditioned on deadline margin",
            classification="RESPONSIVE_BUT_UNRELIABLE",
            basis="PAIRED_SAME_TASKS",
            limitations="canonical 140 s subset is 12/27; matrix tasks differ by requirement",
            evidence_paths=["benchmarks/requirement_conditioned/RESULTS.md", "benchmarks/requirement_conditioned/results.json"],
        ),
        method(
            "goal_sac_no_her",
            "Goal-conditioned SAC",
            "without HER",
            "C_FUTURE_GOAL_CONDITIONED",
            "goal-conditioned SAC",
            "research/goal-conditioned-her",
            "a811047eb587b2586f6dc88aadda1e2eacbd2ab7",
            seeds=[11, 29, 47],
            transitions=900_000,
            panel="Validation 3000-3008",
            routes=9,
            pairs=27,
            rsr=(0, 27),
            completion=0,
            deadline=0,
            speed=27,
            objective="goal-conditioned reward",
            classification="ZERO",
            basis="PAIRED_SAME_TASKS",
            evidence_paths=["benchmarks/goal_conditioned/RESULTS.md", "benchmarks/goal_conditioned/results.json"],
        ),
        method(
            "goal_sac_her",
            "Goal-conditioned SAC",
            "with HER",
            "C_FUTURE_GOAL_CONDITIONED",
            "goal-conditioned SAC + HER",
            "research/goal-conditioned-her",
            "a811047eb587b2586f6dc88aadda1e2eacbd2ab7",
            seeds=[11, 29, 47],
            transitions=900_000,
            panel="Validation 3000-3008",
            routes=9,
            pairs=27,
            rsr=(0, 27),
            completion=0,
            deadline=0,
            speed=27,
            objective="goal-conditioned reward with relabeling",
            classification="ZERO",
            basis="PAIRED_SAME_TASKS",
            limitations="virtual relabels are not independent physical successes",
            evidence_paths=["benchmarks/goal_conditioned/RESULTS.md", "benchmarks/goal_conditioned/results.json"],
        ),
        method(
            "gcsl",
            "Goal-Conditioned Supervised Learning",
            "canonical V1",
            "C_FUTURE_GOAL_CONDITIONED",
            "GCSL",
            "research/gcsl-canonical",
            "5ada9402e3a2d627a7ea417cf1ffc23e1ab3789e",
            seeds=[11, 29, 47],
            transitions=900_000,
            updates=21_750,
            panel="GCSL_VALIDATION_V1",
            split_hash="d808fcc361be35f0c585003278f200d4013cc03718b0ceb318fbe6d55195a159",
            routes=18,
            pairs=54,
            rsr=(0, 54),
            completion=52,
            deadline=52,
            speed=1,
            observation="12D goal-conditioned actor state",
            objective="self-generated hindsight action likelihood",
            classification="GCSL_ZERO",
            basis="PAIRED_SAME_TASKS",
            checkpoint="LOCAL_PRESENT; recorded hashes available",
            limitations="constrained reference is contextual, not algorithm-matched",
            evidence_paths=["benchmarks/gcsl_canonical/RESULTS.md", "benchmarks/gcsl_canonical/results.json"],
        ),
        method(
            "contrastive_depth4",
            "Projected-goal Contrastive RL",
            "depth 4",
            "C_FUTURE_GOAL_CONDITIONED",
            "contrastive RL",
            "research/contrastive-rl",
            "cae33ff317757e24182955569b698ed29448683c",
            seeds=[11, 29, 47],
            transitions=900_000,
            updates=21_750,
            panel="Validation 3000-3008",
            routes=9,
            pairs=27,
            rsr=(0, 27),
            completion=17,
            deadline=17,
            speed=0,
            objective="projected requirement-aware contrastive goal",
            classification="NO_ROBUST_TRANSFER",
            basis="PAIRED_SAME_TASKS",
            checkpoint="LOCAL_PRESENT_HASH_MATCHES_RECORDED",
            evidence_paths=["benchmarks/contrastive_rl/RESULTS.md", "benchmarks/contrastive_rl/results.json"],
        ),
        method(
            "contrastive_depth16",
            "Projected-goal Contrastive RL",
            "depth 16",
            "C_FUTURE_GOAL_CONDITIONED",
            "contrastive RL",
            "research/contrastive-rl",
            "cae33ff317757e24182955569b698ed29448683c",
            seeds=[11, 29, 47],
            transitions=900_000,
            updates=21_750,
            panel="Validation 3000-3008",
            routes=9,
            pairs=27,
            rsr=(1, 27),
            completion=19,
            deadline=19,
            speed=2,
            energy=0.13555,
            objective="projected requirement-aware contrastive goal",
            classification="NO_REPEATABLE_DEPTH_GAIN",
            basis="PAIRED_SAME_TASKS",
            checkpoint="LOCAL_PRESENT_HASH_MATCHES_RECORDED",
            evidence_paths=["benchmarks/contrastive_rl/RESULTS.md", "benchmarks/contrastive_rl/results.json"],
        ),
        method(
            "learned_dynamics_mbrl",
            "Mixed learned-dynamics MBRL",
            "terminated attempt",
            "D_MODEL_BASED_RL",
            "SAC-Lagrangian + learned ensemble",
            "research/model-based-rl",
            "e0a6b7a2068eb0e0f9917831ff420fd7bd10aa9a",
            seeds=[11],
            transitions=60_500,
            updates=6050,
            synthetic=507_500,
            panel="Development 2000-2008 at 50k only",
            routes=9,
            pairs=9,
            rsr=(0, 9),
            completion=0,
            speed=9,
            objective="mixed real/synthetic constrained targets",
            classification="TERMINATED_RESOURCE_INFEASIBLE",
            timing="ABORTED_BEFORE_FINAL_ENDPOINT",
            basis="NOT_COMPARABLE",
            limitations="no 300k endpoint; five planned policies not started",
            evidence_paths=["benchmarks/model_based_rl/TERMINATION.md", "benchmarks/model_based_rl/termination.json"],
        ),
        method(
            "physics_model_mbrl",
            "Exact one-step Physics-Dynamics MBRL",
            "V1",
            "D_MODEL_BASED_RL",
            "SAC-Lagrangian + exact physics augmentation",
            "research/physics-model-mbrl",
            "8e5ebf6fb241f434781c0148c122a84c1ab4c35b",
            seeds=[11, 29, 47],
            transitions=900_000,
            updates=90_000,
            synthetic=8_707_500,
            panel="Validation 3000-3008",
            routes=9,
            pairs=27,
            rsr=(9, 27),
            completion=20,
            deadline=16,
            speed=18,
            energy=0.20377195538338214,
            objective="constrained RL with exact one-step model augmentation",
            classification="NO_MATERIAL_IMPROVEMENT",
            basis="CONTEXTUAL_ONLY",
            limitations="not a one-factor control against Constrained V2",
            evidence_paths=["benchmarks/physics_model_mbrl/RESULTS.md", "benchmarks/physics_model_mbrl/results.json"],
        ),
        method(
            "mve_real_only",
            "Multi-step study real-only control",
            "V1",
            "D_MODEL_BASED_RL",
            "SAC-Lagrangian",
            "research/multistep-physics-mbrl",
            "ebb4402f1bf5f457b265533a0ef95707c51f2a9b",
            seeds=[11, 29, 47],
            transitions=900_000,
            updates=90_000,
            panel="Validation 3000-3008",
            routes=9,
            pairs=27,
            rsr=(10, 27),
            completion=16,
            deadline=10,
            speed=27,
            energy=0.170534,
            objective="matched real-only constrained targets",
            classification="MATCHED_CONTROL",
            basis="PAIRED_SAME_STREAM",
            evidence_paths=["benchmarks/multistep_physics_mbrl/RESULTS.md", "benchmarks/multistep_physics_mbrl/results.json"],
        ),
        method(
            "physics_mve",
            "Multi-Step Physics MVE",
            "V1",
            "D_MODEL_BASED_RL",
            "SAC-Lagrangian + exact multi-step MVE",
            "research/multistep-physics-mbrl",
            "ebb4402f1bf5f457b265533a0ef95707c51f2a9b",
            seeds=[11, 29, 47],
            transitions=900_000,
            updates=90_000,
            panel="Validation 3000-3008",
            routes=9,
            pairs=27,
            rsr=(0, 27),
            completion=9,
            deadline=9,
            speed=14,
            objective="matched constrained targets plus exact multi-step expansion",
            classification="DEGRADATION_AND_UNSTABLE",
            basis="PAIRED_SAME_STREAM",
            limitations="seed 47 crossed numerical-stability threshold",
            evidence_paths=["benchmarks/multistep_physics_mbrl/RESULTS.md", "benchmarks/multistep_physics_mbrl/results.json"],
        ),
        method(
            "physical_oracle",
            "Full-route privileged physical reference",
            "V1",
            "E_PHYSICS_REFERENCE",
            "non-learning optimization",
            "research/physical-feasibility-reference",
            "3d9e8bcb72ecf4a01be192036df8f00c5ac9de8e",
            seeds="none",
            transitions=0,
            updates=0,
            panel="PHYSICS_REFERENCE_HOLDOUT_V1",
            split_hash="c02c4889333eaf3526277b10bc5c33101ac1aa97f9de836379088c661e72a30b",
            routes=36,
            pairs=36,
            rsr=(36, 36),
            completion=36,
            deadline=36,
            speed=36,
            energy=0.15266159372602325,
            travel_time=131.01388888888573,
            physics="exact dynamics and complete generated route",
            hidden_actor=True,
            objective="lexicographic feasibility then energy refinement",
            safety="full-route physical optimization",
            config_hash="f63717504a810c75d7080388aeb159505cf667ad13248dbe4a53ec0c4bb94cd1",
            classification="PHYSICALLY_SOLVABLE_REFERENCE",
            basis="PAIRED_SAME_TASKS",
            limitations="privileged contextual reference; not globally optimal",
            evidence_paths=["benchmarks/physical_feasibility_reference/RESULTS.md", "benchmarks/physical_feasibility_reference/holdout_results.json"],
        ),
        method(
            "physical_causal",
            "Observation-limited causal physical reference",
            "V1",
            "E_PHYSICS_REFERENCE",
            "non-learning physical controller",
            "research/physical-feasibility-reference",
            "3d9e8bcb72ecf4a01be192036df8f00c5ac9de8e",
            seeds="none",
            transitions=0,
            updates=0,
            panel="PHYSICS_REFERENCE_HOLDOUT_V1",
            split_hash="c02c4889333eaf3526277b10bc5c33101ac1aa97f9de836379088c661e72a30b",
            routes=36,
            pairs=36,
            rsr=(36, 36),
            completion=36,
            deadline=36,
            speed=36,
            energy=0.20585681935254208,
            travel_time=91.54166666666573,
            physics="ordinary observation, exact dynamics, propagated causal state/history",
            objective="causal physical feasibility controller",
            safety="braking envelope and physical optimization",
            config_hash="f63717504a810c75d7080388aeb159505cf667ad13248dbe4a53ec0c4bb94cd1",
            classification="CAUSAL_REFERENCE_STRONG_ON_TESTED_TASKS",
            basis="PAIRED_SAME_TASKS",
            limitations="36-route panel does not prove population-wide reliability",
            evidence_paths=["benchmarks/physical_feasibility_reference/RESULTS.md", "benchmarks/physical_feasibility_reference/holdout_results.json"],
        ),
    ]

    hybrid_common = dict(
        comparison_family="F_PHYSICS_SHIELDED_LEARNING",
        algorithm_family="SAC high-level target-speed policy + physical Shield",
        branch="research/hybrid-rl-validation-v2",
        source_commit="1c63ec85673ad0f3bf4afb4b05aa4f182c353ecc",
        transitions=300_000,
        updates=30_000,
        panel="HYBRID_VALIDATION_V2",
        split_hash="99a863fb423e2ceb5ec0641bb940fab21ff7eb86a1466886da8707862a015e74",
        routes=36,
        pairs=36,
        rsr=(35, 36),
        completion=36,
        deadline=35,
        speed=36,
        observation="8D causal observation plus causal controller state",
        action="macro target speed; Shield emits native actions",
        physics="exact dynamics in Shield; actor has no hidden route",
        objective="learn target speed; Shield preserves viability",
        safety="TIME_AWARE physical Shield",
        classification="FEASIBILITY_PRESERVED_WITHOUT_TARGET_ENERGY_GAIN",
        timing="AMENDED_POST_INCIDENT_VALIDATION",
        basis="PAIRED_SAME_TASKS",
        incidents="V1 remains INCIDENT_ABORTED",
        checkpoint="LOCAL_PRESENT_HASH_MATCHES_RECORDED",
        limitations="one task now proven infeasible; raw RSR remains 35/36",
        evidence_paths=["benchmarks/hybrid_rl_validation_v2/VALIDATION_RESULTS.md", "benchmarks/hybrid_rl_validation_v2/validation_results.json", "benchmarks/hybrid_rl_validation_v2/pair_ledger.json"],
    )
    for seed, energy, time in (
        (11, 0.17614826280455592, 129.39166666666358),
        (29, 0.17580767360667518, 128.9194444444414),
        (47, 0.17496537986333954, 129.86944444444134),
    ):
        values = dict(hybrid_common)
        rows.append(
            method(
                f"hybrid_h{seed}",
                f"Hybrid RL H{seed}",
                "V1 frozen 300k",
                seeds=[seed],
                energy=energy,
                travel_time=time,
                stream_hash=NOT_RECORDED,
                **values,
            )
        )

    physical_common = dict(
        comparison_family="G_IMPROVED_NONLEARNING_CONTROL",
        algorithm_family="causal non-learning physical control",
        branch="research/causal-mpc-v2",
        source_commit="c7c6175b22e373f476aa60c10a755cb7072d9d5f",
        seeds="none",
        transitions=0,
        updates=0,
        panel="MPC_V2_CAPABILITY",
        routes=18,
        pairs=18,
        rsr=(18, 18),
        completion=18,
        deadline=18,
        speed=18,
        observation="causal physical state and visible infrastructure",
        physics="exact known dynamics; no hidden route",
        hidden_actor=False,
        privileged_critic=False,
        safety="physical Shield and deadline recoverability checks",
        basis="PAIRED_SAME_TASKS",
        evidence_paths=["benchmarks/causal_mpc_v2/RESULTS.md", "benchmarks/causal_mpc_v2/capability_results.json"],
    )
    for method_id, name, energy, objective in (
        ("causal60_capability", "CAUSAL60", 0.196323, "fixed causal speed policy"),
        ("time_aware_capability", "TIME_AWARE", 0.180326, "deterministic pace-aware target speed"),
        ("fixed_price_mpc_v2", "FixedPrice MPC V2", 0.172002, "fixed time-price MPC V2"),
        ("adaptive_price_mpc_v2", "AdaptivePrice MPC V2", 0.17066648205049648, "reserve-feedback time-price MPC V2"),
    ):
        rows.append(
            method(
                method_id,
                name,
                "V2 Capability",
                energy=energy,
                objective=objective,
                classification="MPC_IMPROVED_BUT_RL_VALUE_UNCLEAR",
                **physical_common,
            )
        )

    rows.extend(
        [
            method(
                "slacktrader_v1",
                "SlackTrader",
                "V1 stopped before training",
                "H_UNSUCCESSFUL_FOLLOWUP",
                "planned SAC shadow-price learner",
                "research/slacktrader-v1",
                "303fc16c95845579ad461873af790204b5e933a4",
                seeds="NOT_STARTED",
                transitions=0,
                updates=0,
                panel="four-task MPC Development preflight",
                routes=4,
                pairs=16,
                rsr=(16, 16),
                objective="planned learned time price; only non-learning MPC preflight executed",
                classification="RL_NOT_JUSTIFIED_BY_MPC_PREFLIGHT",
                timing="STOPPED_BEFORE_MAIN_TRAINING",
                basis="PAIRED_SAME_TASKS",
                limitations="not a completed RL method; Validation has zero episodes",
                checkpoint="NO_MODEL_TRAINED",
                evidence_paths=["benchmarks/slacktrader_v1/README.md", "benchmarks/slacktrader_v1/mpc_development_results.json"],
            ),
            method(
                "temporal_continue",
                "CONTINUE_FEEDBACK temporal options",
                "V1",
                "H_UNSUCCESSFUL_FOLLOWUP",
                "non-learning temporal-option controller",
                "research/temporal-options-v1",
                "29b31837984b654e9011d3e180a8e33926345489",
                seeds="none",
                transitions=0,
                updates=0,
                counterfactual_steps=NOT_RECORDED,
                panel="TEMPORAL_OPTIONS_CAPABILITY",
                routes=18,
                pairs=18,
                rsr=(18, 18),
                completion=18,
                deadline=18,
                speed=18,
                energy=0.17272235199190977,
                travel_time=97.19999999999875,
                observation="70D causal option-state interface",
                action="18 masked high-level START/CONTINUE actions",
                physics="exact causal planner and Shield",
                objective="heuristic option continuation",
                safety="physical Shield",
                classification="SEMANTIC_OR_INTEGRITY_STOP",
                timing="PROSPECTIVE_CAPABILITY",
                basis="PAIRED_SAME_TASKS",
                incidents="completion_bound_s was not retained",
                limitations="non-learning; no independently recomputable bound-falsification audit",
                evidence_paths=["benchmarks/temporal_options_v1/RESULTS.md", "benchmarks/temporal_options_v1/results.json"],
            ),
            method(
                "temporal_reset",
                "RESET_FEEDBACK temporal options",
                "V1",
                "H_UNSUCCESSFUL_FOLLOWUP",
                "non-learning receding-horizon controller",
                "research/temporal-options-v1",
                "29b31837984b654e9011d3e180a8e33926345489",
                seeds="none",
                transitions=0,
                updates=0,
                panel="TEMPORAL_OPTIONS_CAPABILITY",
                routes=18,
                pairs=18,
                rsr=(18, 18),
                completion=18,
                deadline=18,
                speed=18,
                energy=0.171765449,
                observation="70D causal option-state interface",
                action="reset/replan option every decision",
                physics="exact causal planner and Shield",
                objective="non-learning reset architecture",
                safety="physical Shield",
                classification="SEMANTIC_OR_INTEGRITY_STOP",
                basis="PAIRED_SAME_TASKS",
                incidents="completion_bound_s was not retained",
                evidence_paths=["benchmarks/temporal_options_v1/RESULTS.md", "benchmarks/temporal_options_v1/results.json"],
            ),
            method(
                "temporal_commit",
                "COMMIT_FEEDBACK temporal options",
                "V1",
                "H_UNSUCCESSFUL_FOLLOWUP",
                "non-learning committed-option controller",
                "research/temporal-options-v1",
                "29b31837984b654e9011d3e180a8e33926345489",
                seeds="none",
                transitions=0,
                updates=0,
                panel="TEMPORAL_OPTIONS_CAPABILITY",
                routes=18,
                pairs=18,
                rsr=(18, 18),
                completion=18,
                deadline=18,
                speed=18,
                energy=0.176739850,
                observation="70D causal option-state interface",
                action="commit to selected option",
                physics="exact causal planner and Shield",
                objective="non-learning committed option architecture",
                safety="physical Shield",
                classification="SEMANTIC_OR_INTEGRITY_STOP",
                basis="PAIRED_SAME_TASKS",
                incidents="completion_bound_s was not retained",
                evidence_paths=["benchmarks/temporal_options_v1/RESULTS.md", "benchmarks/temporal_options_v1/results.json"],
            ),
            method(
                "counterfactual_predictor",
                "Counterfactual Option Value predictor",
                "V1 pilot",
                "H_UNSUCCESSFUL_FOLLOWUP",
                "diagnostic supervised predictor",
                "research/counterfactual-option-value-v1",
                "9802f9262ea7f753365fc4bdd20b1fe08379291d",
                seeds="Fit/Calibration/Test route panels; not RL seeds",
                transitions=0,
                updates=1800,
                counterfactual_steps=476_879,
                panel="CF_OPTION_TEST_V1 states",
                routes=12,
                pairs=88,
                objective="predict sampled alternative feasibility and energy regret",
                classification="INSUFFICIENT_IMPROVEMENT_OPPORTUNITY",
                timing="PREREGISTERED_BLIND_TEST_PILOT",
                basis="PAIRED_SAME_TASKS",
                limitations="not a controller evaluation and not reinforcement learning",
                checkpoint="SUPERVISED_PREDICTOR_ARTIFACT_ONLY",
                evidence_paths=["benchmarks/counterfactual_option_value_v1/RESULTS.md", "benchmarks/counterfactual_option_value_v1/results.json"],
            ),
        ]
    )
    return rows


def assert_headlines() -> None:
    same = read_json("benchmarks/same_information_scalar_vs_constrained/results.json")
    assert same["primary_endpoint"]["pooled"]["scalar"] == {
        "count": 0,
        "denominator": 54,
        "proportion": 0.0,
    }
    assert same["primary_endpoint"]["pooled"]["constrained"]["count"] == 23
    gcsl = read_json("benchmarks/gcsl_canonical/results.json")
    assert gcsl["pooled"]["GCSL"]["rsr"] == 0
    assert gcsl["pooled"]["GCSL"]["episodes"] == 54
    physical = read_json("benchmarks/physical_feasibility_reference/holdout_results.json")
    assert physical["causal"]["summary"]["requirement_satisfaction_count"] == 36
    assert physical["oracle"]["summary"]["requirement_satisfaction_count"] == 36
    hybrid = read_json("benchmarks/hybrid_rl_validation_v2/validation_results.json")
    assert all(summary["rsr"] == 35 for summary in hybrid["controller_summaries"].values())
    causal = read_json("benchmarks/causal_mpc_v2/capability_results.json")
    assert all(arm["summary"]["rsr"] == 18 for arm in causal["arms"].values())
    assert causal["paired_feasible_energy"]["adaptive_vs_time_aware"][
        "arithmetic_mean_saving_percent"
    ] == 5.421834807082291
    assert causal["paired_feasible_energy"]["adaptive_vs_fixed"][
        "arithmetic_mean_saving_percent"
    ] == 0.8011839951300701
    temporal = read_json("benchmarks/temporal_options_v1/results.json")
    assert all(arm["summary"]["rsr"] == 18 for arm in temporal["arms"].values())
    assert temporal["comparisons"]["primary_continue_vs_reset"][
        "arithmetic_mean_saving_percent"
    ] == -0.5620832885179694
    counterfactual = read_json("benchmarks/counterfactual_option_value_v1/results.json")
    assert counterfactual["opportunity_test"]["distinguishable_behavior"][
        "saving_0_2wh_count"
    ] == 16
    assert counterfactual["opportunity_test"]["valid_reference_snapshot_count"] == 88


def descriptive_interval(values: list[float], seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    means = []
    for _ in range(20_000):
        means.append(statistics.fmean(rng.choice(values) for _ in values))
    means.sort()
    return {
        "lower": means[499],
        "upper": means[19_499],
        "confidence_level": 0.95,
        "replicates": 20_000,
        "sampling_unit": "task",
        "interpretation": "descriptive_percentile_interval",
        "deterministic_rng_seed": seed,
    }


def paired_summary(
    comparison_id: str,
    treatment: str,
    baseline: str,
    values: list[dict[str, float | int]],
    *,
    population: str,
    qualification: str,
    source: str,
    seed: int,
) -> dict[str, Any]:
    savings = [float(row["saving_percent"]) for row in values]
    differences = [float(row["energy_difference_kwh"]) for row in values]
    return {
        "comparison_id": comparison_id,
        "treatment": treatment,
        "baseline": baseline,
        "comparison_basis": "PAIRED_SAME_TASKS",
        "comparable_population": population,
        "distinct_task_count": len(values),
        "paired_count": len(values),
        "arithmetic_mean_of_per_task_percentage_saving": statistics.fmean(savings),
        "median_percentage_saving": statistics.median(savings),
        "percentage_saving_range": [min(savings), max(savings)],
        "mean_absolute_energy_difference_kwh": statistics.fmean(differences),
        "absolute_energy_difference_range_kwh": [min(differences), max(differences)],
        "descriptive_95_percent_interval_for_mean_percentage_saving": descriptive_interval(
            savings, seed
        ),
        "qualification": qualification,
        "source": source,
    }


def paired_rows() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    hybrid_path = "benchmarks/hybrid_rl_validation_v2/validation_pairwise_analysis.json"
    hybrid = read_json(hybrid_path)
    for index, controller in enumerate(("H11", "H29", "H47"), start=1):
        record = hybrid["hybrid"][controller]["vs_time_aware"]
        savings = record["saving_percent"]
        differences = record["energy_difference_kwh"]
        result.append(
            {
                "comparison_id": f"{controller.lower()}_vs_time_aware",
                "treatment": controller,
                "baseline": "TIME_AWARE",
                "comparison_basis": "PAIRED_SAME_TASKS",
                "comparable_population": "35 jointly feasible Hybrid Validation V2 tasks",
                "distinct_task_count": 35,
                "paired_count": 35,
                "arithmetic_mean_of_per_task_percentage_saving": savings["mean"],
                "median_percentage_saving": savings["median"],
                "percentage_saving_range": [savings["minimum"], savings["maximum"]],
                "mean_absolute_energy_difference_kwh": differences["mean"],
                "absolute_energy_difference_range_kwh": [
                    differences["minimum"],
                    differences["maximum"],
                ],
                "descriptive_95_percent_interval_for_mean_percentage_saving": savings[
                    "bootstrap_95_percent_interval_for_mean"
                ],
                "qualification": "three trained seeds share the same routes; intervals are descriptive and correlated across seed comparisons",
                "source": hybrid_path,
            }
        )

    causal_path = "benchmarks/causal_mpc_v2/capability_results.json"
    causal = read_json(causal_path)
    for index, (key, treatment, baseline) in enumerate(
        (
            ("fixed_vs_time_aware", "FIXED_PRICE_V2", "TIME_AWARE"),
            ("adaptive_vs_time_aware", "ADAPTIVE_PRICE_V2", "TIME_AWARE"),
            ("adaptive_vs_fixed", "ADAPTIVE_PRICE_V2", "FIXED_PRICE_V2"),
        ),
        start=10,
    ):
        rows = []
        for item in causal["paired_feasible_energy"][key]["task_level"]:
            rows.append(
                {
                    "saving_percent": item["saving_percent"],
                    "energy_difference_kwh": item["baseline_energy_kwh"]
                    - item["treatment_energy_kwh"],
                }
            )
        result.append(
            paired_summary(
                key,
                treatment,
                baseline,
                rows,
                population="18 Capability tasks; all four arms feasible",
                qualification="non-learning comparison; Fixed versus TIME_AWARE changes candidate richness and terminal treatment together",
                source=causal_path,
                seed=20261023 + index,
            )
        )

    temporal_path = "benchmarks/temporal_options_v1/results.json"
    temporal = read_json(temporal_path)
    by_arm = {
        name: {episode["numeric_seed"]: episode for episode in arm["episodes"]}
        for name, arm in temporal["arms"].items()
    }
    rows = []
    registered = temporal["comparisons"]["primary_continue_vs_reset"]
    registered_by_seed = {
        row["numeric_seed"]: row["saving_percent"] for row in registered["task_level"]
    }
    for numeric_seed, baseline_episode in by_arm["RESET_FEEDBACK"].items():
        treatment_episode = by_arm["CONTINUE_FEEDBACK"][numeric_seed]
        rows.append(
            {
                "saving_percent": registered_by_seed[numeric_seed],
                "energy_difference_kwh": baseline_episode["energy_kwh"]
                - treatment_episode["energy_kwh"],
            }
        )
    result.append(
        paired_summary(
            "continue_vs_reset",
            "CONTINUE_FEEDBACK",
            "RESET_FEEDBACK",
            rows,
            population="18 Temporal Options Capability tasks; both arms feasible",
            qualification="architectural non-learning effect; integrity stop limits independent deadline-bound audit",
            source=temporal_path,
            seed=20261040,
        )
    )

    physical_path = "benchmarks/physical_feasibility_reference/holdout_results.json"
    physical = read_json(physical_path)
    causal_by_seed = {row["track_seed"]: row for row in physical["causal"]["episodes"]}
    rows = []
    for oracle in physical["oracle"]["episodes"]:
        causal_row = causal_by_seed[oracle["track_seed"]]
        baseline_energy = causal_row["signed_net_energy_kwh"]
        treatment_energy = oracle["signed_net_energy_kwh"]
        rows.append(
            {
                "saving_percent": 100.0
                * (baseline_energy - treatment_energy)
                / baseline_energy,
                "energy_difference_kwh": baseline_energy - treatment_energy,
            }
        )
    result.append(
        paired_summary(
            "privileged_oracle_vs_causal_reference",
            "FULL_ROUTE_ORACLE",
            "CAUSAL_REFERENCE",
            rows,
            population="36 independent physical-reference holdout tasks",
            qualification="paired but not information-matched; Oracle is contextual and not certified globally optimal",
            source=physical_path,
            seed=20261041,
        )
    )

    result.append(
        {
            "comparison_id": "sameinfo_constrained_vs_scalar",
            "treatment": "MATCHED_CONSTRAINED",
            "baseline": "SAME_INFORMATION_SCALAR",
            "comparison_basis": "PAIRED_SAME_STREAM",
            "comparable_population": "18 Validation tasks for each of three matched training-seed pairs",
            "distinct_task_count": 18,
            "paired_count": 54,
            "rsr_count_difference": 23,
            "arithmetic_mean_of_per_task_percentage_saving": None,
            "median_percentage_saving": None,
            "percentage_saving_range": None,
            "mean_absolute_energy_difference_kwh": None,
            "absolute_energy_difference_range_kwh": None,
            "descriptive_95_percent_interval_for_mean_percentage_saving": None,
            "qualification": "Scalar has no feasible episode, so no paired energy statistic exists; algorithmic mechanisms differ beyond scalarization",
            "source": "benchmarks/same_information_scalar_vs_constrained/results.json",
        }
    )
    return result


def evidence_rows() -> list[dict[str, Any]]:
    return [
        {"evidence_id": "sameinfo", "comparison_family": "B_EXPLICIT_CONSTRAINTS", "methods": "Matched Constrained vs Same-Information Scalar", "population": "18 routes x three matched seed pairs", "comparison_basis": "PAIRED_SAME_STREAM", "distinct_routes": 18, "controller_task_pairs": 108, "result": "23/54 vs 0/54 RSR", "classification": "CONSTRAINED_ROBUST_ADVANTAGE", "qualification": "not an algorithm-only ablation", "source": "benchmarks/same_information_scalar_vs_constrained/results.json"},
        {"evidence_id": "gcsl", "comparison_family": "C_FUTURE_GOAL_CONDITIONED", "methods": "GCSL", "population": "GCSL_VALIDATION_V1", "comparison_basis": "PAIRED_SAME_TASKS", "distinct_routes": 18, "controller_task_pairs": 54, "result": "0/54 RSR; 52/54 completion/deadline; 1/54 speed compliance", "classification": "GCSL_ZERO", "qualification": "constrained reference is contextual", "source": "benchmarks/gcsl_canonical/results.json"},
        {"evidence_id": "physical_reference", "comparison_family": "E_PHYSICS_REFERENCE", "methods": "full-route Oracle and causal physical reference", "population": "independent 36-route holdout", "comparison_basis": "PAIRED_SAME_TASKS", "distinct_routes": 36, "controller_task_pairs": 72, "result": "36/36 RSR for both", "classification": "PHYSICALLY_SOLVABLE_REFERENCE", "qualification": "Oracle privileged; neither 36/36 proves population-wide 100%", "source": "benchmarks/physical_feasibility_reference/holdout_results.json"},
        {"evidence_id": "hybrid_v1", "comparison_family": "F_PHYSICS_SHIELDED_LEARNING", "methods": "Hybrid Validation V1", "population": "aborted transaction", "comparison_basis": "NOT_COMPARABLE", "distinct_routes": NOT_RECORDED, "controller_task_pairs": 0, "result": "no valid performance result", "classification": "INCIDENT_ABORTED", "qualification": "two completed episodes lacked persisted results; one environment reset without action", "source": "benchmarks/hybrid_rl_execution/VALIDATION_INTEGRITY_INCIDENT.md"},
        {"evidence_id": "hybrid_v2", "comparison_family": "F_PHYSICS_SHIELDED_LEARNING", "methods": "CAUSAL60, TIME_AWARE, H11, H29, H47", "population": "amended post-incident 36-route Validation", "comparison_basis": "PAIRED_SAME_TASKS", "distinct_routes": 36, "controller_task_pairs": 180, "result": "35/36 RSR for every arm; H11/H29/H47 save 1.035/1.245/1.700% vs TIME_AWARE on 35 jointly feasible tasks", "classification": "FEASIBILITY_PRESERVED_WITHOUT_TARGET_ENERGY_GAIN", "qualification": "task 10200029 is now proven infeasible; raw RSR remains 35/36", "source": "benchmarks/hybrid_rl_validation_v2/validation_results.json"},
        {"evidence_id": "causal_mpc_v2", "comparison_family": "G_IMPROVED_NONLEARNING_CONTROL", "methods": "CAUSAL60, TIME_AWARE, FixedPrice V2, AdaptivePrice V2", "population": "18-route Capability", "comparison_basis": "PAIRED_SAME_TASKS", "distinct_routes": 18, "controller_task_pairs": 72, "result": "18/18 each; Adaptive saves 5.422% vs TIME_AWARE and 0.801% vs Fixed", "classification": "MPC_IMPROVED_BUT_RL_VALUE_UNCLEAR", "qualification": "Fixed vs TIME_AWARE mixes profile-family and terminal-treatment changes", "source": "benchmarks/causal_mpc_v2/capability_results.json"},
        {"evidence_id": "slacktrader", "comparison_family": "H_UNSUCCESSFUL_FOLLOWUP", "methods": "SlackTrader V1 preflight", "population": "four Development tasks", "comparison_basis": "PAIRED_SAME_TASKS", "distinct_routes": 4, "controller_task_pairs": 16, "result": "AdaptivePrice preflight 9.301% worse than TIME_AWARE", "classification": "RL_NOT_JUSTIFIED_BY_MPC_PREFLIGHT", "qualification": "main RL training never started", "source": "benchmarks/slacktrader_v1/mpc_development_results.json"},
        {"evidence_id": "temporal_options", "comparison_family": "H_UNSUCCESSFUL_FOLLOWUP", "methods": "FROZEN_MPC, RESET, CONTINUE, COMMIT", "population": "18-route Capability", "comparison_basis": "PAIRED_SAME_TASKS", "distinct_routes": 18, "controller_task_pairs": 72, "result": "18/18 each; CONTINUE uses 0.562% more energy than RESET", "classification": "SEMANTIC_OR_INTEGRITY_STOP", "qualification": "non-learning; completion bounds not retained", "source": "benchmarks/temporal_options_v1/results.json"},
        {"evidence_id": "counterfactual", "comparison_family": "H_UNSUCCESSFUL_FOLLOWUP", "methods": "Counterfactual supervised predictor pilot", "population": "88 valid Test states", "comparison_basis": "PAIRED_SAME_TASKS", "distinct_routes": 11, "controller_task_pairs": 88, "result": "16/88 states have feasible sampled improvement >=0.2 Wh", "classification": "INSUFFICIENT_IMPROVEMENT_OPPORTUNITY", "qualification": "not a completed RL method or controller evaluation", "source": "benchmarks/counterfactual_option_value_v1/results.json"},
        {"evidence_id": "constrained_v2", "comparison_family": "B_EXPLICIT_CONSTRAINTS", "methods": "Constrained V2", "population": "Validation 3000-3008 x three seeds", "comparison_basis": "PAIRED_SAME_TASKS", "distinct_routes": 9, "controller_task_pairs": 27, "result": "21/27 RSR; 27/27 completion/deadline; 21/27 speed", "classification": "LEARNABLE_NOT_SEED_ROBUST", "qualification": "same nine routes reused across three trained seeds", "source": "benchmarks/constrained_rl_v2/results.json"},
        {"evidence_id": "reward_family", "comparison_family": "A_STANDARD_REWARD_RL", "methods": "SB3 SAC, PPO, Binary, LLM reward", "population": "Validation 3000-3008 x three seeds", "comparison_basis": "PAIRED_SAME_TASKS", "distinct_routes": 9, "controller_task_pairs": 108, "result": "5/27, 0/27, 0/27, 1/27 RSR", "classification": "MULTIPLE_NEGATIVE_OR_WEAK_RESULTS", "qualification": "not all reward implementations are one-factor matched", "source": "benchmarks/llm_reward/FINAL_RESULTS.md"},
        {"evidence_id": "requirement_conditioning", "comparison_family": "C_FUTURE_GOAL_CONDITIONED", "methods": "Requirement-Conditioned SAC", "population": "five requirements x nine routes x three seeds", "comparison_basis": "PAIRED_SAME_TASKS", "distinct_routes": 9, "controller_task_pairs": 135, "result": "48/135 overall; canonical 140 s subset 12/27", "classification": "RESPONSIVE_BUT_UNRELIABLE", "qualification": "full matrix contains distinct task requirements on shared routes", "source": "benchmarks/requirement_conditioned/results.json"},
        {"evidence_id": "goal_sac", "comparison_family": "C_FUTURE_GOAL_CONDITIONED", "methods": "Goal SAC without/with HER", "population": "Validation 3000-3008 x three seeds", "comparison_basis": "PAIRED_SAME_TASKS", "distinct_routes": 9, "controller_task_pairs": 54, "result": "0/27 and 0/27 RSR", "classification": "NO_HER_BENEFIT", "qualification": "virtual relabels are not physical successes", "source": "benchmarks/goal_conditioned/results.json"},
        {"evidence_id": "contrastive", "comparison_family": "C_FUTURE_GOAL_CONDITIONED", "methods": "Projected-goal CRL depth 4/16", "population": "Validation 3000-3008 x three seeds", "comparison_basis": "PAIRED_SAME_STREAM", "distinct_routes": 9, "controller_task_pairs": 54, "result": "0/27 vs 1/27 RSR", "classification": "NO_REPEATABLE_DEPTH_GAIN", "qualification": "300k scale only", "source": "benchmarks/contrastive_rl/results.json"},
        {"evidence_id": "learned_mbrl", "comparison_family": "D_MODEL_BASED_RL", "methods": "mixed learned-dynamics MBRL", "population": "one seed, partial Development", "comparison_basis": "NOT_COMPARABLE", "distinct_routes": 9, "controller_task_pairs": 9, "result": "0/9 at 50k; no final endpoint", "classification": "TERMINATED_RESOURCE_INFEASIBLE", "qualification": "termination was compute-based, not performance-based", "source": "benchmarks/model_based_rl/termination.json"},
        {"evidence_id": "physics_mbrl", "comparison_family": "D_MODEL_BASED_RL", "methods": "exact one-step Physics MBRL", "population": "Validation 3000-3008 x three seeds", "comparison_basis": "CONTEXTUAL_ONLY", "distinct_routes": 9, "controller_task_pairs": 27, "result": "9/27 RSR", "classification": "NO_MATERIAL_IMPROVEMENT", "qualification": "mixes one-step augmentation with a different real-data target", "source": "benchmarks/physics_model_mbrl/results.json"},
        {"evidence_id": "mve", "comparison_family": "D_MODEL_BASED_RL", "methods": "Physics MVE vs matched real-only", "population": "Validation 3000-3008 x three seeds", "comparison_basis": "PAIRED_SAME_STREAM", "distinct_routes": 9, "controller_task_pairs": 54, "result": "0/27 vs 10/27 RSR", "classification": "DEGRADATION_AND_UNSTABLE", "qualification": "one MVE seed crossed stability threshold", "source": "benchmarks/multistep_physics_mbrl/results.json"},
    ]


def registry_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Method Registry",
        "",
        "`method_registry.json` is authoritative. Rows group training seeds only when the scientific configuration is identical; checkpoints are not counted as independent seeds.",
        "",
        "| ID | Method/version | Family | Seeds | Training transitions | Panel | RSR | Classification |",
        "| --- | --- | --- | --- | ---: | --- | ---: | --- |",
    ]
    for row in rows:
        seeds = json.dumps(row["training_seeds"], separators=(",", ":"))
        rsr = f'{row["rsr_numerator"]}/{row["rsr_denominator"]}'
        lines.append(
            f'| {row["method_id"]} | {row["method_name"]} ({row["version"]}) | '
            f'{row["comparison_family"]} | {seeds} | {row["real_native_training_transitions"]} | '
            f'{row["evaluation_panel"]} | {rsr} | {row["scientific_classification"]} |'
        )
    lines.extend(
        [
            "",
            "Fields not supported by immutable artifacts are `NOT_RECORDED`; they are not imputed. Full observation, action, physics-information, objective, integrity, checkpoint, and evidence-path fields are in the JSON.",
        ]
    )
    return "\n".join(lines) + "\n"


def evidence_csv(rows: list[dict[str, Any]]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def evidence_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Master Evidence Matrix",
        "",
        "`master_evidence_matrix.csv` is authoritative. `PAIRED_SAME_TASKS` and `PAIRED_SAME_STREAM` rows support within-panel contrasts; `CONTEXTUAL_ONLY` rows must not be ranked as if populations and information contracts matched.",
        "",
        "| Evidence | Methods | Basis | Routes | Result | Frozen classification | Qualification |",
        "| --- | --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f'| {row["evidence_id"]} | {row["methods"]} | {row["comparison_basis"]} | '
            f'{row["distinct_routes"]} | {row["result"]} | {row["classification"]} | {row["qualification"]} |'
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The strongest algorithmically close comparison is Same-Information Scalar versus Matched Constrained on the same stream and tasks. Hybrid and MPC energy effects are paired within their own panels. Physical-reference 36/36, Hybrid 35/36, and MPC 18/18 are cross-panel results and are not a global controller ranking. The full-route Oracle is privileged contextual evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def generated_outputs() -> dict[Path, str]:
    assert_headlines()
    methods = method_rows()
    pairs = paired_rows()
    evidence = evidence_rows()
    return {
        PACKAGE_DIR / "method_registry.json": json.dumps(
            {"schema_version": 1, "row_count": len(methods), "methods": methods},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        PACKAGE_DIR / "METHOD_REGISTRY.md": registry_markdown(methods),
        PACKAGE_DIR / "paired_comparison_registry.json": json.dumps(
            {"schema_version": 1, "comparison_count": len(pairs), "comparisons": pairs},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        PACKAGE_DIR / "master_evidence_matrix.csv": evidence_csv(evidence),
        PACKAGE_DIR / "MASTER_EVIDENCE_MATRIX.md": evidence_markdown(evidence),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.write == arguments.check:
        parser.error("choose exactly one of --write or --check")
    outputs = generated_outputs()
    if arguments.write:
        for path, content in outputs.items():
            path.write_text(content, encoding="utf-8")
        return 0
    stale = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path, content in outputs.items()
        if not path.exists() or path.read_text(encoding="utf-8") != content
    ]
    if stale:
        raise SystemExit("stale generated evidence artifacts: " + ", ".join(stale))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
