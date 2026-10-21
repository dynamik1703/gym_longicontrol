"""Fail-closed V2 preparation and frozen-policy integrity checks."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from benchmarks.hybrid_rl_execution.agent import ConstrainedSAC
from benchmarks.hybrid_rl_execution.checkpointing import (
    file_sha256,
    load_checkpoint,
)

from .constants import (
    AUTHORIZATION,
    EXPECTED_ACTOR_SHA256,
    EXPECTED_CHECKPOINT_SHA256,
    HERE,
    INCIDENT_PRESERVATION_COMMIT,
    ORIGINAL_MAIN_TRAINING_COMMIT,
    ORIGINAL_WORKTREE,
    PAIR_LEDGER,
    POLICY_SEEDS,
    PREPARATION_STATUS,
    ROOT,
    SPLIT_LEDGER,
    TRAINING_EXECUTION_CONFIGURATION_SHA256,
    V1_ORDERED_SHA256,
    V2_ID,
)
from .ledger import (
    atomic_json,
    load_json,
    validate_pair_ledger,
    validate_registration,
)
from .split import verify_regeneration

MAIN_RESULTS = ROOT / "benchmarks/hybrid_rl_execution/main_training_results.json"
MAIN_MANIFEST = ORIGINAL_WORKTREE / "runs/hybrid-rl-execution-v1/manifest.json"
INCIDENT_JSON = (
    ROOT / "benchmarks/hybrid_rl_execution/validation_integrity_incident.json"
)
INCIDENT_MD = (
    ROOT / "benchmarks/hybrid_rl_execution/VALIDATION_INTEGRITY_INCIDENT.md"
)
DESIGN_STATUS = ROOT / "benchmarks/hybrid_rl_design/preparation_status.json"
PHYSICAL_STATUS = ROOT / "benchmarks/physical_feasibility_reference/freeze.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actor_parameter_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state["modules"]["actor"].items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(tuple(contiguous.shape)).encode("ascii"))
        digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def verify_incident_evidence() -> dict[str, Any]:
    incident = load_json(INCIDENT_JSON)
    interactions = {
        row["controller_id"]: row for row in incident["known_environment_interactions"]
    }
    checks = {
        "state_permanent": incident["state"]
        == "INTEGRITY_STOPPED_BEFORE_REGISTERED_TRANSACTION",
        "seed_10100000": incident["affected_validation_task"]["numeric_seed"]
        == 10_100_000,
        "causal_one_completed_unpersisted": interactions["CAUSAL60"]
        == {
            "controller_id": "CAUSAL60",
            "status": "COMPLETED_RESULT_NOT_PERSISTED",
            "performance_episode_count": 1,
        },
        "time_aware_one_completed_unpersisted": interactions["TIME_AWARE"]
        == {
            "controller_id": "TIME_AWARE",
            "status": "COMPLETED_RESULT_NOT_PERSISTED",
            "performance_episode_count": 1,
        },
        "hybrid_reset_without_action_or_episode": interactions[
            "UNBOUND_HYBRID_AGENT_NONE"
        ]["status"]
        == "ENVIRONMENT_RESET_THEN_ABORTED_BEFORE_FIRST_ACTION"
        and interactions["UNBOUND_HYBRID_AGENT_NONE"]["performance_episode_count"]
        == 0,
        "two_completed_zero_persisted": incident[
            "known_completed_performance_episodes"
        ]
        == 2
        and incident["persisted_episode_results"] == 0,
        "zero_registered_pairs": not incident["registered_validation_transaction"]
        and not incident["validation_pair_ledger_created"],
        "zero_training_changes": not incident["training_or_adaptation"]["occurred"]
        and incident["training_or_adaptation"]["gradient_updates"] == 0
        and incident["training_or_adaptation"]["checkpoint_changes"] == 0,
        "paper_zero_and_sealed": incident["paper"]
        == {
            "state": "SEALED_UNINSPECTED",
            "episode_count": 0,
            "content_fingerprints_constructed": 0,
        },
        "v1_ordered_hash_preserved": V1_ORDERED_SHA256
        == "f3864fed8725874dc315004375af0d2850dc2d2d56957d52c64d043ca6b3352b",
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "incident_json_sha256": _sha256(INCIDENT_JSON),
        "incident_markdown_sha256": _sha256(INCIDENT_MD),
        "incident_preservation_commit": INCIDENT_PRESERVATION_COMMIT,
    }


def verify_frozen_scientific_sources() -> dict[str, Any]:
    results = load_json(MAIN_RESULTS)
    expected = results["provenance"]["source_hashes"]
    observed = {relative: _sha256(ROOT / relative) for relative in expected}
    design = load_json(DESIGN_STATUS)["scientific_source_sha256"]
    physical = load_json(PHYSICAL_STATUS)["source_sha256"]
    controller_checks = {
        "CAUSAL60": _sha256(
            ROOT
            / "benchmarks/physical_feasibility_reference/causal_controller.py"
        )
        == physical["causal_controller.py"],
        "TIME_AWARE": _sha256(ROOT / "benchmarks/hybrid_rl_design/baseline.py")
        == design["baseline.py"],
        "HYBRID_SHIELD_CONTROLLER": _sha256(
            ROOT / "benchmarks/hybrid_rl_design/controller.py"
        )
        == design["controller.py"],
    }
    return {
        "passed": observed == expected and all(controller_checks.values()),
        "expected_source_sha256": expected,
        "observed_source_sha256": observed,
        "controller_source_unchanged": controller_checks,
        "original_canonical_unchanged": observed[
            "benchmarks/hybrid_rl_execution/canonical.json"
        ]
        == TRAINING_EXECUTION_CONFIGURATION_SHA256,
    }


def load_frozen_policies(
    *,
    return_agents: bool = False,
) -> tuple[dict[str, Any], dict[str, ConstrainedSAC]]:
    manifest = load_json(MAIN_MANIFEST)
    if manifest["state"] != "COMPLETED":
        raise RuntimeError("original main training is not completed")
    if manifest["provenance"]["configuration_sha256"] != (
        TRAINING_EXECUTION_CONFIGURATION_SHA256
    ):
        raise RuntimeError("training configuration provenance changed")
    metadata: dict[str, Any] = {}
    agents: dict[str, ConstrainedSAC] = {}
    for controller_id, seed in POLICY_SEEDS.items():
        run = manifest["runs"][str(seed)]
        checkpoint = run["final_checkpoint"]
        expected_checkpoint = EXPECTED_CHECKPOINT_SHA256[controller_id]
        if checkpoint["sha256"] != expected_checkpoint:
            raise RuntimeError(f"{controller_id} manifest checkpoint changed")
        path = Path(checkpoint["path"])
        if not path.is_relative_to(ORIGINAL_WORKTREE):
            raise RuntimeError("V2 must use original checkpoints read-only")
        observed_file_hash = file_sha256(path)
        if observed_file_hash != expected_checkpoint:
            raise RuntimeError(f"{controller_id} checkpoint bytes changed")
        loaded = load_checkpoint(
            path,
            expected_sha256=expected_checkpoint,
            expected_configuration_sha256=TRAINING_EXECUTION_CONFIGURATION_SHA256,
            expected_source_hashes=manifest["provenance"]["source_hashes"],
            expected_attempt_id=checkpoint["attempt_id"],
        )
        if (
            loaded.runtime["native_transition_count"] != 300_000
            or loaded.runtime["gradient_update_count"] != 30_000
        ):
            raise RuntimeError(f"{controller_id} is not the frozen 300k endpoint")
        actor_hash = actor_parameter_sha256(loaded.agent_state)
        if actor_hash != EXPECTED_ACTOR_SHA256[controller_id]:
            raise RuntimeError(f"{controller_id} actor fingerprint changed")
        agent = ConstrainedSAC(seed)
        agent.load_state_dict(loaded.agent_state)
        if agent.update_count != 30_000:
            raise RuntimeError(f"{controller_id} update count changed")
        loaded.environment.close()
        metadata[controller_id] = {
            "training_seed": seed,
            "checkpoint_path": str(path),
            "checkpoint_sha256": observed_file_hash,
            "actor_parameter_sha256": actor_hash,
            "native_training_transitions": 300_000,
            "gradient_updates": 30_000,
            "loaded_read_only": True,
        }
        if return_agents:
            agents[controller_id] = agent
    return metadata, agents


def controller_identities(policy_metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "CAUSAL60": {
            "kind": "frozen_deterministic_controller",
            "implementation": (
                "benchmarks.physical_feasibility_reference.causal_controller."
                "ObservationLimitedController"
            ),
        },
        "TIME_AWARE": {
            "kind": "frozen_nonlearning_controller_through_identical_shield",
            "implementation": (
                "benchmarks.hybrid_rl_design.baseline.TimeAwareEnergyStrategy"
            ),
        },
        **{key: dict(policy_metadata[key]) for key in POLICY_SEEDS},
    }


def verify_git_lineage() -> dict[str, Any]:
    branch = subprocess.run(
        ("git", "branch", "--show-current"),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(
        ("git", "merge-base", "--is-ancestor", INCIDENT_PRESERVATION_COMMIT, "HEAD"),
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        ("git", "merge-base", "--is-ancestor", ORIGINAL_MAIN_TRAINING_COMMIT, "HEAD"),
        cwd=ROOT,
        check=True,
    )
    return {
        "passed": branch == "research/hybrid-rl-validation-v2",
        "branch": branch,
        "incident_commit_is_ancestor": True,
        "main_training_commit_is_ancestor": True,
    }


def verify_test_isolation() -> dict[str, Any]:
    tests = sorted((HERE / "tests").glob("test_*.py"))
    forbidden = (
        "evaluate_seed(",
        "evaluate_validation(",
        "environment.reset(",
        "StochasticTrack(",
        ".step(",
        "execute_next_v2_pair(",
    )
    violations = {
        path.name: [token for token in forbidden if token in path.read_text()]
        for path in tests
    }
    violations = {name: tokens for name, tokens in violations.items() if tokens}
    return {
        "passed": bool(tests) and not violations,
        "test_files": [path.name for path in tests],
        "forbidden_runtime_tokens": list(forbidden),
        "violations": violations,
    }


def build_preparation_status() -> dict[str, Any]:
    incident = verify_incident_evidence()
    sources = verify_frozen_scientific_sources()
    split = verify_regeneration(SPLIT_LEDGER)
    policy_metadata, _ = load_frozen_policies()
    pair_ledger = load_json(PAIR_LEDGER)
    validate_pair_ledger(pair_ledger)
    authorization = load_json(AUTHORIZATION)
    lineage = verify_git_lineage()
    tests = verify_test_isolation()
    split_payload = load_json(SPLIT_LEDGER)
    validate_registration(
        pair_ledger,
        split_payload,
        controller_identities=controller_identities(policy_metadata),
    )
    pair_states = {pair["state"] for pair in pair_ledger["pairs"]}
    gates = {
        "incident_evidence_integrity": incident["passed"],
        "v1_permanently_incident_aborted": authorization["v1_state"]
        == "INCIDENT_ABORTED",
        "deterministic_v2_split_regeneration": split[
            "deterministic_regeneration"
        ],
        "v2_route_and_full_task_clean": split["route_clean_verified"],
        "frozen_scientific_sources": sources["passed"],
        "five_controller_identities_frozen": pair_ledger["controller_order"]
        == ["CAUSAL60", "TIME_AWARE", "H11", "H29", "H47"]
        and set(pair_ledger["controller_identities"])
        == {"CAUSAL60", "TIME_AWARE", "H11", "H29", "H47"},
        "three_checkpoints_read_only_verified": len(policy_metadata) == 3
        and all(row["loaded_read_only"] for row in policy_metadata.values()),
        "actor_fingerprints_unchanged": all(
            policy_metadata[key]["actor_parameter_sha256"]
            == EXPECTED_ACTOR_SHA256[key]
            for key in POLICY_SEEDS
        ),
        "durable_pair_registration_complete": pair_ledger[
            "registration_complete"
        ]
        and len(pair_ledger["pairs"]) == 180,
        "pair_registration_bound_to_split_and_controllers": True,
        "all_pairs_pending": pair_states == {"PENDING"},
        "zero_v2_performance_episodes": pair_ledger[
            "performance_episode_count"
        ]
        == 0,
        "v2_execution_locked_and_unauthorized": authorization[
            "preparation_lock"
        ]
        and authorization["v2_preparation_complete"]
        and not authorization["v2_validation_authorized"]
        and not authorization["v2_validation_enabled"],
        "preparation_tests_statically_isolated": tests["passed"],
        "paper_fail_closed": split_payload["paper"]
        == {
            "state": "SEALED_UNINSPECTED",
            "numeric_exclusion_only": list(range(4000, 4018)),
            "content_fingerprints_constructed": 0,
            "episode_count": 0,
        },
        "git_lineage": lineage["passed"],
    }
    ready = all(gates.values())
    status = {
        "schema_version": 1,
        "study": V2_ID,
        "v1_state": "INCIDENT_ABORTED",
        "v2_state": "FROZEN_UNOPENED",
        "v2_preparation_complete": ready,
        "v2_validation_authorized": False,
        "v2_validation_enabled": False,
        "v2_completed_episodes": 0,
        "new_training_transitions": 0,
        "new_gradient_updates": 0,
        "paper_state": "SEALED_UNINSPECTED",
        "paper_episodes": 0,
        "paper_content_fingerprints": 0,
        "gates": gates,
        "incident": incident,
        "lineage": lineage,
        "split": split,
        "scientific_sources": sources,
        "policies": policy_metadata,
        "pair_ledger": {
            "run_transaction_id": pair_ledger["run_transaction_id"],
            "planned_pair_count": 180,
            "states": {"PENDING": 180},
            "performance_episode_count": 0,
        },
        "test_isolation": tests,
        "acceptance_thresholds": {
            "rsr_minimum_per_hybrid_seed": "35/36",
            "mean_paired_saving_vs_causal_percent": 10.0,
            "mean_paired_saving_vs_time_aware_percent": 3.0,
            "energy_subset": "jointly feasible trajectories only",
            "sampling_unit": "track instance; learned seeds are replications",
        },
    }
    atomic_json(PREPARATION_STATUS, status)
    return status


if __name__ == "__main__":
    print(json.dumps(build_preparation_status(), indent=2, sort_keys=True))
