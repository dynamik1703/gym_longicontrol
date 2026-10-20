"""Future matched diagnostics and sealed Validation execution."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.constrained_rl_v2.adapter import build_agent as build_constrained_agent
from benchmarks.constrained_v2_collector_replication.checkpointing import (
    restore_policy as restore_constrained_policy,
)
from benchmarks.scalar_sac.experiment import _base_environment

from .agent import build_scalar_agent
from .checkpointing import file_sha256, load_verified, restore_policy
from .config import CHECKPOINTS, TRAINING_SEEDS, load_configuration
from .evaluation import evaluate_diagnostic, evaluate_validation
from .execution import atomic_json, condition_name, load_or_create_manifest
from .reference import reference_checkpoint, verify_references


def _scalar_payload(configuration, manifest, output_root: Path, seed: int, step: int):
    run = manifest["runs"][condition_name(seed)]
    if run["state"] != "COMPLETED":
        raise PermissionError("All Scalar final policies must be frozen first")
    path = output_root / condition_name(seed) / f"step-{step:06d}" / "checkpoint.pt"
    metadata = json.loads(path.with_name("checkpoint.pt.metadata.json").read_text())
    if file_sha256(path) != metadata["sha256"]:
        raise RuntimeError("Scalar scientific checkpoint hash changed")
    return load_verified(
        path,
        expected_sha256=metadata["sha256"],
        expected_configuration_sha256=configuration.configuration_sha256,
    )


def _policy(configuration, *, method: str, seed: int, step: int, scalar_payload=None):
    environment = _base_environment(configuration.constrained_v2)
    try:
        if method == "constrained":
            agent, _ = build_constrained_agent(
                configuration.constrained_v2,
                environment,
                training_seed=seed,
                device="cpu",
                threads=4,
            )
            from benchmarks.constrained_v2_collector_replication.checkpointing import (
                load_verified as load_constrained,
            )

            expected = configuration.raw["reference_checkpoints"][str(seed)][str(step)]
            payload = load_constrained(
                reference_checkpoint(configuration, seed, step),
                expected_sha256=expected,
            )
            restore_constrained_policy(agent.policy, payload)
        elif method == "scalar":
            agent, _ = build_scalar_agent(
                environment, training_seed=seed, device="cpu", threads=4
            )
            restore_policy(agent.policy, scalar_payload)
        else:
            raise ValueError("Unknown method")
        return agent.policy, environment
    except Exception:
        environment.close()
        raise


def run_reference_diagnostics(output_root: str | Path) -> None:
    configuration = load_configuration()
    verify_references(configuration)
    root = Path(output_root)
    manifest_path, manifest = load_or_create_manifest(root)
    if any(row["state"] != "COMPLETED" for row in manifest["runs"].values()):
        raise PermissionError(
            "Reference diagnostics wait for all Scalar final policies"
        )
    count = 0
    for seed in TRAINING_SEEDS:
        for step in CHECKPOINTS:
            destination = (
                root / f"C{seed}_reference" / f"step-{step:06d}" / "diagnostic.json"
            )
            if destination.exists():
                count += 18
                continue
            policy, environment = _policy(
                configuration, method="constrained", seed=seed, step=step
            )
            try:
                result = evaluate_diagnostic(
                    policy, configuration, real_transitions=step
                )
            finally:
                environment.close()
            result.update(
                {
                    "method": "Constrained V2 reference",
                    "training_seed": seed,
                    "checkpoint_sha256": configuration.raw["reference_checkpoints"][
                        str(seed)
                    ][str(step)],
                }
            )
            atomic_json(destination, result)
            count += 18
    manifest["reference_diagnostic"] = {
        "state": "COMPLETED",
        "episode_count": count,
    }
    atomic_json(manifest_path, manifest)


def run_validation_matrix(output_root: str | Path) -> None:
    configuration = load_configuration()
    from .config import require_validation_authorized

    require_validation_authorized(configuration)
    verify_references(configuration)
    root = Path(output_root)
    manifest_path, manifest = load_or_create_manifest(root)
    if any(row["state"] != "COMPLETED" for row in manifest["runs"].values()):
        raise PermissionError("Validation waits for all Scalar final policies")
    completed = []
    for method in ("constrained", "scalar"):
        for seed in TRAINING_SEEDS:
            payload = (
                None
                if method == "constrained"
                else _scalar_payload(configuration, manifest, root, seed, 300_000)
            )
            policy, environment = _policy(
                configuration,
                method=method,
                seed=seed,
                step=300_000,
                scalar_payload=payload,
            )
            try:
                result = evaluate_validation(policy, configuration)
            finally:
                environment.close()
            destination = root / f"{method}-{seed}-validation.json"
            result.update({"method": method, "training_seed": seed})
            atomic_json(destination, result)
            completed.append(
                {"method": method, "training_seed": seed, "path": str(destination)}
            )
    manifest["validation"] = {
        "state": "COMPLETED",
        "episode_count": 108,
        "completed_conditions": completed,
    }
    atomic_json(manifest_path, manifest)
