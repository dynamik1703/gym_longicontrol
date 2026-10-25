"""Read-only checkpoint and native-runtime provenance verification."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.feedback_chunk_rl_v1.checkpointing import (
    load_checkpoint as load_feedback_checkpoint,
)
from benchmarks.feedback_chunk_rl_v1.sac_agent import ConstrainedResidualSAC
from benchmarks.hybrid_rl_validation_v2.integrity import load_frozen_policies

from .constants import (
    C_CHECKPOINTS,
    C_IDENTITIES,
    H_CHECKPOINTS,
    H_MANIFEST,
    H_MANIFEST_SHA256,
    H_SEEDS,
    POWER_MODEL_SHA256,
    ROOT,
    V1_LEDGER_SHA256,
    V1_MANIFEST,
    V1_MANIFEST_SHA256,
    V1_PAIR_LEDGER,
)
from .utils import canonical_sha256, file_sha256, load_json


def _distribution_manifest() -> list[dict[str, Any]]:
    rows = []
    for distribution in importlib.metadata.distributions():
        path = Path(distribution._path)  # noqa: SLF001 - provenance needs origin
        rows.append(
            {
                "name": str(distribution.metadata["Name"]),
                "version": str(distribution.version),
                "metadata_path": str(path),
                "metadata_kind": path.suffix,
                "direct_url": distribution.read_text("direct_url.json"),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            row["name"].lower(),
            row["version"],
            row["metadata_path"],
        ),
    )


def native_runtime_manifest() -> dict[str, Any]:
    executable = Path(sys.executable)
    resolved = executable.resolve()
    distributions = _distribution_manifest()
    manifest = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": str(executable),
        "resolved_executable": str(resolved),
        "executable_sha256": file_sha256(resolved),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "distributions": distributions,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def actor_parameter_sha256(state: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state["modules"]["actor"].items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def verify_source_hashes(expected: dict[str, str]) -> dict[str, Any]:
    observed = {relative: file_sha256(ROOT / relative) for relative in expected}
    if observed != expected:
        changed = sorted(key for key in expected if observed.get(key) != expected[key])
        raise RuntimeError(f"frozen checkpoint scientific sources changed: {changed}")
    return {"count": len(expected), "source_hashes": observed}


def verify_v1_incident_immutable() -> dict[str, Any]:
    if file_sha256(V1_PAIR_LEDGER) != V1_LEDGER_SHA256:
        raise RuntimeError("original V1 pair ledger bytes changed")
    ledger = load_json(V1_PAIR_LEDGER)
    counts = {
        state: sum(row["state"] == state for row in ledger["pairs"])
        for state in ("COMPLETED", "RUNNING", "NOT_STARTED")
    }
    expected = {"COMPLETED": 252, "RUNNING": 1, "NOT_STARTED": 647}
    if counts != expected:
        raise RuntimeError("original V1 pair ledger states changed")
    if file_sha256(V1_MANIFEST) != V1_MANIFEST_SHA256:
        raise RuntimeError("original V1 manifest bytes changed")
    return {
        "ledger_sha256": V1_LEDGER_SHA256,
        "manifest_sha256": V1_MANIFEST_SHA256,
        "pair_counts": counts,
        "validation_performance_episodes": ledger["validation_performance_episodes"],
        "paper_performance_episodes": ledger["paper_performance_episodes"],
    }


def load_c_agents() -> tuple[dict[str, Any], dict[str, ConstrainedResidualSAC]]:
    if file_sha256(V1_MANIFEST) != V1_MANIFEST_SHA256:
        raise RuntimeError("V1 training manifest changed")
    manifest = load_json(V1_MANIFEST)
    if manifest["state"] not in (
        "TRAINING_COMPLETED_DIAGNOSTIC_PENDING",
        "DIAGNOSTIC_RUNNING",
    ):
        raise RuntimeError("V1 training completion state changed")
    provenance = manifest["provenance"]
    verify_source_hashes(provenance["source_hashes"])
    metadata: dict[str, Any] = {}
    agents: dict[str, ConstrainedResidualSAC] = {}
    for controller_id, (method, seed) in C_IDENTITIES.items():
        run = manifest["runs"][f"{method}__seed-{seed}"]
        checkpoint = run["final_checkpoint"]
        expected = C_CHECKPOINTS[controller_id]
        if checkpoint["sha256"] != expected["sha256"]:
            raise RuntimeError(f"{controller_id} manifest checkpoint changed")
        if checkpoint["attempt_id"] != expected["attempt_id"]:
            raise RuntimeError(f"{controller_id} Attempt ID changed")
        path = Path(checkpoint["path"])
        before = file_sha256(path)
        if before != expected["sha256"]:
            raise RuntimeError(f"{controller_id} checkpoint bytes changed")
        loaded = load_feedback_checkpoint(
            path,
            expected_sha256=expected["sha256"],
            expected_configuration_sha256=provenance["canonical_sha256"],
            expected_source_hashes=provenance["source_hashes"],
            expected_attempt_id=expected["attempt_id"],
        )
        if loaded.runtime["native_transition_count"] != 300_000:
            raise RuntimeError(f"{controller_id} native budget changed")
        if loaded.runtime["gradient_update_count"] != 30_000:
            raise RuntimeError(f"{controller_id} update budget changed")
        agent = ConstrainedResidualSAC(method, seed)
        agent.load_state_dict(loaded.agent_state)
        if agent.update_count != 30_000:
            raise RuntimeError(f"{controller_id} agent update count changed")
        loaded.environment.close()
        after = file_sha256(path)
        if after != before:
            raise RuntimeError(f"{controller_id} checkpoint mutated during load")
        probe = np.linspace(0.05, 0.95, 16, dtype=np.float64)
        inference_one = agent.act(probe, deterministic=True).tolist()
        inference_two = agent.act(probe, deterministic=True).tolist()
        if inference_one != inference_two:
            raise RuntimeError(f"{controller_id} deterministic actor is unstable")
        metadata[controller_id] = {
            "worker_runtime": "C",
            "method": method,
            "training_seed": seed,
            "attempt_id": expected["attempt_id"],
            "checkpoint_path": str(path),
            "checkpoint_sha256": before,
            "actor_parameter_sha256": actor_parameter_sha256(loaded.agent_state),
            "native_training_transitions": 300_000,
            "gradient_updates": 30_000,
            "deterministic_probe": inference_one,
            "deterministic_probe_sha256": canonical_sha256(inference_one),
            "loaded_read_only": True,
        }
        agents[controller_id] = agent
    return metadata, agents


def load_h_agents() -> tuple[dict[str, Any], dict[str, Any]]:
    if file_sha256(H_MANIFEST) != H_MANIFEST_SHA256:
        raise RuntimeError("historical training manifest changed")
    manifest = load_json(H_MANIFEST)
    verify_source_hashes(manifest["provenance"]["source_hashes"])
    raw_metadata, raw_agents = load_frozen_policies(return_agents=True)
    metadata: dict[str, Any] = {}
    agents: dict[str, Any] = {}
    for controller_id, seed in H_SEEDS.items():
        short = f"H{seed}"
        expected = H_CHECKPOINTS[controller_id]
        row = raw_metadata[short]
        if row["checkpoint_sha256"] != expected["sha256"]:
            raise RuntimeError(f"{controller_id} checkpoint changed")
        run = manifest["runs"][str(seed)]
        if run["final_checkpoint"]["attempt_id"] != expected["attempt_id"]:
            raise RuntimeError(f"{controller_id} Attempt ID changed")
        agent = raw_agents[short]
        probe = np.linspace(0.05, 0.95, 16, dtype=np.float64)
        inference_one = float(agent.act(probe, deterministic=True))
        inference_two = float(agent.act(probe, deterministic=True))
        if inference_one != inference_two:
            raise RuntimeError(f"{controller_id} deterministic actor is unstable")
        metadata[controller_id] = {
            "worker_runtime": "H",
            "training_seed": seed,
            "attempt_id": expected["attempt_id"],
            "checkpoint_path": row["checkpoint_path"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "actor_parameter_sha256": row["actor_parameter_sha256"],
            "native_training_transitions": 300_000,
            "gradient_updates": 30_000,
            "deterministic_probe": [inference_one],
            "deterministic_probe_sha256": canonical_sha256([inference_one]),
            "loaded_read_only": True,
        }
        agents[controller_id] = agent
    return metadata, agents


def production_identity() -> dict[str, Any]:
    paths = (
        ROOT / "src/gym_longicontrol/domain/dynamics.py",
        ROOT / "src/gym_longicontrol/domain/task.py",
        ROOT / "src/gym_longicontrol/domain/vehicle.py",
        ROOT / "src/gym_longicontrol/envs/longicontrol.py",
        ROOT / "src/gym_longicontrol/assets/vehicle/BMW_electric_i3_2014.npz",
    )
    hashes = {str(path.relative_to(ROOT)): file_sha256(path) for path in paths}
    if hashes[str(paths[-1].relative_to(ROOT))] != POWER_MODEL_SHA256:
        raise RuntimeError("power-model artifact changed")
    return {"source_sha256": hashes, "power_model_sha256": POWER_MODEL_SHA256}


def git_object_hash(path: Path) -> str:
    return subprocess.run(
        ("git", "hash-object", str(path)),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C"},
    ).stdout.strip()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("runtime-manifest", "load-c", "load-h"))
    arguments = parser.parse_args(argv)
    if arguments.command == "runtime-manifest":
        print(json.dumps(native_runtime_manifest(), sort_keys=True))
    elif arguments.command == "load-c":
        metadata, _ = load_c_agents()
        print(json.dumps(metadata, sort_keys=True))
    else:
        metadata, _ = load_h_agents()
        print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
