from __future__ import annotations

import ast
import json
from pathlib import Path

from benchmarks.paper_synthesis_v1 import verify


def test_repository_provenance_and_historical_hashes() -> None:
    registry = verify.load_json(verify.REGISTRY)
    verify.verify_repository_provenance(registry)
    verify.verify_historical_hashes(registry)


def test_generated_artifacts_reconcile() -> None:
    verify.verify_generated_artifacts()


def test_analysis_cannot_invoke_environment_evaluation() -> None:
    verify.verify_analysis_has_no_scientific_entrypoints()


def test_paper_seal_is_checked_only_from_status_metadata() -> None:
    source = (verify.PACKAGE_DIR / "verify.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    opened_literals = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_json"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    assert not opened_literals
    verify.verify_paper_seal_metadata_only()


def test_preparation_status_denies_execution_authorization() -> None:
    verify.verify_preparation_status()


def test_local_checkpoint_audit_never_deserializes_models(tmp_path: Path) -> None:
    checkpoint = tmp_path / "opaque.bin"
    checkpoint.write_bytes(b"opaque checkpoint bytes")
    registry = {
        "local_checkpoint_byte_audit": {
            "candidate": {
                "path": str(checkpoint),
                "sha256": verify.sha256(checkpoint),
            },
            "missing": {
                "path": str(tmp_path / "missing.bin"),
                "sha256": "0" * 64,
            },
        }
    }
    assert verify.local_checkpoint_status(registry) == {
        "candidate": "AVAILABLE_HASH_MATCH",
        "missing": "NOT_AVAILABLE_ON_THIS_HOST",
    }


def test_registry_uses_not_recorded_instead_of_null_imputation() -> None:
    registry = json.loads(
        (verify.PACKAGE_DIR / "method_registry.json").read_text(encoding="utf-8")
    )
    required = {
        "method_name",
        "algorithm_family",
        "source_commit",
        "observation_contract",
        "action_semantics",
        "physics_information",
        "comparison_basis",
        "authoritative_evidence_paths",
    }
    assert all(required <= row.keys() for row in registry["methods"])
    assert any("NOT_RECORDED" in row.values() for row in registry["methods"])
