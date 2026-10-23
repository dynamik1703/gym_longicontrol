"""Read-only integrity verifier for the paper-synthesis package."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from . import evidence, feasibility

PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]
REGISTRY = PACKAGE_DIR / "source_registry.json"
PACKAGE_PREFIX = "benchmarks/paper_synthesis_v1/"

REQUIRED_FILES = (
    "README.md",
    "SOURCE_REGISTRY.md",
    "METHOD_REGISTRY.md",
    "method_registry.json",
    "FEASIBILITY_PROOF.md",
    "FEASIBILITY_AUDIT.md",
    "feasibility_audit.json",
    "historical_task_status.csv",
    "MASTER_EVIDENCE_MATRIX.md",
    "master_evidence_matrix.csv",
    "paired_comparison_registry.json",
    "CLAIM_EVIDENCE_MATRIX.md",
    "PAPER_TITLE_ABSTRACT.md",
    "PAPER_OUTLINE.md",
    "FIGURE_TABLE_PLAN.md",
    "THREATS_TO_VALIDITY.md",
    "RELATED_WORK_GAPS.md",
    "PAPER_EVAL_PROTOCOL_DRAFT.md",
    "PAPER_EVAL_ARM_PROPOSAL.md",
    "INTEGRITY_REPORT.md",
    "OPEN_DECISIONS.md",
    "preparation_status.json",
    "source_registry.json",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.rstrip()


def verify_repository_provenance(registry: dict[str, Any]) -> None:
    base_commit = registry["base_commit"]
    assert git("rev-parse", f"{base_commit}^{{commit}}") == base_commit
    assert git("rev-parse", f"{base_commit}^{{tree}}") == registry["base_tree"]
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", base_commit, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    changed = git("diff", "--name-only", base_commit, "--").splitlines()
    assert all(path.startswith(PACKAGE_PREFIX) for path in changed)

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    for line in status.splitlines():
        path = line[3:]
        assert path.startswith(PACKAGE_PREFIX), path


def verify_historical_hashes(registry: dict[str, Any]) -> None:
    for relative, expected in registry["historical_file_sha256"].items():
        path = REPOSITORY_ROOT / relative
        assert path.is_file(), relative
        assert sha256(path) == expected, relative


def local_checkpoint_status(registry: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, record in registry["local_checkpoint_byte_audit"].items():
        path = Path(record["path"])
        if not path.is_file():
            result[name] = "NOT_AVAILABLE_ON_THIS_HOST"
        elif sha256(path) == record["sha256"]:
            result[name] = "AVAILABLE_HASH_MATCH"
        else:
            result[name] = "AVAILABLE_HASH_MISMATCH"
    return result


def verify_generated_artifacts() -> None:
    feasibility.main(["--check"])
    evidence.main(["--check"])
    for name in REQUIRED_FILES:
        assert (PACKAGE_DIR / name).is_file(), name

    audit = load_json(PACKAGE_DIR / "feasibility_audit.json")
    coverage = audit["coverage"]
    assert coverage["unique_routes"] == 411
    assert coverage["unique_full_tasks"] == 411
    assert coverage["study_task_occurrences"] == 424
    assert coverage["classification_all_inspectable"] == {
        "FEASIBLE_WITNESS": 187,
        "PROVEN_INFEASIBLE": 9,
        "UNRESOLVED": 215,
    }
    assert len(audit["tasks"]) == 411

    methods = load_json(PACKAGE_DIR / "method_registry.json")
    comparisons = load_json(PACKAGE_DIR / "paired_comparison_registry.json")
    assert methods["row_count"] == len(methods["methods"]) == 34
    assert comparisons["comparison_count"] == len(comparisons["comparisons"]) == 9


def verify_analysis_has_no_scientific_entrypoints() -> None:
    for name in ("feasibility.py", "evidence.py"):
        tree = ast.parse((PACKAGE_DIR / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in {"reset", "step"}, name
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert "gym_longicontrol.envs" not in module, name
            if isinstance(node, ast.Import):
                for imported in node.names:
                    assert "gym_longicontrol.envs" not in imported.name, name


def verify_paper_seal_metadata_only() -> None:
    hybrid = load_json(
        REPOSITORY_ROOT / "benchmarks/hybrid_rl_validation_v2/preparation_status.json"
    )
    assert hybrid["paper_state"] == "SEALED_UNINSPECTED"
    assert hybrid["paper_episodes"] == 0
    assert hybrid["paper_content_fingerprints"] == 0

    execution = load_json(
        REPOSITORY_ROOT / "benchmarks/hybrid_rl_execution/preparation_status.json"
    )
    assert execution["paper_authorized"] is False
    assert execution["paper"]["state"] == "SEALED_UNINSPECTED"
    assert execution["paper"]["episode_count"] == 0
    assert execution["paper"]["content_fingerprints_constructed"] == 0

    causal = load_json(
        REPOSITORY_ROOT / "benchmarks/causal_mpc_v2/preparation_status.json"
    )
    assert causal["paper_state"] == "SEALED_UNINSPECTED"

    gcsl = load_json(
        REPOSITORY_ROOT / "benchmarks/gcsl_canonical/preparation_status.json"
    )
    assert gcsl["authorization"]["paper_test_authorized"] is False
    assert gcsl["split_integrity"]["paper_state"] == "SEALED_UNINSPECTED"
    assert gcsl["split_integrity"]["paper_content_fingerprints_constructed"] == 0
    assert gcsl["preparation_interactions"]["paper_episodes"] == 0

    counterfactual = load_json(
        REPOSITORY_ROOT
        / "benchmarks/counterfactual_option_value_v1/preparation_status.json"
    )
    assert counterfactual["paper_state"] == "SEALED_UNINSPECTED"


def verify_preparation_status() -> None:
    status = load_json(PACKAGE_DIR / "preparation_status.json")
    assert status["state"] == "DRAFT_COMPLETE_FOR_PROTOCOL_REVIEW"
    assert status["paper"]["state"] == "SEALED_UNINSPECTED"
    assert status["paper"]["authorized"] is False
    assert status["paper"]["episodes"] == 0
    assert status["paper"]["content_fingerprints_constructed"] == 0
    assert status["new_scientific_activity"] == {
        "controller_evaluation_episodes": 0,
        "environment_resets": 0,
        "environment_steps": 0,
        "gradient_updates": 0,
        "training_episodes": 0,
    }


def main() -> int:
    registry = load_json(REGISTRY)
    verify_repository_provenance(registry)
    verify_historical_hashes(registry)
    verify_generated_artifacts()
    verify_analysis_has_no_scientific_entrypoints()
    verify_paper_seal_metadata_only()
    verify_preparation_status()
    checkpoints = local_checkpoint_status(registry)
    mismatches = [
        name for name, state in checkpoints.items() if state.endswith("MISMATCH")
    ]
    assert not mismatches, mismatches
    print("paper_synthesis_integrity=PASS")
    print("local_checkpoint_status=" + json.dumps(checkpoints, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
