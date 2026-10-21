"""Freeze the prospective protocol, route panels and scientific source hashes."""

from __future__ import annotations

import json
import subprocess
from typing import Any

from .execution import atomic_json, utc_now
from .integrity import CANONICAL, HERE, inherited_hashes, method_hashes
from .split import materialize

STATUS = HERE / "preparation_status.json"


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()


def prepare() -> dict[str, Any]:
    if CANONICAL.exists() or STATUS.exists():
        raise PermissionError("study preparation was already frozen")
    ledger = materialize()
    method = method_hashes()
    inherited = inherited_hashes()
    canonical = {
        "schema_version": 1,
        "study": "COUNTERFACTUAL_OPTION_VALUE_V1",
        "prepared_utc": utc_now(),
        "frozen_source_commit": "29b31837984b654e9011d3e180a8e33926345489",
        "preparation_commit": _source_commit(),
        "authorization": {
            "scientific_labels_authorized": True,
            "bounded_supervised_training_authorized": True,
            "full_rl_training_authorized": False,
            "actor_critic_authorized": False,
            "historical_validation_authorized": False,
            "paper_authorized": False,
        },
        "execution": {
            "preflight_complete": False,
            "scientific_labels_opened": False,
            "dataset_complete": False,
            "test_predictions_frozen": False,
            "test_labels_opened": False,
        },
        "physical_task": {
            "environment": "StochasticTrack-v1",
            "route_length_m": 1000.0,
            "deadline_s": 140.0,
            "speed_tolerance_m_s": 0.0,
            "native_dt_s": 0.1,
            "maximum_native_steps": 1800,
            "native_action_range": [-1.0, 1.0],
            "feasibility_authority": "is_feasible(EpisodeMetrics, TaskSpecification)",
            "safety_claim": "empirical causal Shield; not a global theorem",
        },
        "sampling": {
            "targets_m": [80, 180, 300, 430, 560, 700, 820, 930],
            "maximum_snapshots": 288,
            "maximum_alternatives_per_snapshot": 2,
            "route_group_partitioning": True,
        },
        "deadline_falsification_definition": (
            "retained completion_bound_s and deadline_viable must exactly match a "
            "fresh inherited pessimistic_completion_bound evaluation on the "
            "proposed-action successor at every retained native step"
        ),
        "predictor_preregistration": {
            "families": ["ridge_plus_logistic", "two_by_64_neural_ensemble"],
            "neural_seeds": [11, 29, 47],
            "neural_epochs": 300,
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "energy_margin_wh": [0.2, 0.5, 1.0],
            "risk_threshold": [0.05, 0.1, 0.2],
            "fit_only_updates": True,
            "calibration_only_selection": True,
        },
        "go_no_go": {
            "test_opportunity_fraction_minimum": 0.20,
            "meaningful_saving_wh": 0.2,
            "regret_must_improve_vs_reference": True,
            "selected_net_energy_benefit_required": True,
            "new_hard_failures_allowed": 0,
            "minimum_valid_test_routes": 10,
            "minimum_valid_test_snapshots": 72,
            "minimum_routes_with_useful_selected_deviation": 3,
            "maximum_single_route_share_of_selected_saving": 0.5,
        },
        "resource_limits": {
            "active_compute_hours_maximum": 12.0,
            "retained_artifacts_bytes_maximum": 4_000_000_000,
            "sampled_state_maximum": 288,
            "alternative_branch_maximum_per_state": 2,
        },
        "split_ledger_sha256": __import__("hashlib")
        .sha256((HERE / "split_ledger.json").read_bytes())
        .hexdigest(),
        "panel_hashes": {
            key: ledger[key]["ordered_sha256"]
            for key in ("fit_panel", "calibration_panel", "test_panel")
        },
        "frozen_method_sha256": method,
        "inherited_source_sha256": inherited,
        "paper_state": "SEALED_UNINSPECTED",
        "temporal_options_v1_classification": "SEMANTIC_OR_INTEGRITY_STOP",
    }
    atomic_json(CANONICAL, canonical)
    source_lines = [
        "# Source Audit",
        "",
        "Frozen source: `29b31837984b654e9011d3e180a8e33926345489`.",
        "Temporal Options V1 remains read-only and classified",
        "`SEMANTIC_OR_INTEGRITY_STOP`.",
        "",
        "## New method files",
        "",
    ]
    source_lines.extend(f"- `{name}` — `{digest}`" for name, digest in method.items())
    source_lines.extend(["", "## Inherited scientific files", ""])
    source_lines.extend(
        f"- `{name}` — `{digest}`" for name, digest in inherited.items()
    )
    (HERE / "SOURCE_AUDIT.md").write_text(
        "\n".join(source_lines) + "\n", encoding="utf-8"
    )
    status = {
        "schema_version": 1,
        "state": "METHOD_AND_SPLITS_FROZEN_PREFLIGHT_PENDING",
        "route_clean_verified": ledger["route_clean_verified"],
        "panel_hashes": canonical["panel_hashes"],
        "method_file_count": len(method),
        "inherited_file_count": len(inherited),
        "scientific_labels_generated": 0,
        "test_labels_opened": False,
        "paper_state": "SEALED_UNINSPECTED",
    }
    atomic_json(STATUS, status)
    return status


if __name__ == "__main__":
    print(json.dumps(prepare(), sort_keys=True))
