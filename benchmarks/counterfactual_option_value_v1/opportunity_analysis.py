"""Descriptive opportunity analysis with privileged hindsight clearly labeled."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from typing import Any

from .execution import atomic_json, sha256_file
from .integrity import BRANCH_LEDGER, HERE, verify_frozen_method
from .predictor import MANIFEST
from .predictor_evaluation import PREDICTION_FREEZE

OUTPUT = HERE / "opportunity_analysis.json"


def _quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "max": ordered[-1],
    }


def _load_pairs() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not PREDICTION_FREEZE.exists():
        raise PermissionError(
            "opportunity analysis waits for blind Test prediction freeze"
        )
    ledger = json.loads(BRANCH_LEDGER.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    snapshots = {row["snapshot_id"]: row for row in manifest["snapshots"]}
    pairs = []
    for item in ledger["pair_records"]:
        path = HERE / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError("immutable pair changed")
        pairs.append(json.loads(path.read_text(encoding="utf-8")))
    return pairs, snapshots


def _group_summary(
    pairs: list[dict[str, Any]], snapshots: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    valid_states = 0
    saving_any = saving_02 = saving_05 = 0
    best_advantages_wh: list[float] = []
    harmful_wh: list[float] = []
    introduced = alternatives = 0
    distinguishable_valid = distinguishable_saving_02 = 0
    action_family = defaultdict(
        lambda: {"count": 0, "saving_count": 0, "saving_wh": []}
    )
    continue_opportunity = {"count": 0, "saving_count": 0, "saving_wh": []}
    start_opportunity = {"count": 0, "saving_count": 0, "saving_wh": []}
    phase = defaultdict(lambda: {"states": 0, "saving_02": 0})
    by_progress = defaultdict(lambda: {"states": 0, "saving_02": 0})
    route = defaultdict(lambda: {"states": 0, "saving_02": 0, "best_wh": []})
    for pair in pairs:
        snapshot = snapshots[pair["snapshot_id"]]
        valid = [
            row
            for row in pair["alternatives"]
            if pair["reference_feasible"] and row["feasible"]
        ]
        introduced += sum(
            row["introduced_infeasibility"] for row in pair["alternatives"]
        )
        alternatives += len(pair["alternatives"])
        for row in valid:
            delta_wh = float(row["delta_energy_kwh"]) * 1_000.0
            if delta_wh > 0.0:
                harmful_wh.append(delta_wh)
            metadata = next(
                item
                for item in snapshot["candidate_metadata"]
                if item["action_id"] == row["action_id"]
            )
            family = metadata["family"]
            action_family[family]["count"] += 1
            if delta_wh < 0.0:
                action_family[family]["saving_count"] += 1
                action_family[family]["saving_wh"].append(-delta_wh)
            category = (
                continue_opportunity
                if row["action_id"] == "CONTINUE"
                else start_opportunity
            )
            category["count"] += 1
            if delta_wh < 0.0:
                category["saving_count"] += 1
                category["saving_wh"].append(-delta_wh)
        if not pair["reference_feasible"]:
            continue
        valid_states += 1
        best_wh = max(
            [0.0, *(-float(row["delta_energy_kwh"]) * 1_000.0 for row in valid)]
        )
        best_advantages_wh.append(best_wh)
        saving_any += int(best_wh > 0.0)
        saving_02 += int(best_wh >= 0.2)
        saving_05 += int(best_wh >= 0.5)
        distinguishable = [
            row
            for row in valid
            if row["first_block_executed_behavior_differs"]
            or row["first_block_option_memory_differs"]
        ]
        if distinguishable:
            distinguishable_valid += 1
            distinguishable_best = max(
                [
                    0.0,
                    *(
                        -float(row["delta_energy_kwh"]) * 1_000.0
                        for row in distinguishable
                    ),
                ]
            )
            distinguishable_saving_02 += int(distinguishable_best >= 0.2)
        active = snapshot["active_option"]
        phase_name = "inactive" if active is None else active["phase"]
        phase[phase_name]["states"] += 1
        phase[phase_name]["saving_02"] += int(best_wh >= 0.2)
        progress_bin = f"{int(pair['nominal_target_m'])}m"
        by_progress[progress_bin]["states"] += 1
        by_progress[progress_bin]["saving_02"] += int(best_wh >= 0.2)
        seed = str(pair["numeric_seed"])
        route[seed]["states"] += 1
        route[seed]["saving_02"] += int(best_wh >= 0.2)
        route[seed]["best_wh"].append(best_wh)

    def category_result(value: dict[str, Any]) -> dict[str, Any]:
        return {
            "candidate_count": value["count"],
            "saving_count": value["saving_count"],
            "saving_fraction": (
                value["saving_count"] / value["count"] if value["count"] else None
            ),
            "saving_wh_distribution": _quantiles(value["saving_wh"]),
        }

    return {
        "valid_reference_snapshot_count": valid_states,
        "fraction_any_feasible_saving": saving_any / valid_states
        if valid_states
        else None,
        "fraction_saving_at_least_0_2wh": saving_02 / valid_states
        if valid_states
        else None,
        "fraction_saving_at_least_0_5wh": saving_05 / valid_states
        if valid_states
        else None,
        "best_feasible_advantage_wh": _quantiles(best_advantages_wh),
        "harmful_alternative_wh": _quantiles(harmful_wh),
        "introduced_infeasibility_count": introduced,
        "alternative_count": alternatives,
        "introduced_infeasibility_fraction": introduced / alternatives
        if alternatives
        else None,
        "distinguishable_behavior": {
            "valid_snapshot_count": distinguishable_valid,
            "saving_0_2wh_count": distinguishable_saving_02,
            "saving_0_2wh_fraction_of_valid_states": (
                distinguishable_saving_02 / valid_states if valid_states else None
            ),
        },
        "continue": category_result(continue_opportunity),
        "start": category_result(start_opportunity),
        "by_action_family": {
            key: category_result(value) for key, value in sorted(action_family.items())
        },
        "by_option_phase": dict(sorted(phase.items())),
        "by_progress": dict(sorted(by_progress.items())),
        "by_route": {
            key: {**value, "best_wh": _quantiles(value["best_wh"])}
            for key, value in sorted(route.items())
        },
    }


def analyze() -> dict[str, Any]:
    verify_frozen_method(require_labels_authorized=True)
    if OUTPUT.exists():
        raise PermissionError("opportunity analysis already exists")
    pairs, snapshots = _load_pairs()
    panels = {}
    for panel_id in (
        "CF_OPTION_FIT_V1",
        "CF_OPTION_CALIBRATION_V1",
        "CF_OPTION_TEST_V1",
    ):
        panel_pairs = [pair for pair in pairs if pair["panel_id"] == panel_id]
        panels[panel_id] = _group_summary(panel_pairs, snapshots)
    result = {
        "schema_version": 1,
        "interpretation": (
            "COUNTERFACTUAL_HINDSIGHT_DIAGNOSTIC; independent one-deviation "
            "advantages are not additive and are neither a causal policy nor an "
            "upper bound"
        ),
        "panels": panels,
        "all_internal_panels": _group_summary(pairs, snapshots),
    }
    atomic_json(OUTPUT, result)
    test = panels["CF_OPTION_TEST_V1"]
    lines = [
        "# Opportunity Analysis",
        "",
        "All best-alternative values below are privileged",
        "`COUNTERFACTUAL_HINDSIGHT_DIAGNOSTIC` measurements. They are not additive",
        "across states and are not a causal learned policy or certified upper bound.",
        "",
        "## Blind Test panel",
        "",
        f"- Valid reference snapshots: {test['valid_reference_snapshot_count']}",
        f"- Any feasible saving: {test['fraction_any_feasible_saving']}",
        f"- Saving at least 0.2 Wh: {test['fraction_saving_at_least_0_2wh']}",
        f"- Saving at least 0.5 Wh: {test['fraction_saving_at_least_0_5wh']}",
        f"- New infeasibility fraction: {test['introduced_infeasibility_fraction']}",
        "- Best feasible advantage distribution (Wh): "
        f"{test['best_feasible_advantage_wh']}",
        "",
        "START/CONTINUE, route, progress, phase and action-family breakdowns are",
        "retained in `opportunity_analysis.json`.",
        "",
    ]
    (HERE / "OPPORTUNITY_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(
        json.dumps(
            {"complete": True, "panels": list(analyze()["panels"])}, sort_keys=True
        )
    )
