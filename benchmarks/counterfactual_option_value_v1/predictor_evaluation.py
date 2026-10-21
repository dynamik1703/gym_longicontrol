"""Blind Test prediction freeze followed by immutable outcome scoring."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from typing import Any

import numpy as np

from .execution import atomic_json, sha256_file, utc_now
from .integrity import BRANCH_LEDGER, HERE, verify_frozen_method
from .predictor import (
    ARTIFACT,
    MANIFEST,
    MODEL_SELECTION,
    FrozenPredictor,
    feature_vector,
)

PREDICTIONS = HERE / "test_predictions.json"
PREDICTION_FREEZE = HERE / "test_prediction_freeze.json"
EVALUATION = HERE / "test_evaluation.json"
TEST_PANEL = "CF_OPTION_TEST_V1"


def generate_test_predictions() -> dict[str, Any]:
    """Generate predictions from causal manifest fields without opening labels."""

    verify_frozen_method(require_labels_authorized=True)
    if PREDICTIONS.exists() or PREDICTION_FREEZE.exists():
        raise PermissionError("Test predictions are already frozen")
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    selection = json.loads(MODEL_SELECTION.read_text(encoding="utf-8"))
    if selection["artifact_sha256"] != sha256_file(ARTIFACT):
        raise RuntimeError("predictor artifact changed after Calibration")
    selected = selection["selected"]
    predictor = FrozenPredictor(artifact, selected["family"])
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    snapshots = [row for row in manifest["snapshots"] if row["panel_id"] == TEST_PANEL]
    predictions = []
    for snapshot in snapshots:
        actions = snapshot["alternative_action_ids"]
        features = np.stack([feature_vector(snapshot, action) for action in actions])
        energy, risk = predictor.predict(features)
        candidates = [
            {
                "action_id": action,
                "predicted_delta_energy_kwh": float(delta),
                "predicted_infeasibility_risk": float(probability),
            }
            for action, delta, probability in zip(actions, energy, risk, strict=True)
        ]
        eligible = [
            row
            for row in candidates
            if row["predicted_delta_energy_kwh"]
            <= -float(selected["margin_wh"]) / 1_000.0
            and row["predicted_infeasibility_risk"] <= float(selected["risk_threshold"])
        ]
        choice = min(
            eligible,
            key=lambda row: (
                row["predicted_delta_energy_kwh"],
                row["predicted_infeasibility_risk"],
                row["action_id"],
            ),
            default=None,
        )
        predictions.append(
            {
                "snapshot_id": snapshot["snapshot_id"],
                "numeric_seed": snapshot["numeric_seed"],
                "nominal_target_m": snapshot["nominal_target_m"],
                "reference_action_id": snapshot["reference_action_id"],
                "heuristic_runner_up_action_id": snapshot[
                    "heuristic_runner_up_action_id"
                ],
                "candidate_predictions": candidates,
                "selected_action_id": None if choice is None else choice["action_id"],
                "abstained": choice is None,
            }
        )
    payload = {
        "schema_version": 1,
        "panel_id": TEST_PANEL,
        "family": selected["family"],
        "margin_wh": selected["margin_wh"],
        "risk_threshold": selected["risk_threshold"],
        "prediction_count": len(predictions),
        "predictions": predictions,
        "test_outcome_labels_loaded": False,
    }
    atomic_json(PREDICTIONS, payload)
    freeze = {
        "schema_version": 1,
        "frozen_utc": utc_now(),
        "predictor_artifact_sha256": sha256_file(ARTIFACT),
        "model_selection_sha256": sha256_file(MODEL_SELECTION),
        "test_predictions_sha256": sha256_file(PREDICTIONS),
        "normalization_frozen": True,
        "weights_frozen": True,
        "thresholds_frozen": True,
        "test_labels_opened": False,
    }
    atomic_json(PREDICTION_FREEZE, freeze)
    return payload


def _test_pairs() -> dict[str, dict[str, Any]]:
    ledger = json.loads(BRANCH_LEDGER.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    test_ids = {
        row["snapshot_id"]
        for row in manifest["snapshots"]
        if row["panel_id"] == TEST_PANEL
    }
    pairs: dict[str, dict[str, Any]] = {}
    for item in ledger["pair_records"]:
        if item["snapshot_id"] not in test_ids:
            continue
        path = HERE / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError("immutable Test counterfactual record changed")
        pairs[item["snapshot_id"]] = json.loads(path.read_text(encoding="utf-8"))
    return pairs


def _bootstrap_route_mean(
    route_values: dict[int, float], *, seed: int = 20261021, draws: int = 2_000
) -> dict[str, float | int | None]:
    if not route_values:
        return {"route_count": 0, "mean": None, "ci95": None, "draws": draws}
    values = np.asarray(list(route_values.values()), dtype=np.float64)
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(draws, len(values)), replace=True).mean(
        axis=1
    )
    return {
        "route_count": len(values),
        "mean": float(values.mean()),
        "ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "draws": draws,
    }


def score_test_predictions() -> dict[str, Any]:
    verify_frozen_method(require_labels_authorized=True)
    if EVALUATION.exists():
        raise PermissionError("Test was already scored")
    freeze = json.loads(PREDICTION_FREEZE.read_text(encoding="utf-8"))
    if (
        freeze["predictor_artifact_sha256"] != sha256_file(ARTIFACT)
        or freeze["model_selection_sha256"] != sha256_file(MODEL_SELECTION)
        or freeze["test_predictions_sha256"] != sha256_file(PREDICTIONS)
    ):
        raise RuntimeError("blind Test freeze hash mismatch")
    prediction_payload = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    pairs = _test_pairs()  # First Test-label read occurs after all freeze checks.
    absolute_errors = []
    meaningful_sign = []
    ranking = []
    selected_rows = []
    route_delta: dict[int, float] = defaultdict(float)
    route_regret_improvement: dict[int, float] = defaultdict(float)
    totals = defaultdict(float)
    route_useful: set[int] = set()
    one_deviation = []
    for prediction in prediction_payload["predictions"]:
        pair = pairs[prediction["snapshot_id"]]
        actual = {row["action_id"]: row for row in pair["alternatives"]}
        predicted = {
            row["action_id"]: row for row in prediction["candidate_predictions"]
        }
        for action_id, row in actual.items():
            if pair["reference_feasible"] and row["feasible"]:
                error = abs(
                    predicted[action_id]["predicted_delta_energy_kwh"]
                    - row["delta_energy_kwh"]
                )
                absolute_errors.append(error)
                if abs(row["delta_energy_kwh"]) >= 0.0002:
                    meaningful_sign.append(
                        (predicted[action_id]["predicted_delta_energy_kwh"] < 0.0)
                        == (row["delta_energy_kwh"] < 0.0)
                    )
        feasible_actual = {
            action_id: row["delta_energy_kwh"]
            for action_id, row in actual.items()
            if pair["reference_feasible"] and row["feasible"]
        }
        actual_best_action, actual_best = min(
            [(None, 0.0), *feasible_actual.items()],
            key=lambda item: (item[1], str(item[0])),
        )
        predicted_best_action, _ = min(
            [
                (None, 0.0),
                *(
                    (action_id, row["predicted_delta_energy_kwh"])
                    for action_id, row in predicted.items()
                ),
            ],
            key=lambda item: (item[1], str(item[0])),
        )
        ranking.append(predicted_best_action == actual_best_action)
        chosen_id = prediction["selected_action_id"]
        chosen = None if chosen_id is None else actual[chosen_id]
        selected_delta = 0.0
        false_safe = harmful = False
        if chosen is not None:
            false_safe = bool(pair["reference_feasible"] and not chosen["feasible"])
            if pair["reference_feasible"] and chosen["feasible"]:
                selected_delta = float(chosen["delta_energy_kwh"])
                harmful = selected_delta > 0.0
                if selected_delta < 0.0:
                    route_useful.add(int(pair["numeric_seed"]))
        reference_regret = -float(actual_best)
        selected_regret = selected_delta - float(actual_best)
        runner_id = prediction["heuristic_runner_up_action_id"]
        runner = actual.get(runner_id)
        runner_delta = (
            float(runner["delta_energy_kwh"])
            if runner is not None and pair["reference_feasible"] and runner["feasible"]
            else 0.0
        )
        runner_regret = runner_delta - float(actual_best)
        seed = int(pair["numeric_seed"])
        route_delta[seed] += selected_delta
        route_regret_improvement[seed] += reference_regret - selected_regret
        totals["reference_regret_kwh"] += reference_regret
        totals["selected_regret_kwh"] += selected_regret
        totals["runner_up_regret_kwh"] += runner_regret
        totals["hindsight_delta_energy_kwh"] += float(actual_best)
        totals["selected_delta_energy_kwh"] += selected_delta
        totals["accepted"] += int(chosen is not None)
        totals["false_safe"] += int(false_safe)
        totals["harmful"] += int(harmful)
        selected_row = {
            "snapshot_id": pair["snapshot_id"],
            "numeric_seed": seed,
            "nominal_target_m": pair["nominal_target_m"],
            "selected_action_id": chosen_id,
            "actual_delta_energy_kwh": selected_delta,
            "false_safe": false_safe,
            "harmful": harmful,
            "reference_regret_kwh": reference_regret,
            "selected_regret_kwh": selected_regret,
            "counterfactual_hindsight_diagnostic_delta_kwh": actual_best,
        }
        selected_rows.append(selected_row)
        if float(pair["nominal_target_m"]) == 430.0:
            predicted_choice = None if chosen_id is None else predicted[chosen_id]
            one_deviation.append(
                {
                    **selected_row,
                    "predicted_delta_energy_kwh": (
                        None
                        if predicted_choice is None
                        else predicted_choice["predicted_delta_energy_kwh"]
                    ),
                    "predicted_infeasibility_risk": (
                        None
                        if predicted_choice is None
                        else predicted_choice["predicted_infeasibility_risk"]
                    ),
                    "reference_feasible": pair["reference_feasible"],
                    "selected_feasible": (
                        pair["reference_feasible"]
                        if chosen is None
                        else chosen["feasible"]
                    ),
                }
            )
    count = len(prediction_payload["predictions"])
    result = {
        "schema_version": 1,
        "panel_id": TEST_PANEL,
        "blind_freeze": {
            **freeze,
            "test_labels_opened_after_freeze": True,
        },
        "energy_difference_mae_kwh": (
            statistics.fmean(absolute_errors) if absolute_errors else None
        ),
        "action_ranking_accuracy": statistics.fmean(ranking) if ranking else None,
        "meaningful_sign_accuracy_at_0_2wh": (
            statistics.fmean(meaningful_sign) if meaningful_sign else None
        ),
        "meaningful_sign_count": len(meaningful_sign),
        "snapshot_count": count,
        "route_count": len({row["numeric_seed"] for row in selected_rows}),
        "accepted_deviation_count": int(totals["accepted"]),
        "abstention_rate": 1.0 - float(totals["accepted"]) / count,
        "predicted_improvements_actually_harmful": int(totals["harmful"]),
        "false_safe_decisions": int(totals["false_safe"]),
        "routes_with_useful_predictions": len(route_useful),
        "regret_kwh": {
            "always_reference": totals["reference_regret_kwh"],
            "frozen_causal_predictor": totals["selected_regret_kwh"],
            "always_heuristic_runner_up": totals["runner_up_regret_kwh"],
        },
        "net_selected_delta_energy_kwh": totals["selected_delta_energy_kwh"],
        "counterfactual_hindsight_diagnostic_delta_energy_kwh": totals[
            "hindsight_delta_energy_kwh"
        ],
        "route_aggregate": {
            "selected_delta_energy": _bootstrap_route_mean(dict(route_delta)),
            "regret_improvement_vs_reference": _bootstrap_route_mean(
                dict(route_regret_improvement), seed=20261022
            ),
        },
        "selected_snapshot_rows": selected_rows,
        "one_deviation_430m": {
            "classification": "internal single-intervention diagnostic",
            "rows": one_deviation,
            "accepted_count": sum(
                row["selected_action_id"] is not None for row in one_deviation
            ),
            "net_delta_energy_kwh": sum(
                row["actual_delta_energy_kwh"] for row in one_deviation
            ),
            "new_feasibility_failures": sum(row["false_safe"] for row in one_deviation),
        },
        "test_is_now_opened_study_data": True,
        "test_must_not_be_reused_as_untouched_validation": True,
    }
    atomic_json(EVALUATION, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("predict", "score"))
    args = parser.parse_args()
    result = (
        generate_test_predictions()
        if args.stage == "predict"
        else score_test_predictions()
    )
    print(
        json.dumps(
            {"stage": args.stage, "complete": True, "keys": list(result)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
