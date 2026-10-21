"""Frozen causal linear and small-neural counterfactual predictors."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from benchmarks.temporal_options_v1.observation_adapter import ACTION_IDS

from .execution import atomic_json, sha256_file
from .integrity import BRANCH_LEDGER, HERE, verify_frozen_method

ARTIFACT = HERE / "predictor_artifacts.json"
MODEL_SELECTION = HERE / "model_selection.json"
MANIFEST = HERE / "dataset_manifest.json"
NEURAL_SEEDS = (11, 29, 47)
ENERGY_MARGIN_WH = (0.2, 0.5, 1.0)
RISK_THRESHOLDS = (0.05, 0.10, 0.20)
RIDGE_ALPHA = 1.0e-3
LOGISTIC_L2 = 1.0e-3
LOGISTIC_LR = 0.05
LOGISTIC_STEPS = 1_000
NEURAL_EPOCHS = 300
NEURAL_LR = 1.0e-3
NEURAL_WEIGHT_DECAY = 1.0e-4


def _one_hot(action_id: str) -> np.ndarray:
    result = np.zeros(len(ACTION_IDS), dtype=np.float64)
    result[ACTION_IDS.index(action_id)] = 1.0
    return result


def feature_vector(snapshot: dict[str, Any], candidate_action_id: str) -> np.ndarray:
    """Only the frozen 70D causal state and two causal action identities."""

    observation = np.asarray(snapshot["causal_observation_70d"], dtype=np.float64)
    if observation.shape != (70,) or not np.isfinite(observation).all():
        raise ValueError("predictor requires a finite frozen 70D observation")
    return np.concatenate(
        (
            observation,
            _one_hot(candidate_action_id),
            _one_hot(snapshot["reference_action_id"]),
        )
    )


def _load_pairs(
    panel_id: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    ledger = json.loads(BRANCH_LEDGER.read_text(encoding="utf-8"))
    if ledger["state"] != "COMPLETE":
        raise RuntimeError("counterfactual dataset is incomplete")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    snapshots = {
        row["snapshot_id"]: row
        for row in manifest["snapshots"]
        if row["panel_id"] == panel_id
    }
    pairs = []
    for item in ledger["pair_records"]:
        if item["snapshot_id"] not in snapshots:
            continue
        path = HERE / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError("immutable counterfactual pair changed")
        pairs.append(json.loads(path.read_text(encoding="utf-8")))
    return pairs, snapshots


def _rows(panel_id: str) -> list[dict[str, Any]]:
    pairs, snapshots = _load_pairs(panel_id)
    rows = []
    for pair in pairs:
        snapshot = snapshots[pair["snapshot_id"]]
        for alternative in pair["alternatives"]:
            rows.append(
                {
                    "snapshot_id": pair["snapshot_id"],
                    "numeric_seed": pair["numeric_seed"],
                    "route_sha256": pair["route_sha256"],
                    "action_id": alternative["action_id"],
                    "x": feature_vector(snapshot, alternative["action_id"]),
                    "energy_valid": bool(
                        pair["reference_feasible"] and alternative["feasible"]
                    ),
                    "delta_energy_kwh": float(alternative["delta_energy_kwh"]),
                    "infeasible": float(not alternative["feasible"]),
                    "introduced_infeasibility": bool(
                        alternative["introduced_infeasibility"]
                    ),
                }
            )
    return rows


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _linear_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    design = np.c_[x, np.ones(len(x))]
    regularizer = np.eye(design.shape[1]) * alpha
    regularizer[-1, -1] = 0.0
    return np.linalg.solve(design.T @ design + regularizer, design.T @ y)


def _logistic_fit(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.c_[x, np.ones(len(x))]
    weights = np.zeros(design.shape[1], dtype=np.float64)
    for _ in range(LOGISTIC_STEPS):
        probabilities = _sigmoid(design @ weights)
        gradient = design.T @ (probabilities - y) / len(y)
        gradient[:-1] += LOGISTIC_L2 * weights[:-1]
        weights -= LOGISTIC_LR * gradient
    return weights


class SmallMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value).squeeze(-1)


def _state_to_json(model: nn.Module) -> dict[str, Any]:
    return {
        key: value.detach().cpu().tolist() for key, value in model.state_dict().items()
    }


def _state_from_json(model: nn.Module, value: dict[str, Any]) -> None:
    template = model.state_dict()
    state = {
        key: torch.as_tensor(array, dtype=template[key].dtype)
        for key, array in value.items()
    }
    model.load_state_dict(state)


def _train_neural(
    x: np.ndarray,
    energy_mask: np.ndarray,
    energy_target: np.ndarray,
    infeasible_target: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    features = torch.as_tensor(x, dtype=torch.float32)
    mask = torch.as_tensor(energy_mask, dtype=torch.bool)
    energy = torch.as_tensor(energy_target, dtype=torch.float32)
    risk = torch.as_tensor(infeasible_target, dtype=torch.float32)
    energy_model = SmallMLP(x.shape[1])
    risk_model = SmallMLP(x.shape[1])
    energy_optimizer = torch.optim.Adam(
        energy_model.parameters(), lr=NEURAL_LR, weight_decay=NEURAL_WEIGHT_DECAY
    )
    risk_optimizer = torch.optim.Adam(
        risk_model.parameters(), lr=NEURAL_LR, weight_decay=NEURAL_WEIGHT_DECAY
    )
    for _ in range(NEURAL_EPOCHS):
        energy_optimizer.zero_grad()
        energy_loss = torch.mean((energy_model(features)[mask] - energy[mask]) ** 2)
        energy_loss.backward()
        energy_optimizer.step()
        risk_optimizer.zero_grad()
        risk_loss = nn.functional.binary_cross_entropy_with_logits(
            risk_model(features), risk
        )
        risk_loss.backward()
        risk_optimizer.step()
    return {
        "seed": seed,
        "energy_state": _state_to_json(energy_model),
        "risk_state": _state_to_json(risk_model),
    }


def train_models() -> dict[str, Any]:
    verify_frozen_method(require_labels_authorized=True)
    if ARTIFACT.exists() or MODEL_SELECTION.exists():
        raise PermissionError("predictor artifacts already exist")
    fit_rows = _rows("CF_OPTION_FIT_V1")
    if not fit_rows:
        raise RuntimeError("Fit has no counterfactual rows")
    x_raw = np.stack([row["x"] for row in fit_rows])
    mean = x_raw.mean(axis=0)
    scale = x_raw.std(axis=0)
    scale[scale < 1.0e-8] = 1.0
    x = (x_raw - mean) / scale
    mask = np.asarray([row["energy_valid"] for row in fit_rows], dtype=bool)
    energy_kwh = np.asarray(
        [row["delta_energy_kwh"] for row in fit_rows], dtype=np.float64
    )
    if not mask.any():
        raise RuntimeError("Fit has no jointly feasible energy labels")
    energy_mean = float(energy_kwh[mask].mean())
    energy_scale = float(energy_kwh[mask].std())
    if energy_scale < 1.0e-9:
        energy_scale = 1.0e-9
    energy_standard = (energy_kwh - energy_mean) / energy_scale
    infeasible = np.asarray([row["infeasible"] for row in fit_rows], dtype=np.float64)
    linear_energy = _linear_fit(x[mask], energy_standard[mask], RIDGE_ALPHA)
    linear_risk = _logistic_fit(x, infeasible)
    neural = [
        _train_neural(x, mask, energy_standard, infeasible, seed)
        for seed in NEURAL_SEEDS
    ]
    artifact = {
        "schema_version": 1,
        "input_contract": {
            "causal_observation_dimension": 70,
            "candidate_action_one_hot_dimension": 18,
            "reference_action_one_hot_dimension": 18,
            "total_dimension": int(x.shape[1]),
            "forbidden_inputs": [
                "route_id",
                "numeric_seed",
                "true_future_limits",
                "future_outcomes",
                "Oracle_information",
            ],
        },
        "normalization": {"mean": mean.tolist(), "scale": scale.tolist()},
        "energy_target_normalization": {
            "mean_kwh": energy_mean,
            "scale_kwh": energy_scale,
        },
        "linear": {
            "ridge_alpha": RIDGE_ALPHA,
            "energy_weights": linear_energy.tolist(),
            "logistic_l2": LOGISTIC_L2,
            "logistic_lr": LOGISTIC_LR,
            "logistic_steps": LOGISTIC_STEPS,
            "risk_weights": linear_risk.tolist(),
        },
        "neural": {
            "architecture": [64, 64],
            "activation": "tanh",
            "separate_energy_and_risk_networks": True,
            "epochs": NEURAL_EPOCHS,
            "optimizer": "Adam",
            "learning_rate": NEURAL_LR,
            "weight_decay": NEURAL_WEIGHT_DECAY,
            "ensemble_seeds": list(NEURAL_SEEDS),
            "members": neural,
        },
        "fit_row_count": len(fit_rows),
        "fit_route_count": len({row["route_sha256"] for row in fit_rows}),
        "fit_gradient_updates": len(NEURAL_SEEDS) * 2 * NEURAL_EPOCHS,
        "rl_gradient_updates": 0,
    }
    atomic_json(ARTIFACT, artifact)
    return artifact


@dataclass(frozen=True)
class FrozenPredictor:
    artifact: dict[str, Any]
    family: str

    def predict(self, x_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.asarray(self.artifact["normalization"]["mean"], dtype=np.float64)
        scale = np.asarray(self.artifact["normalization"]["scale"], dtype=np.float64)
        x = (np.asarray(x_raw, dtype=np.float64) - mean) / scale
        target = self.artifact["energy_target_normalization"]
        if self.family == "linear":
            design = np.c_[x, np.ones(len(x))]
            energy_standard = design @ np.asarray(
                self.artifact["linear"]["energy_weights"], dtype=np.float64
            )
            risk = _sigmoid(
                design
                @ np.asarray(self.artifact["linear"]["risk_weights"], dtype=np.float64)
            )
        elif self.family == "neural":
            energy_members = []
            risk_members = []
            features = torch.as_tensor(x, dtype=torch.float32)
            for member in self.artifact["neural"]["members"]:
                energy_model = SmallMLP(x.shape[1])
                risk_model = SmallMLP(x.shape[1])
                _state_from_json(energy_model, member["energy_state"])
                _state_from_json(risk_model, member["risk_state"])
                energy_model.eval()
                risk_model.eval()
                with torch.no_grad():
                    energy_members.append(energy_model(features).numpy())
                    risk_members.append(torch.sigmoid(risk_model(features)).numpy())
            energy_standard = np.mean(energy_members, axis=0)
            risk = np.mean(risk_members, axis=0)
        else:
            raise ValueError("unknown predictor family")
        energy = energy_standard * float(target["scale_kwh"]) + float(
            target["mean_kwh"]
        )
        return np.asarray(energy), np.asarray(risk)


def _calibration_predictions(
    family: str, artifact: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = _rows("CF_OPTION_CALIBRATION_V1")
    x = np.stack([row["x"] for row in rows])
    energy, risk = FrozenPredictor(artifact, family).predict(x)
    predictions = [
        {
            **{key: row[key] for key in ("snapshot_id", "route_sha256", "action_id")},
            "predicted_delta_energy_kwh": float(predicted_energy),
            "predicted_infeasibility_risk": float(predicted_risk),
            "actual_delta_energy_kwh": row["delta_energy_kwh"],
            "actual_energy_valid": row["energy_valid"],
            "actual_infeasible": bool(row["infeasible"]),
            "actual_introduced_infeasibility": row["introduced_infeasibility"],
        }
        for row, predicted_energy, predicted_risk in zip(
            rows, energy, risk, strict=True
        )
    ]
    return rows, predictions


def _score_rule(
    predictions: list[dict[str, Any]], margin_wh: float, risk_threshold: float
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        grouped.setdefault(row["snapshot_id"], []).append(row)
    selected = []
    regret_reference = regret_selected = 0.0
    net_delta = 0.0
    false_safe = harmful = 0
    accepted = 0
    for snapshot_id, candidates in grouped.items():
        feasible_values = [
            row["actual_delta_energy_kwh"]
            for row in candidates
            if row["actual_energy_valid"]
        ]
        best = min([0.0, *feasible_values])
        eligible = [
            row
            for row in candidates
            if row["predicted_delta_energy_kwh"] <= -margin_wh / 1_000.0
            and row["predicted_infeasibility_risk"] <= risk_threshold
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
        actual = 0.0
        if choice is not None:
            accepted += 1
            false_safe += int(choice["actual_infeasible"])
            if choice["actual_energy_valid"]:
                actual = float(choice["actual_delta_energy_kwh"])
                net_delta += actual
                harmful += int(actual > 0.0)
            selected.append(
                {"snapshot_id": snapshot_id, "action_id": choice["action_id"]}
            )
        regret_reference += -best
        regret_selected += actual - best
    return {
        "margin_wh": margin_wh,
        "risk_threshold": risk_threshold,
        "snapshot_count": len(grouped),
        "accepted_count": accepted,
        "abstention_count": len(grouped) - accepted,
        "false_safe_count": false_safe,
        "harmful_count": harmful,
        "net_delta_energy_kwh": net_delta,
        "reference_regret_kwh": regret_reference,
        "selected_regret_kwh": regret_selected,
        "regret_improvement_kwh": regret_reference - regret_selected,
        "selected": selected,
    }


def calibrate_and_select() -> dict[str, Any]:
    verify_frozen_method(require_labels_authorized=True)
    if MODEL_SELECTION.exists():
        raise PermissionError("model selection is already frozen")
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    candidates = []
    for family in ("linear", "neural"):
        _, predictions = _calibration_predictions(family, artifact)
        for margin in ENERGY_MARGIN_WH:
            for risk in RISK_THRESHOLDS:
                candidates.append(
                    {"family": family, **_score_rule(predictions, margin, risk)}
                )
    admissible = [
        row
        for row in candidates
        if row["false_safe_count"] == 0 and row["net_delta_energy_kwh"] < 0.0
    ]
    selected = max(
        admissible,
        key=lambda row: (
            row["regret_improvement_kwh"],
            -row["harmful_count"],
            row["abstention_count"],
            row["family"] == "linear",
        ),
        default={
            "family": "linear",
            "margin_wh": 1.0e9,
            "risk_threshold": 0.0,
            "snapshot_count": len(
                {row["snapshot_id"] for row in _rows("CF_OPTION_CALIBRATION_V1")}
            ),
            "accepted_count": 0,
            "abstention_count": len(
                {row["snapshot_id"] for row in _rows("CF_OPTION_CALIBRATION_V1")}
            ),
            "false_safe_count": 0,
            "harmful_count": 0,
            "net_delta_energy_kwh": 0.0,
            "reference_regret_kwh": 0.0,
            "selected_regret_kwh": 0.0,
            "regret_improvement_kwh": 0.0,
            "selected": [],
            "fallback_reason": "no zero-false-safe positive-benefit calibration rule",
        },
    )
    result = {
        "schema_version": 1,
        "artifact_sha256": sha256_file(ARTIFACT),
        "predefined_family_count": 2,
        "neural_initialization_seeds": list(NEURAL_SEEDS),
        "threshold_grid": {
            "energy_margin_wh": list(ENERGY_MARGIN_WH),
            "risk": list(RISK_THRESHOLDS),
        },
        "selection_rule": (
            "zero false-safe and negative net delta required; maximize calibration "
            "regret improvement with frozen tie-breaks"
        ),
        "selected": selected,
        "all_candidates": candidates,
        "test_labels_loaded": False,
    }
    atomic_json(MODEL_SELECTION, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("train", "calibrate"))
    args = parser.parse_args()
    result = train_models() if args.stage == "train" else calibrate_and_select()
    print(
        json.dumps(
            {"stage": args.stage, "complete": True, "keys": list(result)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
