from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from benchmarks.causal_mpc_v2.controller import PriceMPCController
from benchmarks.causal_mpc_v2.fixed_price import (
    lambda_to_normalized_action,
    normalized_action_to_lambda,
)
from benchmarks.causal_mpc_v2.integrity import (
    HERE,
    require_not_paper,
    verify_frozen_method,
    verify_paper_seal,
)
from benchmarks.causal_mpc_v2.split import build_split_ledger
from benchmarks.physical_feasibility_reference.causal_controller import (
    ObservationLimitedController,
)


class _FailingPlanner:
    def solve(self, *args, **kwargs):
        raise RuntimeError("fixture failure")


def test_controller_api_has_no_privileged_inputs() -> None:
    signature = inspect.signature(PriceMPCController.action)
    assert tuple(signature.parameters) == (
        "self",
        "observation",
        "strategy_action",
    )
    lowered = PriceMPCController.information.lower()
    assert "track" not in lowered
    assert "seed" not in lowered
    assert "oracle" not in lowered


def test_planner_failure_executes_exact_fallback() -> None:
    limit = (60.0 / 3.6) / 37.0
    ordinary = np.array(
        [0.0, 0.5, limit, limit, limit, 1.0, 1.0, 0.0],
        dtype=np.float64,
    )
    reference = ObservationLimitedController()
    expected = reference.action(ordinary)
    controller = PriceMPCController(planner=_FailingPlanner())
    controller.prepare_strategy_step(ordinary)
    observed = controller.action(ordinary, 0.0)
    assert observed == expected
    assert controller.last_decision is not None
    assert controller.last_decision.intervention_reason == "PLANNER_INFEASIBLE_FALLBACK"


def test_time_price_mapping_round_trip_and_units() -> None:
    for action in (-1.0, -0.25, 0.0, 0.75, 1.0):
        price = normalized_action_to_lambda(action)
        assert 1.0e-5 <= price <= 1.0e-2
        assert lambda_to_normalized_action(price) == pytest.approx(action)


def test_splits_are_clean_and_capability_is_unopened() -> None:
    ledger = build_split_ledger()
    assert ledger["route_clean_verified"]
    assert len(ledger["calibration_panel"]["accepted_tasks"]) == 12
    assert len(ledger["capability_panel"]["accepted_tasks"]) == 18
    assert ledger["capability_panel"]["state"] == "FROZEN_UNOPENED"
    assert ledger["capability_panel"]["performance_episode_count"] == 0


def test_paper_is_fail_closed() -> None:
    assert verify_paper_seal()["state"] == "SEALED_UNINSPECTED"
    with pytest.raises(PermissionError, match="SEALED_UNINSPECTED"):
        require_not_paper(4000)


def test_no_training_or_validation_authority() -> None:
    canonical = json.loads((HERE / "canonical.json").read_text(encoding="utf-8"))
    assert canonical["authorization"] == {
        "paper_authorized": False,
        "rl_training_authorized": False,
        "validation_authorized": False,
    }


def test_calibration_is_complete_and_method_is_frozen() -> None:
    calibration = json.loads(
        (HERE / "calibration_results.json").read_text(encoding="utf-8")
    )
    assert calibration["budget"] == {
        "adaptive_configurations": 3,
        "episodes": 96,
        "fixed_configurations": 5,
    }
    assert calibration["selected"] == {
        "adaptive_high_reserve_s": 24.0,
        "adaptive_low_reserve_s": 2.0,
        "fixed_normalized_action": 0.0,
        "fixed_price_kwh_per_s": 0.00031622776601683783,
    }
    canonical = json.loads((HERE / "canonical.json").read_text(encoding="utf-8"))
    assert canonical["execution"] == {
        "calibration_complete": True,
        "capability_complete": False,
        "capability_opened": False,
    }
    verify_frozen_method()
