"""Four predeclared native controllers and complete per-step evidence."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import asdict

from benchmarks.bankers_viability_shield_v1.safe_action_set import SafeActionSet
from benchmarks.physical_feasibility_reference.oracle_controller import Transition
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible

from .independent_verification import authority
from .inherited_shield_adapter import (
    ACTION_GRID,
    BANKERS,
    CONFIG,
    BankersShield,
    CausalMemory,
    MeteredVehicle,
    _ExternalExecutionAdapter,
    inherited_test,
)
from .masked_causal_controller import closest_certified
from .serialization import canonical_bytes, require


class CounterexampleFound(RuntimeError):
    def __init__(self, rows):
        self.rows = rows
        super().__init__("BANKERS_V2_COUNTEREXAMPLE_FOUND")


def run(
    environment, method, *, seed=None, max_steps=1800, initial_state=None, budget=None
):
    vehicle = MeteredVehicle()
    shield = BankersShield(vehicle)
    memory = CausalMemory()
    causal = _ExternalExecutionAdapter(vehicle=vehicle)
    ordinary, info = environment.reset(seed=seed)
    causal.reset()
    if initial_state is not None:
        # Synthetic-only physical fixture; never used in Development.
        environment.unwrapped.state = initial_state
        ordinary = environment.unwrapped._observation()
        memory.state = causal.state = initial_state
    rows = []
    for step in range(max_steps):
        if budget is not None:
            budget()
        memory.ingest(ordinary)
        fallback = causal.action(ordinary)
        require(causal.state == memory.state, "causal state parity")
        causal_snapshot = {
            "state": asdict(causal.state),
            "known_transitions": sorted(causal._known_transitions.items()),
            "last_action": causal._last_action,
        }
        transitions = tuple(
            Transition(p, v) for p, v in sorted(causal._known_transitions.items())
        )
        start = time.perf_counter()
        menu = SafeActionSet(
            tuple(shield.certify(ordinary, memory, a) for a in ACTION_GRID)
        )
        mask_s = time.perf_counter() - start
        start = time.perf_counter()
        old = (
            [
                inherited_test(
                    memory.state,
                    a,
                    float(ordinary[2] * 37),
                    transitions,
                    fallback,
                    vehicle,
                )
                for a in ACTION_GRID
            ]
            if method != "CAUSAL60"
            else []
        )
        old_s = time.perf_counter() - start
        proposal = (
            fallback if method in ("CAUSAL60", "BANKERS_MASKED_CAUSAL60") else 1.0
        )
        selected = None
        if method == "CAUSAL60":
            action = fallback
        elif method == "LEGACY_NATIVE_POST_GREEDY":
            action = 1.0 if old[-1]["accepted"] else fallback
        elif method in BANKERS:
            require(menu.mask[0] and any(menu.mask), "EMPTY_MASK_OR_BACKUP_UNAVAILABLE")
            selected = (
                max(i for i, valid in enumerate(menu.mask) if valid)
                if method == "BANKERS_GREEDY"
                else closest_certified(menu.mask, fallback)
            )
            action = ACTION_GRID[selected]
        elif method == "V2_SYNTHETIC_MINIMAL":
            require(menu.mask[0], "EMPTY_MASK_OR_BACKUP_UNAVAILABLE")
            selected, action = 0, -1.0
        else:
            raise ValueError(method)
        start = time.perf_counter()
        final = shield.certify(ordinary, memory, action)
        final_s = time.perf_counter() - start
        if method in BANKERS or method == "V2_SYNTHETIC_MINIMAL":
            require(
                final.accepted and final == menu.certificates[selected],
                "final certificate discrepancy",
            )
        next_ordinary, _, terminated, truncated, info = environment.step([action])
        predicted, _, _ = advance(memory.state, action, vehicle, CONFIG)
        row = {
            "step": step,
            "method": method,
            "causal_pre_state": asdict(memory.state),
            "observation": ordinary.tolist(),
            "memory": memory.snapshot(),
            "causal60_memory": causal_snapshot,
            "safe_action_mask": list(menu.mask),
            "certificates": [c.as_dict() for c in menu.certificates],
            "proposal_action": proposal,
            "causal60_proposal": fallback,
            "selected_index": selected,
            "selected_certificate": final.as_dict(),
            "executed_action": action,
            "predicted_next_state": asdict(predicted),
            "actual_next_state": asdict(environment.unwrapped.state),
            "actual_speed_limit_m_s": info["speed_limit_m_s"],
            "speed_excess_m_s": info["speed_excess_m_s"],
            "signed_energy_increment_kwh": info["step_energy_kwh"],
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "inherited_comparison": old,
            "deadline_diagnostic": (
                {
                    "state": "CALCULATED",
                    "applicable": True,
                    "completion_bound_s": old[-1]["completion_bound_s"],
                    "viable": old[-1]["deadline_viable"],
                }
                if old
                else {
                    "state": "NOT_APPLICABLE",
                    "applicable": False,
                    "completion_bound_s": None,
                    "viable": None,
                }
            ),
            "authority": authority(
                [c.as_dict() for c in menu.certificates], memory.state
            ),
            "timing": {
                "mask_s": mask_s,
                "legacy_s": old_s,
                "final_s": final_s,
                "mask_physics_calls": menu.physics_prediction_count,
                "final_physics_calls": final.physics_prediction_count,
            },
        }
        rows.append(row)
        require(
            canonical_bytes(row["actual_next_state"])
            == canonical_bytes(row["predicted_next_state"]),
            "actual physics parity",
        )
        if (method in BANKERS or method == "V2_SYNTHETIC_MINIMAL") and row[
            "speed_excess_m_s"
        ] != 0:
            raise CounterexampleFound(rows)
        causal.observe_external_action(action)
        memory.execute(action, vehicle)
        ordinary = next_ordinary
        if terminated or truncated:
            break
    metrics = EpisodeMetrics(**info["episode_metrics"])
    return rows, {
        "steps": len(rows),
        "metrics": asdict(metrics),
        "feasible": is_feasible(metrics, TaskSpecification(140, 0)),
        "actual_progress_m": environment.unwrapped.state.position_m,
        "physics_calls_online": vehicle.predictions,
        "mask_s_max": max(r["timing"]["mask_s"] for r in rows),
        "mask_s_sum": sum(r["timing"]["mask_s"] for r in rows),
        "mask_physics_calls": sum(r["timing"]["mask_physics_calls"] for r in rows),
        "actions": {
            str(k): v for k, v in Counter(r["executed_action"] for r in rows).items()
        },
    }
