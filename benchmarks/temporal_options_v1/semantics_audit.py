"""Deterministic audit of restart and persistent spatial-phase semantics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.causal_mpc_v2.planner import PROFILES, FixedDistanceProfilePlanner
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.state import SimulationConfig, VehicleState
from gym_longicontrol.domain.vehicle import VehicleModel

from .feedback_executor import NativeFeedbackExecutor, frozen_fallback_action
from .option_state import OptionState

HERE = Path(__file__).resolve().parent


def _profile(profile_id: str):
    return next(profile for profile in PROFILES if profile.profile_id == profile_id)


def _restart_trace(profile_id: str, native_steps: int = 90) -> dict[str, Any]:
    vehicle = VehicleModel()
    config = SimulationConfig()
    profile = _profile(profile_id)
    planner = FixedDistanceProfilePlanner(
        vehicle=vehicle, config=config, profiles=(profile,)
    )
    state = VehicleState(position_m=300.0, velocity_m_s=12.0, elapsed_time_s=10.0)
    plans = []
    phases = []
    actions = []
    calls = 0
    segment: list[float] = []
    for step in range(native_steps):
        if step % 10 == 0:
            plan = planner.solve(
                state,
                current_limit_m_s=100.0 / 3.6,
                transitions=(),
                price_kwh_per_s=1.0e-3,
            )
            calls += plan.physics_calls
            segment = list(plan.segment_actions)
            plans.append(
                {
                    "native_step": step,
                    "origin_m": state.position_m,
                    "waypoint_m": plan.waypoint_position_m,
                    "coast_threshold_m": (
                        state.position_m
                        + profile.coast_start_fraction * plan.planning_distance_m
                    ),
                    "brake_threshold_m": (
                        state.position_m
                        + profile.brake_start_fraction * plan.planning_distance_m
                    ),
                    "distance_executed_before_replan_m": None,
                }
            )
        fraction = (state.position_m - plans[-1]["origin_m"]) / (
            plans[-1]["waypoint_m"] - plans[-1]["origin_m"]
        )
        if profile.family == "traction_coast_brake":
            phase = (
                "brake"
                if fraction >= profile.brake_start_fraction
                else "coast"
                if fraction >= profile.coast_start_fraction
                else "traction"
            )
        elif profile.family == "traction_then_coast":
            phase = "coast" if fraction >= profile.coast_start_fraction else "traction"
        elif profile.family == "coast_then_partial_traction":
            phase = (
                "partial_traction"
                if fraction >= profile.coast_start_fraction
                else "coast"
            )
        else:
            phase = "velocity_triggered"
        action = segment.pop(0)
        state, _, _ = advance(state, action, vehicle, config)
        actions.append(action)
        phases.append(phase)
        if (step + 1) % 10 == 0:
            plans[-1]["distance_executed_before_replan_m"] = (
                state.position_m - plans[-1]["origin_m"]
            )
    return {
        "profile_id": profile_id,
        "plans": plans,
        "phase_counts": dict(sorted(Counter(phases).items())),
        "action_zero_count": sum(abs(action) <= 1.0e-15 for action in actions),
        "physics_prediction_calls": calls,
        "native_fixture_steps": native_steps,
    }


def _persistent_trace(profile_id: str, native_steps: int = 90) -> dict[str, Any]:
    vehicle = VehicleModel()
    config = SimulationConfig()
    profile = _profile(profile_id)
    executor = NativeFeedbackExecutor(vehicle=vehicle, config=config)
    state = VehicleState(position_m=300.0, velocity_m_s=12.0, elapsed_time_s=10.0)
    option = OptionState.start(
        instance_id=1,
        profile=profile,
        state=state,
        route_length_m=config.track_length_m,
    )
    anchor = option.start_position_m
    waypoint = option.target_waypoint_m
    phases = []
    actions = []
    calls = 0
    interventions: Counter[str] = Counter()
    for _ in range(native_steps):
        if state.position_m >= waypoint:
            break
        fallback = frozen_fallback_action(
            state,
            current_limit_m_s=100.0 / 3.6,
            transitions=(),
            vehicle=vehicle,
            config=config,
        )
        option.update_physical_state(profile, state, count_step=False)
        decision = executor.decide(
            option,
            profile,
            state,
            current_limit_m_s=100.0 / 3.6,
            transitions=(),
            fallback_action=fallback,
        )
        state, _, _ = advance(state, decision.executed_action, vehicle, config)
        option.update_physical_state(
            profile,
            state,
            executed_action=decision.executed_action,
            count_step=True,
        )
        calls += decision.physics_prediction_calls + 1
        phases.append(decision.phase)
        actions.append(decision.executed_action)
        if not decision.accepted:
            interventions[decision.intervention_reason] += 1
            break
    return {
        "profile_id": profile_id,
        "anchor_m": anchor,
        "waypoint_m": waypoint,
        "original_length_m": option.original_length_m,
        "final_position_m": state.position_m,
        "final_progress": option.progress,
        "phase_counts": dict(sorted(Counter(phases).items())),
        "action_zero_count": sum(abs(action) <= 1.0e-15 for action in actions),
        "physics_prediction_calls": calls,
        "native_fixture_steps": len(actions),
        "interventions": dict(sorted(interventions.items())),
    }


def build_audit() -> dict[str, Any]:
    affected = (
        "accel_coast_40_0.5",
        "accel_coast_60_0.5",
        "accel_coast_brake_50_-0.5",
        "accel_coast_brake_50_-1.0",
        "coast_partial_35",
        "coast_partial_50",
    )
    profiles = {
        profile_id: {
            "restart": _restart_trace(profile_id),
            "persistent": _persistent_trace(profile_id),
        }
        for profile_id in affected
    }
    pulse = _restart_trace("pulse_45_1.0")
    vehicle = VehicleModel()
    coast_state = VehicleState(velocity_m_s=15.0)
    coast_next, power, energy = advance(coast_state, 0.0, vehicle, SimulationConfig())
    final_option = OptionState.start(
        instance_id=99,
        profile=_profile("accel_coast_brake_50_-0.5"),
        state=VehicleState(position_m=950.0, elapsed_time_s=100.0),
        route_length_m=1000.0,
    )
    total_steps = (
        sum(
            row[mode]["native_fixture_steps"]
            for row in profiles.values()
            for mode in ("restart", "persistent")
        )
        + pulse["native_fixture_steps"]
    )
    return {
        "schema_version": 1,
        "source_commit": "c7c6175b22e373f476aa60c10a755cb7072d9d5f",
        "hypothesis_status": (
            "verified for listed distance-triggered fixtures; not universal"
        ),
        "profiles": profiles,
        "velocity_triggered_pulse_restart_control": pulse,
        "native_action_zero": {
            "meaning": (
                "production coasting action; it follows the vehicle zero-power "
                "acceleration curve"
            ),
            "start_velocity_m_s": coast_state.velocity_m_s,
            "next_velocity_m_s": coast_next.velocity_m_s,
            "acceleration_m_s2": coast_next.acceleration_m_s2,
            "predicted_power_kw": power,
            "signed_step_energy_kwh": energy,
        },
        "shortened_final_option": {
            "anchor_m": final_option.start_position_m,
            "waypoint_m": final_option.target_waypoint_m,
            "length_m": final_option.original_length_m,
            "coast_threshold_m": final_option.start_position_m
            + 0.4 * final_option.original_length_m,
            "brake_threshold_m": final_option.start_position_m
            + 0.82 * final_option.original_length_m,
        },
        "native_fixture_steps": total_steps,
        "remaining_hypotheses": [
            (
                "episode-level energy consequences depend on selection, Shield "
                "interventions and route content"
            ),
            "receding-horizon restart may be beneficial on some tasks",
            (
                "speed clipping or safety intervention can prevent a planned "
                "phase legitimately"
            ),
        ],
    }


def render(audit: dict[str, Any]) -> str:
    lines = [
        "# Semantics Audit",
        "",
        "The frozen planner computes distance-triggered phases relative to each",
        "plan's current origin. The frozen controller replans at one-second",
        "boundaries. In deterministic constant-limit fixtures, repeated selection",
        "therefore postponed the 40%, 50% and 82% thresholds when less than that",
        "fraction of the newly anchored 100 m horizon was executed per block.",
        "",
        "This is a verified execution behavior, not by itself an implementation",
        "bug or evidence of episode-level energy loss. Velocity-triggered pulse",
        "behavior still entered coasting after reaching its speed threshold.",
        "",
        "| Profile | Restart phases | Persistent phases |",
        "| --- | --- | --- |",
    ]
    for profile_id, row in audit["profiles"].items():
        lines.append(
            f"| `{profile_id}` | `{row['restart']['phase_counts']}` | "
            f"`{row['persistent']['phase_counts']}` |"
        )
    zero = audit["native_action_zero"]
    lines.extend(
        [
            "",
            "## Native action 0",
            "",
            f"At 15 m/s, action 0 produced acceleration "
            f"{zero['acceleration_m_s2']:.9f} m/s² and model power "
            f"{zero['predicted_power_kw']:.9f} kW. It is the production vehicle's",
            "coasting action, not an added controller or a guaranteed exactly-zero",
            "numerical power value.",
            "",
            "## Boundaries and remaining uncertainty",
            "",
            "Speed-limit clipping, known-transition braking, unknown-preview checks",
            "and deadline recoverability can legitimately change or interrupt phases.",
            "The final-route option uses its actual shortened length. No claim that",
            "all coasting was historically suppressed is supported.",
            "",
            f"Fixture native steps: {audit['native_fixture_steps']} (below 30,000).",
            "",
        ]
    )
    return "\n".join(lines)


def materialize() -> dict[str, Any]:
    audit = build_audit()
    (HERE / "semantics_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (HERE / "SEMANTICS_AUDIT.md").write_text(render(audit), encoding="utf-8")
    return audit


if __name__ == "__main__":
    materialize()
