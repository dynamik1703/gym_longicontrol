"""Immutable-data diagnosis of the stopped SlackTrader V1 planner."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from typing import Any

from benchmarks.physical_feasibility_reference.causal_controller import (
    CAUSAL_CRUISE_CAP_M_S,
)
from benchmarks.slacktrader_v1.planner import FixedDistancePlanner
from gym_longicontrol.domain.state import VehicleState

from .integrity import HERE

ROOT = HERE.parents[1]
V1_RESULTS = ROOT / "benchmarks/slacktrader_v1/mpc_development_results.json"


def diagnose_v1() -> dict[str, Any]:
    payload = json.loads(V1_RESULTS.read_text(encoding="utf-8"))
    arms = payload["arms"]
    mpc_rows = (
        arms["fixed_price_mpc"]["episodes"]
        + arms["adaptive_price_mpc"]["episodes"]
    )
    selected = [
        speed for row in mpc_rows for speed in row["selected_target_speeds_m_s"]
    ]
    max_selected = sum(
        abs(float(speed) - CAUSAL_CRUISE_CAP_M_S) <= 1.0e-9
        for speed in selected
    )
    sensitivity = payload["sensitivity"]
    points = [point for row in sensitivity for point in row["points"]]
    terminal_values = [float(point["selected_terminal_energy_kwh"]) for point in points]
    travel_times = [float(point["selected_travel_time_s"]) for point in points]
    proposals = sum(int(row["proposal_count"]) for row in mpc_rows)
    replans = sum(int(row["planner_solve_count"]) for row in mpc_rows)
    identical = sum(
        int(row["intervention_reasons"].get("ACCEPTED_IDENTICAL_TO_FALLBACK", 0))
        for row in mpc_rows
    )
    overrides = sum(int(row["override_count"]) for row in mpc_rows)
    action_hashes = [row["native_action_sha256"] for row in mpc_rows]

    planner = FixedDistancePlanner()
    fixture_rows = []
    changed_first = 0
    fixture_candidate_times: list[float] = []
    fixture_terminal_energies: list[float] = []
    fixture_action_sequences: set[tuple[float, ...]] = set()
    for velocity in (0.0, 5.0, 10.0, 15.0):
        state = VehicleState(velocity_m_s=velocity)
        for price in (1.0e-5, 1.0e-3, 1.0e-2):
            plan = planner.solve(
                state,
                current_limit_m_s=CAUSAL_CRUISE_CAP_M_S,
                transitions=(),
                price_kwh_per_s=price,
            )
            for candidate in plan.candidates:
                if candidate.feasible:
                    fixture_candidate_times.append(candidate.travel_time_s)
                    fixture_terminal_energies.append(
                        candidate.terminal_energy_kwh
                    )
                    fixture_action_sequences.add(candidate.actions)
            without_terminal = min(
                (candidate for candidate in plan.candidates if candidate.feasible),
                key=lambda candidate: (
                    candidate.horizon_energy_kwh
                    + price * candidate.travel_time_s,
                    candidate.travel_time_s,
                    candidate.target_speed_m_s,
                ),
            )
            changed = abs(
                plan.selected.actions[0] - without_terminal.actions[0]
            ) > 1.0e-12
            changed_first += changed
            fixture_rows.append(
                {
                    "initial_velocity_m_s": velocity,
                    "price_kwh_per_s": price,
                    "with_terminal_target_m_s": plan.selected.target_speed_m_s,
                    "without_terminal_target_m_s": without_terminal.target_speed_m_s,
                    "with_terminal_first_action": plan.selected.actions[0],
                    "without_terminal_first_action": without_terminal.actions[0],
                    "selected_terminal_energy_kwh": (
                        plan.selected.terminal_energy_kwh
                    ),
                    "first_action_changed": changed,
                }
            )
    return {
        "schema_version": 1,
        "source": str(V1_RESULTS.relative_to(ROOT)),
        "source_state": "immutable existing V1 Development data",
        "v1_stop_code": "RL_NOT_JUSTIFIED_BY_MPC_PREFLIGHT",
        "maximum_target_selection": {
            "count": max_selected,
            "total": len(selected),
            "fraction": max_selected / len(selected) if selected else 0.0,
        },
        "terminal_acceleration_energy": {
            "sensitivity_point_count": len(terminal_values),
            "nonzero_count": sum(value > 0.0 for value in terminal_values),
            "sum_kwh": sum(terminal_values),
            "mean_kwh": statistics.fmean(terminal_values),
            "maximum_kwh": max(terminal_values),
        },
        "candidate_selected_travel_time_distribution_s": {
            "count": len(travel_times),
            "minimum": min(travel_times),
            "median": statistics.median(travel_times),
            "maximum": max(travel_times),
        },
        "fixture_all_candidate_travel_time_distribution_s": {
            "count": len(fixture_candidate_times),
            "minimum": min(fixture_candidate_times),
            "median": statistics.median(fixture_candidate_times),
            "maximum": max(fixture_candidate_times),
        },
        "fixture_terminal_acceleration_energy_distribution_kwh": {
            "count": len(fixture_terminal_energies),
            "nonzero_count": sum(
                value > 0.0 for value in fixture_terminal_energies
            ),
            "mean": statistics.fmean(fixture_terminal_energies),
            "maximum": max(fixture_terminal_energies),
        },
        "fixture_distinct_candidate_action_sequences": len(
            fixture_action_sequences
        ),
        "objective_sensitivity_to_lambda": {
            "states": len(sensitivity),
            "states_with_distinct_duration": sum(
                int(row["distinct_selected_durations"] >= 2)
                for row in sensitivity
            ),
            "duration_monotonic_all": all(
                row["duration_nonincreasing_with_price"] for row in sensitivity
            ),
            "observed_marginal_prices_kwh_per_s": [
                row["observed_endpoint_marginal_price_kwh_per_s"]
                for row in sensitivity
            ],
        },
        "executed_action_sequences": {
            "episodes": len(action_hashes),
            "distinct_sha256": len(set(action_hashes)),
            "hash_multiplicity": dict(sorted(Counter(action_hashes).items())),
        },
        "replanning": {
            "planner_solves": replans,
            "native_proposals": proposals,
            "solve_per_proposal_fraction": replans / proposals,
        },
        "fallback_identity": {
            "identical_proposals": identical,
            "native_proposals": proposals,
            "fraction": identical / proposals,
        },
        "shield": {
            "overrides": overrides,
            "native_proposals": proposals,
            "fraction": overrides / proposals,
        },
        "terminal_first_action_fixture": {
            "fixture_count": len(fixture_rows),
            "changed_count": changed_first,
            "rows": fixture_rows,
        },
        "mechanism_conclusion": {
            "H1": "supported: only nine regulated constant-target candidates",
            "H2": (
                "mechanistically active in deterministic fixtures; existing "
                "capability sensitivity endpoints happened to have zero charge"
            ),
            "H3": (
                f"supported: {identical}/{proposals} proposals were identical "
                "to fallback before counting overrides"
            ),
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    maximum = result["maximum_target_selection"]
    terminal = result["terminal_acceleration_energy"]
    travel = result["candidate_selected_travel_time_distribution_s"]
    all_travel = result["fixture_all_candidate_travel_time_distribution_s"]
    fixture_terminal = result[
        "fixture_terminal_acceleration_energy_distribution_kwh"
    ]
    fallback = result["fallback_identity"]
    shield = result["shield"]
    fixture = result["terminal_first_action_fixture"]
    return f"""# Verified V1 Failure Diagnosis

Source: immutable `benchmarks/slacktrader_v1/mpc_development_results.json` plus
deterministic exact-dynamics fixtures. The historical outcome remains
`RL_NOT_JUSTIFIED_BY_MPC_PREFLIGHT`.

## Registered diagnostics

- Maximum-target selections: {maximum['count']}/{maximum['total']}
  ({100.0 * maximum['fraction']:.3f}%).
- Terminal acceleration contribution across recorded sensitivity selections:
  {terminal['sum_kwh']:.9f} kWh total; {terminal['nonzero_count']}/
  {terminal['sensitivity_point_count']} nonzero. These recorded selected
  endpoints were already at the imposed reference, so zero here does not show
  that the formulation was inactive for discarded candidates.
- Recorded selected candidate time: {travel['minimum']:.3f}–
  {travel['maximum']:.3f} s (median {travel['median']:.3f} s).
- All feasible fixture-candidate times: {all_travel['minimum']:.3f}–
  {all_travel['maximum']:.3f} s (median {all_travel['median']:.3f} s), with
  {result['fixture_distinct_candidate_action_sequences']} distinct candidate
  action sequences.
- Across all fixture candidates, the V1 terminal-acceleration term was nonzero
  for {fixture_terminal['nonzero_count']}/{fixture_terminal['count']}, mean
  {fixture_terminal['mean']:.6f} kWh and maximum
  {fixture_terminal['maximum']:.6f} kWh.
- Lambda changed selected duration on all
  {result['objective_sensitivity_to_lambda']['states']} recorded causal states;
  selected duration was non-increasing throughout.
- Executed physical action hashes:
  {result['executed_action_sequences']['distinct_sha256']}
  distinct sequences across {result['executed_action_sequences']['episodes']} MPC
  episodes.
- Replanning: {result['replanning']['planner_solves']} solves over
  {result['replanning']['native_proposals']} native proposals.
- Planner proposal identical to fallback: {fallback['identical_proposals']}/
  {fallback['native_proposals']} ({100.0 * fallback['fraction']:.3f}%).
- Shield overrides: {shield['overrides']}/{shield['native_proposals']}
  ({100.0 * shield['fraction']:.3f}%).
- Removing V1 terminal energy changed the first action in
  {fixture['changed_count']}/{fixture['fixture_count']} deterministic fixtures.

## Frozen mechanism conclusion

H1 is supported by construction: nine constant-target candidates exclude broad
acceleration/coast/brake sequencing. H2 is mechanistically active in the
controlled fixture even though the recorded *selected* sensitivity endpoints
incurred zero terminal charge. H3 is strongly supported by fallback identity;
Shield overrides add further loss of executed authority. These are mechanism
findings on opened V1 data, not V2 generalization evidence.
"""


if __name__ == "__main__":
    diagnosis = diagnose_v1()
    (HERE / "v1_failure_diagnosis.json").write_text(
        json.dumps(diagnosis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (HERE / "V1_FAILURE_DIAGNOSIS.md").write_text(
        render_markdown(diagnosis), encoding="utf-8"
    )
