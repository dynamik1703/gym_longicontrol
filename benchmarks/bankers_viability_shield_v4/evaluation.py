"""Four fixed controllers; method-specific evidence, no shadow grid for CAUSAL60."""

import time
from dataclasses import asdict

from benchmarks.bankers_viability_shield_v1.certificate import BankersShield
from benchmarks.bankers_viability_shield_v2.independent_verification import authority
from benchmarks.bankers_viability_shield_v3.fast_certificate import safety_view
from benchmarks.physical_feasibility_reference.oracle_controller import Transition
from gym_longicontrol.domain.dynamics import advance
from gym_longicontrol.domain.metrics import EpisodeMetrics
from gym_longicontrol.domain.task import TaskSpecification, is_feasible

from .compact_certificate import CompactShield, safe_action_mask
from .controller_adapters import (
    ACTION_GRID,
    BANKERS,
    CONFIG,
    DIAGNOSTIC_STRIDE,
    CausalMemory,
    MeteredVehicle,
    _ExternalExecutionAdapter,
    inherited_test,
    select,
)
from .serialization import require


def deadline_diagnostic(old, diagnostic):
    value = old or (diagnostic[-1] if diagnostic else None)
    if value is None:
        return dict(
            state="NOT_APPLICABLE",
            applicable=False,
            completion_bound_s=None,
            viable=None,
        )
    return dict(
        state="CALCULATED",
        applicable=True,
        completion_bound_s=value["completion_bound_s"],
        viable=value["deadline_viable"],
    )


class CounterexampleFound(RuntimeError):
    def __init__(self, rows):
        self.rows = rows
        super().__init__("BANKERS_V4_COUNTEREXAMPLE_FOUND")


def run(
    environment,
    method,
    *,
    seed=None,
    max_steps=1800,
    initial_state=None,
    budget=None,
    record_sink=None,
    evidence_identity=None,
    attempt="V4_TEST_ONLY",
):
    vehicle = MeteredVehicle()
    from .integrity import source_identity

    evidence_identity = evidence_identity or source_identity()
    shield = CompactShield(evidence_identity)
    frozen = BankersShield(vehicle)
    memory = CausalMemory()
    causal = _ExternalExecutionAdapter(vehicle=vehicle)
    ordinary, info = environment.reset(seed=seed)
    causal.reset()
    if initial_state is not None:
        environment.unwrapped.state = initial_state
        ordinary = environment.unwrapped._observation()
        memory.state = causal.state = initial_state
    rows = []
    for step in range(max_steps):
        if budget:
            budget()
        memory.ingest(ordinary)
        start = time.perf_counter()
        fallback = causal.action(ordinary)
        causal_s = time.perf_counter() - start
        require(causal.state == memory.state, "causal state parity")
        transitions = tuple(
            Transition(p, v) for p, v in sorted(causal._known_transitions.items())
        )
        is_banker = method in BANKERS or method == "V4_SYNTHETIC_MINIMAL"
        # Full-grid inherited comparisons only on preregistered Banker states.
        sampled = is_banker and step % DIAGNOSTIC_STRIDE == 0
        start = time.perf_counter()
        certs = safe_action_mask(shield, ordinary, memory) if is_banker else []
        mask_s = time.perf_counter() - start
        mask = [c["accepted"] for c in certs]
        if is_banker:
            require(mask[0] and any(mask), "EMPTY_MASK_OR_BACKUP_UNAVAILABLE")
        proposal_certificate = None
        if method == "BANKERS_MASKED_CAUSAL60":
            proposal_certificate = (
                certs[ACTION_GRID.index(fallback)]
                if fallback in ACTION_GRID
                else shield.certify(ordinary, memory, fallback)
            )
        start = time.perf_counter()
        old = (
            inherited_test(
                memory.state,
                1.0,
                float(ordinary[2] * 37),
                transitions,
                fallback,
                vehicle,
            )
            if method == "LEGACY_NATIVE_POST_GREEDY"
            else None
        )
        diagnostic = (
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
            if sampled
            else []
        )
        legacy_s = time.perf_counter() - start
        selected, action = select(method, fallback, mask, old)
        start = time.perf_counter()
        final = (
            frozen.certify(ordinary, memory, action).as_dict() if is_banker else None
        )
        final_s = time.perf_counter() - start
        if is_banker:
            require(
                final["accepted"]
                and safety_view(final) == safety_view(certs[selected]),
                "final full-production discrepancy",
            )
        predicted, power, energy = advance(memory.state, action, vehicle, CONFIG)
        next_ordinary, _, terminated, truncated, info = environment.step([action])
        row = dict(
            step=step,
            attempt=attempt,
            evidence_identity=evidence_identity,
            native_time_s=predicted.elapsed_time_s,
            completed=bool(environment.unwrapped.state.position_m >= 1000),
            causal60_memory=dict(
                state=asdict(causal.state),
                known_transitions=sorted(causal._known_transitions.items()),
                last_action=causal._last_action,
            ),
            deadline_diagnostic=deadline_diagnostic(old, diagnostic),
            method=method,
            causal_pre_state=asdict(memory.state),
            observation=ordinary.tolist(),
            memory=memory.snapshot(),
            causal60_proposal=fallback,
            proposal_action=fallback
            if method in ("CAUSAL60", "BANKERS_MASKED_CAUSAL60")
            else 1.0,
            certificates=certs,
            safe_action_mask=mask,
            selected_index=selected,
            selected_certificate=final,
            causal60_proposal_certificate=proposal_certificate,
            executed_action=action,
            predicted_next_state=asdict(predicted),
            actual_next_state=asdict(environment.unwrapped.state),
            actual_power_kw=power,
            signed_energy_increment_kwh=info["step_energy_kwh"],
            actual_speed_limit_m_s=info["speed_limit_m_s"],
            speed_excess_m_s=info["speed_excess_m_s"],
            terminated=bool(terminated),
            truncated=bool(truncated),
            legacy_proposal_check=old,
            diagnostic_sampled=sampled,
            inherited_comparison=diagnostic,
            authority=authority(certs, memory.state) if is_banker else None,
            timing=dict(
                mask_s=mask_s,
                final_s=final_s,
                causal60_s=causal_s,
                legacy_s=legacy_s,
                mask_physics_calls=sum(c["physics_prediction_count"] for c in certs),
                mask_power_calls=0,
                final_physics_calls=final["physics_prediction_count"] if final else 0,
            ),
        )
        rows.append(row)
        if record_sink:
            record_sink(row)
        require(
            asdict(predicted) == asdict(environment.unwrapped.state),
            "actual production parity",
        )
        if is_banker and row["speed_excess_m_s"] != 0:
            raise CounterexampleFound(rows)
        causal.observe_external_action(action)
        memory.execute(action, vehicle)
        ordinary = next_ordinary
        if terminated or truncated:
            break
    metrics = EpisodeMetrics(**info["episode_metrics"])
    return rows, dict(
        steps=len(rows),
        metrics=asdict(metrics),
        feasible=is_feasible(metrics, TaskSpecification(140, 0)),
        actual_progress_m=environment.unwrapped.state.position_m,
        full_model_calls_online=vehicle.predictions,
    )
