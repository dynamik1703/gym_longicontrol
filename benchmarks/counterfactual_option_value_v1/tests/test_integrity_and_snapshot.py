from __future__ import annotations

from dataclasses import asdict

import gymnasium as gym
import numpy as np
import pytest

import gym_longicontrol  # noqa: F401
from benchmarks.causal_mpc_v2.adaptive_price import AdaptivePrice
from benchmarks.physical_feasibility_reference.causal_controller import (
    decode_observation,
)

from ..alternative_selector import select_alternatives
from ..instrumented_controller import InstrumentedContinueController
from ..integrity import verify_native_records
from ..snapshot import capture_snapshot, restore_snapshot, snapshot_bytes


def _native_records(count: int = 3):
    environment = gym.make("StochasticTrack-v1", max_episode_steps=1_800)
    controller = InstrumentedContinueController()
    selector = AdaptivePrice(high_reserve_s=24.0, low_reserve_s=2.0)
    observation, info = environment.reset(seed=12_000_001)
    controller.reset()
    selector.reset()
    base = controller.prepare_strategy_step(observation)
    held = selector.action(base)
    rows = []
    for step in range(count):
        state_before = asdict(controller.state)
        transitions = [
            {"position_m": item.position_m, "limit_m_s": item.limit_m_s}
            for item in controller._transitions()
        ]
        current_limit = float(decode_observation(observation)["current_limit_m_s"])
        action = controller.action(observation, held)
        decision = controller.last_decision
        assert decision is not None
        before_energy = float(info["total_energy_kwh"])
        next_observation, reward, terminated, truncated, next_info = environment.step(
            [action]
        )
        controller.observe_executed_action(
            action, episode_ended=bool(terminated or truncated)
        )
        rows.append(
            {
                "native_step": step,
                "causal_observation": np.asarray(observation).tolist(),
                "causal_state_before": state_before,
                "known_transitions": transitions,
                "current_limit_m_s": current_limit,
                **decision.as_dict(),
                "actual_next_state": asdict(controller.state),
                "actual_next_observation": np.asarray(next_observation).tolist(),
                "native_reward": float(reward),
                "signed_step_energy_kwh": float(next_info["total_energy_kwh"])
                - before_energy,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            }
        )
        observation, info = next_observation, next_info
    environment.close()
    return rows


def test_native_records_replay_and_energy_are_exact():
    result = verify_native_records(_native_records())
    assert result["native_record_count"] == 3
    assert result["deadline_bound_falsifications"] == []
    assert result["exact_native_replay_verified"]


def test_removing_completion_bound_is_detected():
    rows = _native_records()
    del rows[0]["completion_bound_s"]
    with pytest.raises(RuntimeError, match="missing fields"):
        verify_native_records(rows)


def test_corrupting_completion_bound_is_detected():
    rows = _native_records()
    rows[0]["completion_bound_s"] += 0.01
    with pytest.raises(RuntimeError, match="deadline-bound audit falsifications"):
        verify_native_records(rows)


def test_incorrect_acceptance_flag_is_detected():
    rows = _native_records()
    rows[0]["accepted"] = not rows[0]["accepted"]
    with pytest.raises(RuntimeError, match="acceptance flag"):
        verify_native_records(rows)


def test_missing_native_step_is_detected():
    rows = _native_records()
    rows[1]["native_step"] += 1
    with pytest.raises(RuntimeError, match="not contiguous"):
        verify_native_records(rows)


def test_empty_falsification_claim_needs_supporting_records():
    with pytest.raises(RuntimeError, match="empty records"):
        verify_native_records([])


def test_snapshot_serialization_and_restore_are_exact():
    environment = gym.make("StochasticTrack-v1", max_episode_steps=1_800)
    controller = InstrumentedContinueController()
    selector = AdaptivePrice(high_reserve_s=24.0, low_reserve_s=2.0)
    observation, info = environment.reset(seed=12_000_002)
    payload = capture_snapshot(
        environment=environment,
        controller=controller,
        observation=observation,
        info=info,
        held_price=None,
        native_step=0,
        selector_last_action=selector.last_action,
    )
    assert snapshot_bytes(payload) == snapshot_bytes(payload)
    restored = restore_snapshot(payload)
    assert restored[0].unwrapped.state == environment.unwrapped.state
    assert restored[1].snapshot() == controller.snapshot()
    assert np.array_equal(restored[2], observation)
    assert restored[3] == info
    restored[0].close()
    environment.close()


def test_alternative_rule_includes_continue_before_ranked_starts():
    environment = gym.make("StochasticTrack-v1", max_episode_steps=1_800)
    controller = InstrumentedContinueController()
    selector = AdaptivePrice(high_reserve_s=24.0, low_reserve_s=2.0)
    observation, _ = environment.reset(seed=12_000_003)
    held = None
    for step in range(11):
        if step % 10 == 0:
            base = controller.prepare_strategy_step(observation)
            held = selector.action(base)
        action = controller.action(observation, held)
        if step == 10:
            break
        observation, _, terminated, truncated, _ = environment.step([action])
        controller.observe_executed_action(
            action, episode_ended=bool(terminated or truncated)
        )
    assert controller.last_selection is not None
    reference = controller.last_selection.selected.high_level_action
    alternatives = select_alternatives(
        reference_action_id=reference,
        candidates=controller.last_selection.candidates,
    )
    assert len(alternatives.alternative_action_ids) == 2
    assert reference not in alternatives.alternative_action_ids
    if reference != "CONTINUE":
        assert alternatives.alternative_action_ids[0] == "CONTINUE"
    environment.close()
