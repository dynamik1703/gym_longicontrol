"""Causal observation/memory/mask interface; final native re-verification."""

from __future__ import annotations

from .backup_policy import CausalMemory, SafetyIntegrityError
from .certificate import BankersShield
from .safe_action_set import ACTION_GRID, safe_action_set


class PreShield:
    def __init__(self, vehicle=None):
        self.shield = BankersShield(vehicle)
        self.memory = CausalMemory()
        self.pending = None
        self.verified_action = None

    def prepare(self, observation):
        if self.pending is not None:
            raise SafetyIntegrityError("unconsumed mask")
        self.memory.ingest(observation)
        self.pending = safe_action_set(self.shield, observation, self.memory)
        return self.pending

    def select(self, index: int, observation):
        if self.pending is None or type(index) is not int or not 0 <= index < 21:
            raise SafetyIntegrityError("invalid action index or missing mask")
        if not self.pending.mask[index]:
            raise SafetyIntegrityError("actor selected invalid action")
        final = self.shield.certify(observation, self.memory, ACTION_GRID[index])
        if not final.accepted or final != self.pending.certificates[index]:
            raise SafetyIntegrityError("MASK_FINAL_VERIFICATION_DISCREPANCY")
        self.verified_action = ACTION_GRID[index]
        return self.verified_action, final

    def observe_executed_action(self, action):
        if self.verified_action is None or action != self.verified_action:
            raise SafetyIntegrityError("executed action differs from verified action")
        self.memory.execute(action, self.shield.vehicle)
        self.pending = None
        self.verified_action = None

    def actor_input(self, observation):
        if self.pending is None:
            raise SafetyIntegrityError("prepare before reading actor input")
        state = self.memory.state
        return {
            "ordinary_observation": list(observation),
            "physical_memory": self.memory.snapshot(),
            "remaining_distance_m": max(0.0, 1000 - state.position_m),
            "remaining_deadline_s": 140 - state.elapsed_time_s,
            "safe_action_mask": list(self.pending.mask),
        }
