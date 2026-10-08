"""Non-intervening diagnostics for sampled supervised tuples and the policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .config import NATIVE_DT_S
from .goal_adapter import canonical_command
from .replay import SampledSupervision


def _summary(values) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return {"minimum": None, "maximum": None, "mean": None, "std": None}
    return {
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def _conditional_action_ambiguity(
    conditions: np.ndarray, actions: np.ndarray
) -> dict[str, Any]:
    _, inverse, counts = np.unique(
        np.asarray(conditions), axis=0, return_inverse=True, return_counts=True
    )
    repeated_groups = np.flatnonzero(counts > 1)
    standard_deviations = []
    action_ranges = []
    repeated_rows = 0
    for group in repeated_groups:
        values = np.asarray(actions)[inverse == group]
        repeated_rows += len(values)
        standard_deviations.append(float(np.mean(np.std(values, axis=0))))
        action_ranges.append(float(np.max(np.ptp(values, axis=0))))
    return {
        "repeated_condition_group_count": int(len(repeated_groups)),
        "repeated_sample_fraction": float(repeated_rows / len(actions)),
        "within_condition_action_std": _summary(standard_deviations),
        "within_condition_action_range": _summary(action_ranges),
    }


def sampled_tuple_diagnostics(sampled: SampledSupervision) -> dict[str, Any]:
    goals = np.asarray(sampled.batch.goals, dtype=np.float64)
    lags = np.asarray(sampled.batch.lags, dtype=np.int64)
    canonical = np.all(goals == canonical_command(), axis=1)
    timely = goals[:, 1] == 1.0
    safe = goals[:, 2] == 1.0
    unique = np.unique(goals, axis=0)
    source_progress = sampled.source_outcomes[:, 0] / 1000.0
    pair_count = len(goals) * (len(goals) - 1) // 2
    collisions = 0
    if pair_count:
        equality = np.all(goals[:, None] == goals[None, :], axis=-1)
        collisions = int(np.sum(np.triu(equality, k=1)))
    composition = {}
    for deadline, compliance in ((1, 1), (0, 1), (1, 0), (0, 0)):
        composition[f"timely-{deadline}:safe-{compliance}"] = float(
            np.mean((goals[:, 1] == deadline) & (goals[:, 2] == compliance))
        )
    return {
        "sampled_supervised_tuple_count": int(len(goals)),
        "unique_source_transition_count": int(
            len(np.unique(sampled.source_indices))
        ),
        "future_lag_steps": _summary(lags),
        "physical_future_lag_seconds": _summary(lags * NATIVE_DT_S),
        "future_progress_distance": _summary(goals[:, 0] - source_progress),
        "terminal_future_fraction": float(np.mean(sampled.future_terminated)),
        "goal_composition": composition,
        "canonical_target_count": int(np.sum(canonical)),
        "contains_canonical_target": bool(np.any(canonical)),
        "unique_projected_goal_count": int(len(unique)),
        "duplicate_goal_fraction": float(1.0 - len(unique) / len(goals)),
        "pair_collision_fraction": (
            float(collisions / pair_count) if pair_count else 0.0
        ),
        "progress": _summary(goals[:, 0]),
        "minimum_projected_goal_distance_to_canonical": float(
            np.min(np.linalg.norm(goals - canonical_command(), axis=1))
        ),
        "unsafe_fraction": float(np.mean(~safe)),
        "late_fraction": float(np.mean(~timely)),
        "projected_goal_action_ambiguity": _conditional_action_ambiguity(
            goals, sampled.batch.actions
        ),
        "exact_state_goal_action_ambiguity": _conditional_action_ambiguity(
            np.concatenate((sampled.batch.states, goals), axis=1),
            sampled.batch.actions,
        ),
    }


def learning_diagnostics(learner, sampled: SampledSupervision) -> dict[str, Any]:
    """Describe one existing batch without consuming any training RNG."""

    batch = sampled.batch
    states, actions, goals = learner.tensors(batch)
    learner.optimizer.zero_grad(set_to_none=True)
    nll = learner.policy.nll(states, goals, actions)
    loss = nll.mean()
    gradients = torch.autograd.grad(loss, tuple(learner.policy.parameters()))
    with torch.no_grad():
        predicted = learner.policy.deterministic(states, goals)
        mean, log_std = learner.policy(states, goals)
        diagnostic_generator = torch.Generator(device=learner.device)
        diagnostic_generator.manual_seed(0x4743534C)
        entropy = learner.policy.entropy_estimate(
            states, goals, generator=diagnostic_generator
        )
        command = torch.ones_like(goals)
        canonical_actions = learner.policy.deterministic(states, command)
        sampled_actions = predicted
        diagnostic_actions: dict[str, dict[str, float | None]] = {}
        for name, bits in {
            "safe_timely": (1.0, 1.0),
            "safe_late": (0.0, 1.0),
            "unsafe_timely": (1.0, 0.0),
            "unsafe_late": (0.0, 0.0),
        }.items():
            diagnostic_goal = goals.clone()
            diagnostic_goal[:, 1] = bits[0]
            diagnostic_goal[:, 2] = bits[1]
            diagnostic_actions[name] = _summary(
                learner.policy.deterministic(states, diagnostic_goal).cpu().numpy()
            )
    gradient_values = [value.detach().cpu().numpy() for value in gradients]
    nonfinite = int(
        sum(
            value.size - np.count_nonzero(np.isfinite(value))
            for value in gradient_values
        )
        + nll.numel()
        - torch.count_nonzero(torch.isfinite(nll)).item()
    )
    gradient_norm = float(
        torch.sqrt(sum(value.detach().square().sum() for value in gradients)).cpu()
    )
    return {
        "action_nll": float(loss.detach().cpu()),
        "action_prediction_absolute_error": _summary(
            (predicted - actions).abs().cpu().numpy()
        ),
        "predicted_action": _summary(predicted.cpu().numpy()),
        "pre_tanh_mean": _summary(mean.cpu().numpy()),
        "policy_std": _summary(torch.exp(log_std).cpu().numpy()),
        "estimated_policy_entropy": _summary(entropy.cpu().numpy()),
        "parameter_norm": learner.parameter_norm(),
        "gradient_norm": gradient_norm,
        "nonfinite_value_count": nonfinite,
        "goal_dependence": {
            "canonical_action": _summary(canonical_actions.cpu().numpy()),
            "sampled_goal_action": _summary(sampled_actions.cpu().numpy()),
            "canonical_minus_sampled_action": _summary(
                (canonical_actions - sampled_actions).cpu().numpy()
            ),
            "diagnostic_requirement_actions": diagnostic_actions,
        },
    }


@dataclass
class DiagnosticsAccumulator:
    rows: list[dict[str, Any]] = field(default_factory=list)
    seen_source_indices: set[int] = field(default_factory=set)
    total_sampled_tuples: int = 0
    batches_with_canonical_target: int = 0
    canonical_target_count: int = 0
    first_sampled_canonical_target_transition: int | None = None
    nonfinite_value_count: int = 0

    def record(
        self,
        *,
        transition_count: int,
        update_cycle: int,
        sampled: SampledSupervision,
        learning: dict[str, Any],
    ) -> dict[str, Any]:
        tuple_stats = sampled_tuple_diagnostics(sampled)
        self.seen_source_indices.update(map(int, sampled.source_indices))
        self.total_sampled_tuples += tuple_stats["sampled_supervised_tuple_count"]
        self.canonical_target_count += tuple_stats["canonical_target_count"]
        if tuple_stats["contains_canonical_target"]:
            self.batches_with_canonical_target += 1
            if self.first_sampled_canonical_target_transition is None:
                self.first_sampled_canonical_target_transition = transition_count
        self.nonfinite_value_count += learning["nonfinite_value_count"]
        row = {
            "transition_count": int(transition_count),
            "update_cycle": int(update_cycle),
            "hindsight": tuple_stats,
            "learning": learning,
        }
        self.rows.append(row)
        return row

    def summary(self) -> dict[str, Any]:
        batch_count = len(self.rows)
        return {
            "supervised_update_cycles": batch_count,
            "total_sampled_supervised_tuples": self.total_sampled_tuples,
            "unique_source_transition_coverage": len(self.seen_source_indices),
            "batches_with_canonical_target": self.batches_with_canonical_target,
            "fraction_batches_with_canonical_target": (
                self.batches_with_canonical_target / batch_count if batch_count else 0.0
            ),
            "canonical_target_tuple_count": self.canonical_target_count,
            "first_sampled_canonical_target_transition": (
                self.first_sampled_canonical_target_transition
            ),
            "nonfinite_value_count": self.nonfinite_value_count,
        }

    def to_state(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "seen_source_indices": sorted(self.seen_source_indices),
            **self.summary(),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> DiagnosticsAccumulator:
        return cls(
            rows=list(state["rows"]),
            seen_source_indices=set(state["seen_source_indices"]),
            total_sampled_tuples=int(state["total_sampled_supervised_tuples"]),
            batches_with_canonical_target=int(
                state["batches_with_canonical_target"]
            ),
            canonical_target_count=int(state["canonical_target_tuple_count"]),
            first_sampled_canonical_target_transition=state[
                "first_sampled_canonical_target_transition"
            ],
            nonfinite_value_count=int(state["nonfinite_value_count"]),
        )
