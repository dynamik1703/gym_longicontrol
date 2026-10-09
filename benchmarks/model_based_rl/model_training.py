"""Real-only ensemble data and source-audited holdout/early-stop training."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np

from .learned_model import Normalization, ProbabilisticEnsemble, _torch
from .model_state import ModelState


@dataclass(frozen=True)
class RealModelExample:
    transition_id: int
    state: ModelState
    action: float
    next_state: ModelState
    signed_step_energy_kwh: float
    source: str = "real"

    def __post_init__(self) -> None:
        if self.source != "real":
            raise ValueError("The dynamics model may learn only from real transitions")

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        current = self.state.vehicle
        nxt = self.next_state.vehicle
        return self.state.model_input(self.action), np.asarray(
            [
                nxt.position_m - current.position_m,
                nxt.velocity_m_s - current.velocity_m_s,
                nxt.acceleration_m_s2,
                self.signed_step_energy_kwh,
            ],
            dtype=np.float64,
        )


class RealModelDataset:
    def __init__(self, capacity: int = 100_000):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.examples: list[RealModelExample] = []

    def append(self, example: RealModelExample) -> None:
        if example.source != "real":
            raise ValueError("Synthetic transitions cannot enter model training")
        self.examples.append(example)
        if len(self.examples) > self.capacity:
            del self.examples[: len(self.examples) - self.capacity]

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.examples:
            raise ValueError("Cannot train a model on an empty real dataset")
        pairs = [example.arrays() for example in self.examples]
        return np.stack([pair[0] for pair in pairs]), np.stack(
            [pair[1] for pair in pairs]
        )


@dataclass(frozen=True)
class ModelTrainingReport:
    model_version: int
    real_examples: int
    train_examples: int
    holdout_examples: int
    epochs: tuple[int, ...]
    holdout_losses: tuple[float, ...]
    elite_indices: tuple[int, ...]
    gradient_updates: int
    wall_time_s: float


def train_ensemble(
    ensemble: ProbabilisticEnsemble,
    dataset: RealModelDataset,
    *,
    rng: np.random.Generator,
    maximum_epochs: int | None = None,
) -> ModelTrainingReport:
    """Fit bootstrapped members; holdout membership is fixed for this refresh."""

    torch, _nn = _torch()
    inputs, targets = dataset.arrays()
    count = len(inputs)
    holdout_count = min(
        ensemble.config.maximum_holdout_size,
        max(1, int(round(count * ensemble.config.holdout_fraction))),
    )
    if holdout_count >= count:
        raise ValueError("At least one training and one holdout example are required")
    permutation = rng.permutation(count)
    holdout_ids, train_ids = permutation[:holdout_count], permutation[holdout_count:]
    ensemble.input_normalization = Normalization.fit(inputs[train_ids])
    ensemble.target_normalization = Normalization.fit(targets[train_ids])
    x = ensemble.input_normalization.normalize(inputs).astype(np.float32)
    y = ensemble.target_normalization.normalize(targets).astype(np.float32)
    x_holdout = torch.as_tensor(x[holdout_ids], device=ensemble.device)
    y_holdout = torch.as_tensor(y[holdout_ids], device=ensemble.device)
    epoch_limit = maximum_epochs or ensemble.config.maximum_epochs
    epoch_counts: list[int] = []
    final_losses: list[float] = []
    updates = 0
    started = time.perf_counter()
    for member_index, (member, optimizer) in enumerate(
        zip(ensemble.members, ensemble.optimizers)
    ):
        bootstrap = rng.choice(train_ids, size=len(train_ids), replace=True)
        best_loss = float("inf")
        best_state = copy.deepcopy(member.state_dict())
        stale = 0
        epochs_run = 0
        for epoch in range(epoch_limit):
            order = rng.permutation(bootstrap)
            member.train()
            for start in range(0, len(order), ensemble.config.batch_size):
                ids = order[start : start + ensemble.config.batch_size]
                xb = torch.as_tensor(x[ids], device=ensemble.device)
                yb = torch.as_tensor(y[ids], device=ensemble.device)
                optimizer.zero_grad(set_to_none=True)
                loss = ensemble.negative_log_likelihood(member, xb, yb)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite model loss for member {member_index}"
                    )
                loss.backward()
                optimizer.step()
                updates += 1
            member.eval()
            with torch.no_grad():
                holdout = float(
                    ensemble.negative_log_likelihood(
                        member, x_holdout, y_holdout
                    ).item()
                )
            epochs_run = epoch + 1
            improved = holdout < best_loss * (
                1.0 - ensemble.config.relative_improvement_threshold
            )
            if improved:
                best_loss = holdout
                best_state = copy.deepcopy(member.state_dict())
                stale = 0
            else:
                stale += 1
                if stale > ensemble.config.early_stop_patience:
                    break
        member.load_state_dict(best_state)
        epoch_counts.append(epochs_run)
        final_losses.append(best_loss)
    elites = tuple(
        int(index)
        for index in np.argsort(final_losses)[: ensemble.config.elite_size]
    )
    ensemble.elite_indices = elites
    ensemble.version += 1
    ensemble.training_examples_seen += count
    return ModelTrainingReport(
        model_version=ensemble.version,
        real_examples=count,
        train_examples=len(train_ids),
        holdout_examples=len(holdout_ids),
        epochs=tuple(epoch_counts),
        holdout_losses=tuple(final_losses),
        elite_indices=elites,
        gradient_updates=updates,
        wall_time_s=time.perf_counter() - started,
    )
