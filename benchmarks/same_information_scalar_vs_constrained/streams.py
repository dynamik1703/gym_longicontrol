"""Exact frozen Route-clean streams viewed through the scalar signal wrapper."""

from __future__ import annotations

from typing import Any

from benchmarks.constrained_v2_collector_replication.streams import (
    CanonicalTrainingTrackWrapper,
    ExplicitFixtureTrackWrapper,
    TrainingRecorder,
)
from benchmarks.scalar_sac.experiment import _base_environment

from .signals import SameInformationScalarWrapper


def _wrap(
    configuration: Any, tracked: Any, max_simulator_steps: int
) -> TrainingRecorder:
    v2 = configuration.constrained_v2
    scalar = SameInformationScalarWrapper(
        tracked,
        task=v2.task,
        energy_scale_kwh=v2.objective.energy_scale_kwh,
        deadline_normalization_s=v2.deadline_cost.normalization_s,
        max_simulator_steps=max_simulator_steps,
    )
    return TrainingRecorder(scalar)


def build_training_environment(
    configuration: Any, *, training_seed: int, max_simulator_steps: int
) -> TrainingRecorder:
    base = _base_environment(configuration.constrained_v2)
    tracked = CanonicalTrainingTrackWrapper(base, int(training_seed))
    return _wrap(configuration, tracked, max_simulator_steps)


def build_fixture_environment(
    configuration: Any, *, reset_seeds: tuple[int, ...], max_simulator_steps: int
) -> TrainingRecorder:
    base = _base_environment(configuration.constrained_v2)
    tracked = ExplicitFixtureTrackWrapper(base, reset_seeds)
    return _wrap(configuration, tracked, max_simulator_steps)
