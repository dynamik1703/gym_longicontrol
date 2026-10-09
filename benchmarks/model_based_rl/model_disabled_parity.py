"""Bounded proof that model-disabled updates use the untouched V2 learner path."""

from __future__ import annotations

from typing import Any

import numpy as np

from benchmarks.constrained_rl_v2.adapter import build_agent
from benchmarks.constrained_rl_v2.costs import DenseDeadlineTaskWrapper
from benchmarks.scalar_sac.experiment import _base_environment

from .adapter import model_step
from .config import load_configuration
from .fsrl_adapter import model_disabled_update
from .model_state import ModelState, TrackContext
from .physics_model import PhysicsDynamicsModel


def _fill_buffer(buffer, rows) -> None:
    from tianshou.data import Batch

    for row in rows:
        buffer.add(
            Batch(
                obs=row.observation.astype(np.float32),
                act=np.asarray([row.action], dtype=np.float32),
                rew=np.float32(row.objective),
                terminated=row.terminated,
                truncated=row.truncated,
                obs_next=row.next_observation.astype(np.float32),
                info=Batch(cost=np.asarray(row.costs, dtype=np.float32)),
                policy=Batch(),
            )
        )


def run_check() -> dict[str, Any]:
    """Compare one direct frozen update with the model-disabled adapter call."""

    import tianshou
    import torch
    from tianshou.data import ReplayBuffer

    from gym_longicontrol.domain.state import VehicleState

    configuration = load_configuration()
    if tianshou.__version__ != configuration.v2.algorithm.tianshou_version:
        raise RuntimeError("Pinned Tianshou version mismatch")
    environments = [
        DenseDeadlineTaskWrapper(
            _base_environment(configuration.v2),
            task=configuration.v2.task,
            energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
            deadline_normalization_s=configuration.v2.deadline_cost.normalization_s,
        )
        for _ in range(2)
    ]
    try:
        agents = [
            build_agent(
                configuration.v2,
                environment,
                training_seed=117,
                device="cpu",
                threads=1,
            )[0]
            for environment in environments
        ]
        base = environments[0].unwrapped
        context = TrackContext(
            base.track_generator(np.random.default_rng(2000)),
            base.config.sensor_range_m,
            base.config.track_length_m,
            base.energy_factor,
        )
        physics = PhysicsDynamicsModel(base.vehicle, base.config)
        rng = np.random.default_rng(7788)
        rows = []
        for index in range(512):
            state = ModelState(
                VehicleState(
                    position_m=float(index % 800),
                    velocity_m_s=float(1 + index % 30),
                    acceleration_m_s2=float(index % 5 - 2),
                    elapsed_time_s=float(index % 1200) * 0.1,
                ),
                episode_step=index % 1200,
            )
            rows.append(
                model_step(
                    physics,
                    state,
                    float(-1 + 2 * (index % 101) / 100),
                    context=context,
                    vehicle=base.vehicle,
                    config=base.config,
                    energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
                    deadline_s=configuration.v2.task.max_time_s,
                    max_episode_steps=configuration.v2.max_episode_steps,
                    source_real_transition_id=index,
                    rng=rng,
                )
            )
        buffers = [ReplayBuffer(1000), ReplayBuffer(1000)]
        for buffer in buffers:
            _fill_buffer(buffer, rows)
        initial_equal = all(
            torch.equal(left, right)
            for left, right in zip(
                agents[0].policy.parameters(), agents[1].policy.parameters()
            )
        )
        if not initial_equal:
            raise RuntimeError("Same-seed frozen agents did not initialize identically")
        lag_before = [item.get_lag() for item in agents[0].policy.lag_optims]
        np.random.seed(991)
        torch.manual_seed(991)
        agents[0].policy.update(256, buffers[0])
        np.random.seed(991)
        torch.manual_seed(991)
        model_disabled_update(agents[1].policy, buffers[1], 256)
        differences = [
            float(torch.max(torch.abs(left - right)).detach().cpu())
            for left, right in zip(
                agents[0].policy.parameters(), agents[1].policy.parameters()
            )
        ]
        lag_after = [item.get_lag() for item in agents[0].policy.lag_optims]
        maximum = max(differences, default=0.0)
        return {
            "tianshou_version": tianshou.__version__,
            "fsrl_revision": configuration.v2.algorithm.fsrl_git_commit,
            "synthetic_integration_transitions": len(rows),
            "simulator_transitions": 0,
            "direct_update_count": 1,
            "adapter_update_count": 1,
            "n_step": configuration.v2.algorithm.n_step,
            "initial_parameters_equal": initial_equal,
            "maximum_parameter_difference": maximum,
            "pid_before": lag_before,
            "pid_after": lag_after,
            "verified": maximum == 0.0 and lag_before == lag_after,
        }
    finally:
        for environment in environments:
            environment.close()
