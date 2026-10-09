"""Bounded preparation probe (maximum 4,500 Development simulator transitions)."""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

import gym_longicontrol  # noqa: F401
from benchmarks.constrained_rl_v2.costs import DenseDeadlineTaskWrapper
from benchmarks.scalar_sac.experiment import _base_environment
from gym_longicontrol.domain.metrics import _EpisodeMetricsAccumulator
from gym_longicontrol.domain.state import VehicleState

from .adapter import model_step, project_prediction
from .config import load_configuration
from .diagnostics import open_loop_horizon_errors, summarize_predictions
from .fsrl_adapter import mixed_policy_update
from .learned_model import ProbabilisticEnsemble
from .model_state import TrackContext, VehiclePrediction
from .model_training import RealModelDataset, RealModelExample, train_ensemble
from .physics_model import PhysicsDynamicsModel
from .training import state_from_environment


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _action(step: int) -> float:
    phase = step % 240
    if phase < 90:
        return 0.85
    if phase < 130:
        return 0.05
    if phase < 190:
        return -0.75
    return 0.45


def collect_parity_data(
    configuration, transition_cap: int, *, targeted_states: bool = False
):
    development = configuration.raw["preparation"]["allowed_track_seeds"]
    per_seed = transition_cap // len(development)
    examples: list[RealModelExample] = []
    references = []
    contexts: list[TrackContext] = []
    actions: list[float] = []
    observations: list[np.ndarray] = []
    errors = {
        name: []
        for name in (
            "position_m",
            "velocity_m_s",
            "acceleration_m_s2",
            "signed_step_energy_kwh",
            "observation",
            "speed_excess_m_s",
            "deadline_cost",
        )
    }
    terminal_matches = []
    boundary_crossings = 0
    total = 0
    for seed in development:
        environment = DenseDeadlineTaskWrapper(
            _base_environment(configuration.v2),
            task=configuration.v2.task,
            energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
            deadline_normalization_s=configuration.v2.deadline_cost.normalization_s,
        )
        observation, _info = environment.reset(seed=seed)
        local_step = 0
        try:
            while local_step < per_seed:
                base = environment.unwrapped
                if targeted_states:
                    boundaries = tuple(base.track.positions_m[1:])
                    mode = local_step % 12
                    velocity = float(2 + (local_step * 7) % 29)
                    acceleration = float(-2.5 + (local_step % 6))
                    if mode == 0:
                        position = base.config.track_length_m - velocity * 0.05
                    elif boundaries:
                        boundary = float(boundaries[local_step % len(boundaries)])
                        position = max(0.0, boundary - velocity * 0.1)
                    else:
                        position = float((local_step * 17) % 900)
                    elapsed = local_step * base.config.dt_s
                    base.state = VehicleState(
                        position_m=position,
                        velocity_m_s=velocity,
                        acceleration_m_s2=acceleration,
                        elapsed_time_s=elapsed,
                    )
                    base._metrics = _EpisodeMetricsAccumulator()  # noqa: SLF001
                    environment._last_elapsed_time_s = elapsed  # noqa: SLF001
                    observation = base._observation()  # noqa: SLF001
                state = state_from_environment(environment, local_step)
                context = TrackContext(
                    base.track,
                    base.config.sensor_range_m,
                    base.config.track_length_m,
                    base.energy_factor,
                )
                action = _action(local_step)
                physics = PhysicsDynamicsModel(base.vehicle, base.config)
                predicted = model_step(
                    physics,
                    state,
                    action,
                    context=context,
                    vehicle=base.vehicle,
                    config=base.config,
                    energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
                    deadline_s=configuration.v2.task.max_time_s,
                    max_episode_steps=configuration.v2.max_episode_steps,
                    source_real_transition_id=total,
                    rng=np.random.default_rng(0),
                )
                old_limit = context.track.sense(
                    state.vehicle.position_m, context.sensor_range_m
                ).current_limit_m_s
                (
                    next_observation,
                    _reward,
                    terminated,
                    truncated,
                    info,
                ) = environment.step(np.asarray([action], dtype=np.float64))
                next_state = state_from_environment(environment, local_step + 1)
                actual_prediction = VehiclePrediction(
                    next_vehicle=next_state.vehicle,
                    signed_step_energy_kwh=float(info["step_energy_kwh"]),
                )
                actual = project_prediction(
                    state=state,
                    action=action,
                    prediction=actual_prediction,
                    context=context,
                    vehicle=base.vehicle,
                    config=base.config,
                    energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
                    deadline_s=configuration.v2.task.max_time_s,
                    max_episode_steps=configuration.v2.max_episode_steps,
                    source_real_transition_id=total,
                    model_condition="physics",
                    model_version=1,
                )
                errors["position_m"].append(
                    predicted.next_state.vehicle.position_m
                    - next_state.vehicle.position_m
                )
                errors["velocity_m_s"].append(
                    predicted.next_state.vehicle.velocity_m_s
                    - next_state.vehicle.velocity_m_s
                )
                errors["acceleration_m_s2"].append(
                    predicted.next_state.vehicle.acceleration_m_s2
                    - next_state.vehicle.acceleration_m_s2
                )
                errors["signed_step_energy_kwh"].append(
                    predicted.next_state.vehicle.total_energy_kwh
                    - state.vehicle.total_energy_kwh
                    - float(info["step_energy_kwh"])
                )
                errors["observation"].append(
                    float(np.max(np.abs(predicted.next_observation - next_observation)))
                )
                errors["speed_excess_m_s"].append(
                    predicted.costs[0] / base.config.dt_s
                    - float(info["speed_excess_m_s"])
                )
                errors["deadline_cost"].append(
                    predicted.costs[1] - float(info["deadline_deficit_cost_s"])
                )
                terminal_matches.append(
                    (predicted.terminated, predicted.truncated)
                    == (bool(terminated), bool(truncated))
                )
                new_limit = context.track.sense(
                    next_state.vehicle.position_m, context.sensor_range_m
                ).current_limit_m_s
                boundary_crossings += int(old_limit != new_limit)
                examples.append(
                    RealModelExample(
                        total,
                        state,
                        action,
                        next_state,
                        float(info["step_energy_kwh"]),
                    )
                )
                references.append(actual)
                contexts.append(context)
                actions.append(action)
                observations.append(np.asarray(observation))
                observation = next_observation
                total += 1
                local_step += 1
                if terminated or truncated:
                    observation, _info = environment.reset(seed=seed)
                    local_step = 0 if False else local_step
        finally:
            environment.close()
    parity = {
        "simulator_transitions": total,
        "development_track_seeds": development,
        "speed_limit_boundary_crossings": boundary_crossings,
        "maximum_absolute_errors": {
            name: float(np.max(np.abs(values))) for name, values in errors.items()
        },
        "terminal_semantics_accuracy": float(np.mean(terminal_matches)),
        "tolerances": {
            "physical_and_cost_absolute": 1e-12,
            "observation_absolute": 1e-12,
            "termination_exact": True,
        },
        "verified": bool(
            all(np.max(np.abs(values)) <= 1e-12 for values in errors.values())
            and all(terminal_matches)
            and boundary_crossings > 0
        ),
    }
    return parity, examples, references, contexts, actions, observations


def learned_probe(configuration, examples, references, contexts, actions):
    split = int(len(examples) * 0.8)
    dataset = RealModelDataset()
    for example in examples[:split]:
        dataset.append(example)
    simulation_config = __import__(
        "gym_longicontrol.domain.state", fromlist=["SimulationConfig"]
    ).SimulationConfig()
    model = ProbabilisticEnsemble(
        configuration.model, simulation_config, seed=20261009
    )
    rng = np.random.default_rng(20261009)
    report = train_ensemble(model, dataset, rng=rng, maximum_epochs=12)
    vehicle = __import__(
        "gym_longicontrol.domain.vehicle", fromlist=["VehicleModel"]
    ).VehicleModel()
    predicted = [
        model_step(
            model,
            example.state,
            action,
            context=context,
            vehicle=vehicle,
            config=simulation_config,
            energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
            deadline_s=configuration.v2.task.max_time_s,
            max_episode_steps=configuration.v2.max_episode_steps,
            source_real_transition_id=example.transition_id,
            rng=rng,
        )
        for example, context, action in zip(
            examples[split:], contexts[split:], actions[split:]
        )
    ]
    actual = references[split:]
    summary = summarize_predictions(predicted, actual)
    predicted_positions = np.asarray(
        [row.next_state.vehicle.position_m for row in predicted[:50]]
    )
    actual_positions = np.asarray(
        [row.next_state.vehicle.position_m for row in actual[:50]]
    )
    return model, {
        "probe_is_not_main_initialization": True,
        "training_examples": split,
        "heldout_examples": len(examples) - split,
        "maximum_epochs_for_bounded_probe": 12,
        "training_report": asdict(report),
        "heldout_one_step": summary.as_dict(),
        "open_loop_position_rmse_m": open_loop_horizon_errors(
            predicted_positions, actual_positions
        ),
    }


def latency_probe(configuration, model, example, context):
    vehicle = __import__(
        "gym_longicontrol.domain.vehicle", fromlist=["VehicleModel"]
    ).VehicleModel()
    simulation_config = __import__(
        "gym_longicontrol.domain.state", fromlist=["SimulationConfig"]
    ).SimulationConfig()
    physics = PhysicsDynamicsModel(vehicle, simulation_config)
    rng = np.random.default_rng(918273)

    def measure(one_model, iterations):
        started = time.perf_counter()
        for index in range(iterations):
            model_step(
                one_model,
                example.state,
                example.action,
                context=context,
                vehicle=vehicle,
                config=simulation_config,
                energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
                deadline_s=configuration.v2.task.max_time_s,
                max_episode_steps=configuration.v2.max_episode_steps,
                source_real_transition_id=index,
                rng=rng,
            )
        elapsed = time.perf_counter() - started
        return {
            "iterations": iterations,
            "seconds": elapsed,
            "microseconds": elapsed * 1e6 / iterations,
        }

    return {"physics": measure(physics, 5000), "learned": measure(model, 1000)}


def optional_fsrl_probe(configuration, model, examples, references):
    try:
        import fsrl  # noqa: F401
        from tianshou.data import Batch, ReplayBuffer

        from benchmarks.constrained_rl_v2.adapter import build_agent
    except ImportError as error:
        return {"available": False, "reason": str(error)}
    environment = DenseDeadlineTaskWrapper(
        _base_environment(configuration.v2),
        task=configuration.v2.task,
        energy_scale_kwh=configuration.v2.objective.energy_scale_kwh,
        deadline_normalization_s=configuration.v2.deadline_cost.normalization_s,
    )
    try:
        agent, _logger = build_agent(
            configuration.v2,
            environment,
            training_seed=9191,
            device="cpu",
            threads=1,
        )
        policy = agent.policy
        replay = ReplayBuffer(configuration.v2.algorithm.buffer_size)
        for row in references[:1000]:
            replay.add(
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
        synthetic = references[-128:]
        started = time.perf_counter()
        updates = 10
        for _ in range(updates):
            mixed_policy_update(policy, replay, synthetic)
        seconds = time.perf_counter() - started
        policy_parameters = sum(
            value.numel() for value in policy.parameters()
        )
        with tempfile.NamedTemporaryFile(suffix=".pt") as stream:
            import torch

            torch.save(
                {"policy": policy.state_dict(), "model": model.state_dict()},
                stream.name,
            )
            checkpoint_bytes = Path(stream.name).stat().st_size
        return {
            "available": True,
            "updates": updates,
            "seconds": seconds,
            "updates_per_second": updates / seconds,
            "policy_parameter_count": policy_parameters,
            "minimal_policy_plus_model_checkpoint_bytes": checkpoint_bytes,
        }
    finally:
        environment.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("benchmarks/model_based_rl"),
    )
    parser.add_argument("--transition-cap", type=int)
    parser.add_argument("--targeted-states", action="store_true")
    parser.add_argument(
        "--acknowledge-new-interactions",
        action="store_true",
        help="Required for a new bounded probe; never bypasses the 5,000-step cap.",
    )
    args = parser.parse_args(argv)
    configuration = load_configuration()
    transition_cap = (
        configuration.raw["preparation"]["parity_simulator_transitions"]
        if args.transition_cap is None
        else args.transition_cap
    )
    ledger_path = args.output_directory / "preparation_interactions.json"
    already_consumed = 0
    if ledger_path.exists():
        already_consumed = json.loads(ledger_path.read_text(encoding="utf-8"))[
            "simulator_transitions"
        ]
    if transition_cap > 0 and not args.acknowledge_new_interactions:
        raise PermissionError(
            "A new simulator probe requires --acknowledge-new-interactions"
        )
    if already_consumed + transition_cap > 5000:
        raise RuntimeError("Preparation simulator interaction cap would be exceeded")
    (
        parity,
        examples,
        references,
        contexts,
        actions,
        _observations,
    ) = collect_parity_data(
        configuration, transition_cap, targeted_states=args.targeted_states
    )
    if not parity["verified"]:
        raise RuntimeError("Physics model failed preregistered parity tolerances")
    model, learned = learned_probe(
        configuration, examples, references, contexts, actions
    )
    latency = latency_probe(configuration, model, examples[-1], contexts[-1])
    fsrl = optional_fsrl_probe(configuration, model, examples, references)
    model_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.members.parameters()
    )
    resources = {
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "learned_model_parameter_count": model.parameter_count,
        "learned_model_parameter_bytes": model_bytes,
        "transition_latency": latency,
        "synthetic_generation_transitions_per_second": (
            1e6 / latency["learned"]["microseconds"]
        ),
        "fsrl_mixed_update_probe": fsrl,
        "estimated_replay_bytes": {
            "real_rl_100k": 100_000 * 128,
            "real_model_100k": 100_000 * 160,
            "synthetic_100k": 100_000 * 192,
        },
        "estimated_one_policy_hours": None,
        "estimated_six_policy_hours": None,
        "note": (
            "Policy runtime remains unprojected if the pinned FSRL probe is "
            "unavailable."
        ),
    }
    _atomic_json(args.output_directory / "physics_parity.json", parity)
    _atomic_json(args.output_directory / "learned_model_probe.json", learned)
    _atomic_json(args.output_directory / "resource_measurements.json", resources)
    _atomic_json(
        args.output_directory / "preparation_interactions.json",
        {
            "simulator_transitions": parity["simulator_transitions"],
            "maximum_allowed": 5000,
            "validation_transitions": 0,
            "paper_test_transitions": 0,
            "future_main_initialization_reuse": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
