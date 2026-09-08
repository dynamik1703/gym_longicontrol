"""Command-line entry point for SAC training and evaluation."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SAC on LongiControl")

    general = parser.add_argument_group("run")
    general.add_argument(
        "--visualize",
        "-vis",
        action="store_true",
        help="Run one episode from a checkpoint instead of training.",
    )
    general.add_argument(
        "--record",
        "-rec",
        action="store_true",
        help="Record visualization/evaluation with Gymnasium RecordVideo.",
    )
    general.add_argument("--save_id", type=int, default=0)
    general.add_argument("--load_id", type=int)
    general.add_argument(
        "--checkpoint",
        type=Path,
        help="Load an explicit checkpoint path, including legacy runs.",
    )
    general.add_argument("--seed", type=int, default=2)
    general.add_argument(
        "--trust_legacy_checkpoint",
        action="store_true",
        help="Allow pickle loading of a trusted old checkpoint.",
    )
    general.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="PyTorch device. 'auto' selects an available accelerator.",
    )
    general.add_argument(
        "--output_dir",
        type=Path,
        help="Output root (default: rl/pytorch/out in the checkout).",
    )

    sac = parser.add_argument_group("SAC")
    sac.add_argument("--replay_buffer_capacity", "-buf", type=int, default=1_000_000)
    sac.add_argument("--buffer_init_fraction", type=float, default=0.1)
    sac.add_argument("--num_epochs", "-ep", type=int, default=10_000)
    sac.add_argument("--num_steps_per_epoch", "-steps", type=int, default=1_000)
    sac.add_argument("--discount_factor_gamma", "-gamma", type=float, default=0.99)
    sac.add_argument("--soft_update_factor_tau", "-tau", type=float, default=0.01)
    sac.add_argument("--optimization_batch", "-batch", type=int, default=256)
    sac.add_argument("--adam_lr", "-lr", type=float, default=0.001)
    sac.add_argument("--hidden_layer_sizes", nargs="+", type=int, default=[64, 64])
    sac.add_argument("--evaluation_interval", type=int, default=10)
    sac.add_argument("--num_evaluation_episodes", type=int, default=10)

    environment = parser.add_argument_group("environment")
    environment.add_argument("--car_id", default="BMW_electric_i3_2014")
    environment.add_argument("--env_id", default="DeterministicTrack-v1")
    environment.add_argument(
        "--reward_weights",
        "-rw",
        nargs="+",
        type=float,
        default=[1.0, 0.5, 1.0, 1.0],
    )
    environment.add_argument("--energy_factor", type=float, default=1.0)
    environment.add_argument(
        "--speed_limit_positions",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75],
    )
    environment.add_argument(
        "--speed_limits", nargs="+", type=int, default=[50, 80, 40, 50]
    )
    return parser


def get_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse arguments without coupling parsing to module import."""

    return build_parser().parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    positive_integer_names = (
        "replay_buffer_capacity",
        "num_epochs",
        "num_steps_per_epoch",
        "optimization_batch",
        "evaluation_interval",
        "num_evaluation_episodes",
    )
    for name in positive_integer_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 <= args.buffer_init_fraction <= 1:
        raise ValueError("buffer_init_fraction must be between zero and one")
    if args.visualize and args.load_id is None and args.checkpoint is None:
        raise ValueError("--visualize requires --load_id or --checkpoint")
    if args.checkpoint is not None and args.load_id is not None:
        raise ValueError("Use either --checkpoint or --load_id")
    if args.optimization_batch > args.replay_buffer_capacity:
        raise ValueError("optimization_batch must not exceed replay_buffer_capacity")
    if args.adam_lr <= 0 or not 0 <= args.discount_factor_gamma <= 1:
        raise ValueError("adam_lr must be positive and gamma in [0, 1]")
    if not 0 <= args.soft_update_factor_tau <= 1:
        raise ValueError("soft_update_factor_tau must be in [0, 1]")
    if not args.hidden_layer_sizes or min(args.hidden_layer_sizes) <= 0:
        raise ValueError("hidden_layer_sizes must contain positive widths")


def _device_from_name(name: str) -> str:
    import torch

    if name != "auto":
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_environment(args: argparse.Namespace, render_mode: str | None = None) -> Any:
    """Create one registered LongiControl Gymnasium environment."""

    gymnasium = importlib.import_module("gymnasium")
    # Importing the package registers its environment IDs.
    importlib.import_module("gym_longicontrol")

    kwargs: dict[str, Any] = {
        "car_id": args.car_id,
        "reward_weights": args.reward_weights,
        "energy_factor": args.energy_factor,
    }
    if args.env_id.startswith("DeterministicTrack"):
        kwargs.update(
            speed_limit_positions=args.speed_limit_positions,
            speed_limits=args.speed_limits,
        )
    elif not args.env_id.startswith("StochasticTrack"):
        raise ValueError(
            "env_id must be a deterministic or stochastic LongiControl track"
        )
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return gymnasium.make(args.env_id, **kwargs)


def _build_agent(args: argparse.Namespace, training_env: Any, evaluation_env: Any):
    import torch

    from .sac import (
        SAC,
        PolicyNetwork,
        QNetwork,
        ReplayBuffer,
        ValueNetwork,
        seed_everything,
    )

    device = _device_from_name(args.device)
    seed_everything(args.seed, training_env)
    if evaluation_env is not training_env:
        for space_name in ("action_space", "observation_space"):
            space = getattr(evaluation_env, space_name, None)
            if space is not None and hasattr(space, "seed"):
                space.seed(args.seed + 1)

    state_dim = int(training_env.observation_space.shape[0])
    action_dim = int(training_env.action_space.shape[0])
    replay_buffer = ReplayBuffer(
        buffer_capacity=args.replay_buffer_capacity,
        batch_size=args.optimization_batch,
        state_dim=state_dim,
        action_dim=action_dim,
        seed=args.seed,
    )
    return SAC(
        environment=training_env,
        evaluation_environment=evaluation_env,
        policy_function=PolicyNetwork(args.hidden_layer_sizes, action_dim, state_dim),
        q1_function=QNetwork(args.hidden_layer_sizes, 1, state_dim + action_dim),
        q2_function=QNetwork(args.hidden_layer_sizes, 1, state_dim + action_dim),
        value_function=ValueNetwork(args.hidden_layer_sizes, 1, state_dim),
        replay_buffer=replay_buffer,
        adam_learning_rate=args.adam_lr,
        target_entropy=-action_dim,
        discount_factor_gamma=args.discount_factor_gamma,
        soft_update_factor_tau=args.soft_update_factor_tau,
        device=torch.device(device),
    )


def _default_output_root() -> Path:
    return Path(__file__).resolve().parents[1] / "rl" / "pytorch" / "out"


def _run_directory(args: argparse.Namespace) -> Path:
    output_root = args.output_dir or _default_output_root()
    run_id = args.load_id if args.load_id is not None else args.save_id
    return Path(output_root) / args.env_id / f"SAC_id{run_id}"


def _empty_history() -> dict[str, Any]:
    return {
        "training_steps": [],
        "steps_per_s": [],
        "eval_return": [],
        "losses": {
            "q1_loss": [],
            "q2_loss": [],
            "value_loss": [],
            "policy_loss": [],
            "alpha_loss": [],
        },
    }


def _complete_history(history: dict[str, Any]) -> dict[str, Any]:
    template = _empty_history()
    for key in ("training_steps", "steps_per_s", "eval_return"):
        history.setdefault(key, template[key])
    losses = history.setdefault("losses", {})
    for name, values in template["losses"].items():
        losses.setdefault(name, values)
    return history


def _write_log(path: Path, line: str, mode: str = "a") -> None:
    with path.open(mode, encoding="utf-8") as output_file:
        print(line, file=output_file)


def _save_history(path: Path, history: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2, default=lambda value: value.item())
        history_file.write("\n")


def run(args: argparse.Namespace) -> Path:
    """Execute one configured training or visualization run."""

    from .checkpoint import load_checkpoint, save_checkpoint
    from .sac import InitPolicy

    _validate_args(args)
    run_directory = _run_directory(args)
    checkpoint_path = run_directory / f"seed{args.seed}.tar"
    if args.load_id is None and checkpoint_path.exists():
        raise FileExistsError(
            f"Run already exists; use --load_id or a new --save_id: {checkpoint_path}"
        )
    if args.load_id is not None and not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    run_directory.mkdir(parents=True, exist_ok=True)
    history_path = run_directory / f"seed{args.seed}.json"
    log_path = run_directory / f"seed{args.seed}.out"
    resolved_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    training_env = make_environment(args)
    evaluation_env = None
    history = _empty_history()
    previous_epoch = 0
    try:
        evaluation_env = make_environment(args)
        agent = _build_agent(args, training_env, evaluation_env)
        if args.load_id is not None or args.checkpoint is not None:
            loaded = load_checkpoint(
                args.checkpoint or checkpoint_path,
                agent,
                map_location=agent.device,
                load_optimizers=not args.visualize,
                trust_legacy=args.trust_legacy_checkpoint,
                expected_config=resolved_config,
            )
            history = _complete_history(loaded.history)
            previous_epoch = loaded.epoch

        if args.visualize:
            render_mode = "rgb_array" if args.record else "human"
            visual_environment = make_environment(args, render_mode=render_mode)
            try:
                agent.do_visualization(
                    record=args.record,
                    save_dname=run_directory,
                    environment=visual_environment,
                    seed=args.seed,
                )
            finally:
                visual_environment.close()
            return run_directory

        fill_size = int(args.replay_buffer_capacity * args.buffer_init_fraction)
        initial_policy = InitPolicy.from_action_space(
            training_env.action_space, seed=args.seed
        )
        if not len(agent.replay_buffer):
            agent.init_replay_buffer(initial_policy, fill_size, seed=args.seed)
        with (run_directory / f"seed{args.seed}.config.json").open("w") as stream:
            json.dump(resolved_config, stream, indent=2)

        header = (
            f"{'epoch':>11}|{'steps':>11}|{'steps/s':>11}|"
            f"{'return':>11}|{'value loss':>11}\n" + 59 * "_"
        )
        _write_log(log_path, header, mode="a" if args.load_id is not None else "w")

        for epoch in range(args.num_epochs):
            absolute_epoch = previous_epoch + epoch + 1
            started_at = time.perf_counter()
            agent.do_training(args.num_steps_per_epoch, seed=args.seed + absolute_epoch)
            elapsed = max(time.perf_counter() - started_at, 1e-9)

            if (
                absolute_epoch % args.evaluation_interval != 0
                and epoch + 1 != args.num_epochs
            ):
                continue
            mean_return, _ = agent.do_evaluation(
                args.num_evaluation_episodes, seed=args.seed
            )
            history["training_steps"].append(absolute_epoch * args.num_steps_per_epoch)
            history["steps_per_s"].append(int(args.num_steps_per_epoch / elapsed))
            history["eval_return"].append(mean_return)
            for name in history["losses"]:
                history["losses"][name].append(agent.losses[name])

            _save_history(history_path, history)
            metadata = {
                "environment_id": args.env_id,
                "seed": args.seed,
                "device": str(agent.device),
                "hidden_layer_sizes": list(args.hidden_layer_sizes),
                "config": resolved_config,
            }
            save_checkpoint(
                checkpoint_path,
                agent,
                epoch=absolute_epoch,
                history=history,
                metadata=metadata,
            )
            log_line = (
                f"{absolute_epoch:>11} {history['training_steps'][-1]:>11} "
                f"{history['steps_per_s'][-1]:>11} "
                f"{history['eval_return'][-1]:>11.2f} "
                f"{history['losses']['value_loss'][-1]:>11.5f}"
            )
            _write_log(log_path, log_line)
            print(log_line, flush=True)

            if args.record:
                video_env = make_environment(args, render_mode="rgb_array")
                try:
                    agent.do_visualization(
                        record=True,
                        save_dname=run_directory,
                        environment=video_env,
                        seed=args.seed,
                    )
                finally:
                    video_env.close()
    finally:
        training_env.close()
        if evaluation_env is not None and evaluation_env is not training_env:
            evaluation_env.close()
    return run_directory


def main(argv: Sequence[str] | None = None) -> int:
    args = get_args(argv)
    run(args)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
